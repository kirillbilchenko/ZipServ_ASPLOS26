"""Exercise real GPU scale-cache hits, evictions, toggles, and cross-stream consumers."""
from __future__ import annotations

import argparse
import ctypes
import hashlib
import json
from pathlib import Path
import sys

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "integrations/vllm"))
from zipserv_vllm.low_precision_codec import encode_container
from zipserv_vllm.scale_cache import DecodedScaleCache, EncodedScale, load_dictionary_library


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--library", type=Path, required=True)
    parser.add_argument("--moe-library", type=Path, required=True)
    parser.add_argument("--report", type=Path, required=True)
    args = parser.parse_args()
    if args.report.exists():
        parser.error("Preserve previous evidence; choose a new report path")
    torch.set_num_threads(2)
    library = load_dictionary_library(args.library)
    native = ctypes.CDLL(str(args.moe_library.resolve()))
    native.mxwave_moe_decode.argtypes = [ctypes.c_void_p, ctypes.c_void_p, ctypes.c_int, ctypes.c_void_p]
    native.mxwave_moe_decode.restype = ctypes.c_int
    report = {"status": "running", "cases": [], "device": torch.cuda.get_device_name(),
              "library_sha256": hashlib.sha256(args.library.read_bytes()).hexdigest(),
              "native_sha256": hashlib.sha256(args.moe_library.read_bytes()).hexdigest()}
    rng = np.random.default_rng(819)
    raw = {key: bytes([0x88 + i]) * size for i, (key, size) in enumerate(
        (("a", 256 * 1024), ("b", 256 * 1024), ("c", 512 * 1024), ("d", 768 * 1024)))}
    raw["d"] = rng.bytes(len(raw["d"]))  # raw fallback alongside dictionary tiles
    expected = {}
    order = np.array([0, 1, 4, 5, 2, 3, 6, 7])
    for key, value in raw.items():
        words = np.frombuffer(value, dtype='<u4')
        expected[key] = (((words[:, None] >> (np.arange(8, dtype=np.uint32) * 4)) & 15)
                         .astype(np.uint8)[:, order] + 112).reshape(-1)
    streams = [torch.cuda.Stream(), torch.cuda.Stream()]
    delay = torch.ones((1024, 1024), device="cuda")
    delays = [torch.empty_like(delay) for _ in streams]
    for stream in streams:
        stream.wait_stream(torch.cuda.current_stream())
    for mode in ("retain", "lru"):
        cache = DecodedScaleCache(768 * 1024, True, mode)
        for key, value in raw.items():
            blob, _ = encode_container(value, "bytes", "vector4-warp", 1024)
            cache.register(key, EncodedScale(library, blob, 112, "cuda:0"))
        for enabled in (False, True, False, True):
            cache.configure(enabled, 768 * 1024, mode)
            outputs = []
            for i, key in enumerate("abacdabacdabcd"):
                output = torch.empty(len(raw[key]) * 2, dtype=torch.uint8, device="cuda")
                stream = streams[i % 2]
                with torch.cuda.stream(stream):
                    with cache.borrow(key) as descriptor:
                        torch.mm(delay, delay, out=delays[i % 2])
                        status = native.mxwave_moe_decode(descriptor.data_ptr(), output.data_ptr(),
                                                          output.numel(), stream.cuda_stream)
                        if status:
                            raise RuntimeError("Native scale consumer failed")
                        output.record_stream(stream)
                outputs.append((key, output))
            torch.cuda.synchronize()
            for key, output in outputs:
                if not np.array_equal(output.cpu().numpy(), expected[key]):
                    raise RuntimeError(f"Cache returned stale or corrupted values: {mode}/{enabled}/{key}")
            state = cache.report()
            assert state['occupied_bytes'] <= state['capacity_bytes']
            assert state['decoded_cache_bytes'] == (768 * 1024 if enabled else 0)
            assert state['decode_workspace_bytes'] == max(map(len, raw.values()))
            assert state['hits'] > 0 if enabled else state['hits'] == 0
            report['cases'].append({'policy': mode, 'enabled': enabled, 'exact': True,
                                    'nondefault_stream_consumers': True, 'cache': state})
            del outputs, output
        # Resetting metrics must preserve the actual admitted content.
        resident = next(iter(cache.policy.entries))
        cache.reset_stats()
        with cache.borrow(resident):
            pass
        assert cache.report()['hits'] == 1
        # Consumers cannot nest leases or silently capture Python cache decisions.
        try:
            graph = torch.cuda.CUDAGraph()
            with torch.cuda.graph(graph):
                with cache.borrow(resident):
                    pass
        except RuntimeError as error:
            assert 'enforce-eager' in str(error), str(error)
        else:
            raise RuntimeError('Cache allowed unsafe graph capture')
        del cache
    cache = DecodedScaleCache(1024)
    blob, _ = encode_container(b'\x88' * 1024, 'bytes', 'vector4-warp', 1024)
    cache.register('a', EncodedScale(library, blob, 112, 'cuda:0'))
    try:
        with cache.borrow('a'):
            raise ValueError('intentional consumer failure')
    except ValueError:
        pass
    try:
        with cache.borrow('a'):
            pass
    except RuntimeError as error:
        assert 'invalid after a failed consumer' in str(error)
    else:
        raise RuntimeError('Cache reused a failed entry')
    torch.cuda.synchronize()
    report.update(status='passed', graph_capture_rejected=True, failed_consumer_invalidates=True,
                  peak_cuda_allocated_bytes=torch.cuda.max_memory_allocated())
    args.report.parent.mkdir(parents=True, exist_ok=True)
    with args.report.open('x') as output:
        json.dump(report, output, indent=2, allow_nan=False)
        output.write('\n')
    print(json.dumps({'status': 'passed', 'cases': len(report['cases']),
                      'peak_mib': report['peak_cuda_allocated_bytes'] / 2**20}), flush=True)


if __name__ == '__main__':
    main()
