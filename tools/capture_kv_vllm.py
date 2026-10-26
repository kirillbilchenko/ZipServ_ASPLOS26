# Copyright 2026 Kyrylo Bilchenko
# SPDX-License-Identifier: Apache-2.0
"""Capture valid, written KV slots from eager vLLM 0.32 FlashInfer inference.

This is an instrumented data-collection run, not a serving-speed benchmark.
No random inputs, allocator padding, warm-up slots or unwritten pages are saved.
Run this module in the vLLM environment; numpy and torch are imported lazily.
"""
from __future__ import annotations

import argparse
from functools import wraps
import hashlib
import json
from pathlib import Path
import re


def select_slots(slots: list[int], block_size: int, max_full_pages: int) -> list[int]:
    """Select evenly spaced complete physical pages, or the valid short tail.

    Return indices into slot_mapping; -1 padding must never enter a sample.
    Selecting by physical page also avoids treating arbitrary token windows as
    complete resident cache pages. Single-request runs prohibit repeated slots.
    """
    if block_size < 1 or max_full_pages < 1:
        raise ValueError("Page size and sample budget must be positive")
    valid = [(i, s) for i, s in enumerate(slots) if s >= 0]
    if len({s for _, s in valid}) != len(valid):
        raise ValueError("Repeated live slots in a single cache write")
    pages: dict[int, list[int]] = {}
    for i, slot in valid:
        pages.setdefault(slot // block_size, []).append(i)
    full = [indices for indices in pages.values() if len(indices) == block_size]
    if full:
        count = min(max_full_pages, len(full))
        picks = ([0] if count == 1 else
                 [i * (len(full) - 1) // (count - 1) for i in range(count)])
        return [idx for page in picks for idx in full[page]]
    return [i for i, _ in valid]


class Recorder:
    def __init__(self, output: str, request: dict, max_full_pages: int):
        self.output = Path(output)
        self.request = request
        self.max_full_pages = max_full_pages
        self.layers: dict[str, dict] = {}

    def record(self, impl, layer, kv_cache, slot_mapping):
        import torch
        if impl.is_kvcache_nvfp4 or impl.kv_sharing_target_layer_name is not None:
            raise RuntimeError("This recorder supports independent BF16/FP8 KV only")
        # Match FlashInferImpl.do_kv_cache_update's cache views in vLLM 0.32.
        if kv_cache.ndim != 4 or kv_cache.shape[-1] != 2 * impl.head_size:
            raise RuntimeError(f"Unrecognised FlashInfer cache layout: {kv_cache.shape}")
        key, value = kv_cache.transpose(1, 2).split(impl.head_size, dim=-1)
        block_size = key.shape[1]
        slots = slot_mapping.detach().cpu().tolist()
        indices = select_slots(slots, block_size, self.max_full_pages)
        if not indices:
            return
        name = layer.layer_name
        state = self.layers.setdefault(name, {
            "key": [], "value": [], "slots": [], "write_sizes": [],
            "seen": set(), "metadata": {
                "layer": name, "cache_shape": list(kv_cache.shape),
                "cache_stride": list(kv_cache.stride()),
                "cache_tensor_dtype": str(kv_cache.dtype),
                "cache_dtype": impl.cache_dtype, "block_size": block_size,
                "num_kv_heads": impl.num_kv_heads, "head_size": impl.head_size,
                "k_scale": layer._k_scale.detach().float().cpu().tolist(),
                "v_scale": layer._v_scale.detach().float().cpu().tolist(),
            },
        })
        chosen = [slots[i] for i in indices]
        if state["seen"].intersection(chosen):
            raise RuntimeError("A sampled slot was overwritten during one request")
        state["seen"].update(chosen)
        selected = torch.tensor(chosen, device=kv_cache.device, dtype=torch.long)
        blocks, offsets = selected // block_size, selected % block_size
        # Read the cache AFTER the real writer, including actual FP8 quantization.
        # Integer views preserve NaNs/signed zero and avoid float8 indexing casts.
        integer_dtype = torch.int16 if kv_cache.element_size() == 2 else torch.uint8
        if kv_cache.element_size() not in (1, 2):
            raise RuntimeError("Expected two-byte BF16 or one-byte FP8 cache")
        for label, tensor in (("key", key), ("value", value)):
            cpu = tensor.view(integer_dtype)[blocks, offsets].contiguous().cpu()
            state[label].append(cpu.numpy())
        state["slots"].extend(chosen)
        state["write_sizes"].extend([sum(s >= 0 for s in slots)] * len(chosen))

    def finish(self) -> dict:
        import numpy as np
        self.output.mkdir(parents=True, exist_ok=True)
        records = []
        for name, state in sorted(self.layers.items()):
            safe = re.sub(r"[^a-zA-Z0-9_.-]", "_", name)
            filename = f"{self.request['id']}--{safe}.npz"
            path = self.output / filename
            np.savez(path, key=np.concatenate(state["key"]),
                     value=np.concatenate(state["value"]),
                     slots=np.asarray(state["slots"], dtype=np.int64),
                     write_sizes=np.asarray(state["write_sizes"], dtype=np.int32))
            records.append({**state["metadata"], "file": filename,
                            "sampled_tokens": len(state["slots"]),
                            "sha256": hashlib.sha256(path.read_bytes()).hexdigest()})
        if not records:
            raise RuntimeError("No real cache writes captured; backend hook did not run")
        result = {"request": self.request, "records": records}
        (self.output / f"{self.request['id']}.json").write_text(
            json.dumps(result, indent=2) + "\n")
        return {"request": self.request["id"], "layers": len(records),
                "sampled_tokens_per_layer": sorted({r["sampled_tokens"] for r in records})}


class CaptureWorker:
    """Worker extension installed only in the dedicated eager capture process."""

    def kv_capture_begin(self, output: str, request: dict, max_full_pages: int):
        from vllm.v1.attention.backends.flashinfer import FlashInferImpl
        if not hasattr(self, "_zipserv_kv_hook_installed"):
            original = FlashInferImpl.do_kv_cache_update
            worker = self

            @wraps(original)
            def write_and_capture(impl, layer, key, value, kv_cache, slot_mapping):
                result = original(impl, layer, key, value, kv_cache, slot_mapping)
                recorder = getattr(worker, "_zipserv_kv_recorder", None)
                if recorder is not None:
                    recorder.record(impl, layer, kv_cache, slot_mapping)
                return result

            FlashInferImpl.do_kv_cache_update = write_and_capture
            self._zipserv_kv_hook_installed = True
        self._zipserv_kv_recorder = Recorder(output, request, max_full_pages)

    def kv_capture_finish(self):
        recorder, self._zipserv_kv_recorder = self._zipserv_kv_recorder, None
        return recorder.finish()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True)
    parser.add_argument("--model-id", required=True)
    parser.add_argument("--revision", required=True)
    parser.add_argument("--requests", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--kv-dtype", choices=("auto", "fp8_e4m3"), required=True)
    parser.add_argument("--pages-per-write", type=int, default=8)
    parser.add_argument("--limit-requests", type=int)
    args = parser.parse_args()
    if args.pages_per_write < 1:
        parser.error("pages-per-write must be positive")
    args.output.mkdir(parents=True, exist_ok=False)
    import torch
    import vllm
    from vllm import LLM, SamplingParams
    from transformers import AutoTokenizer
    if vllm.__version__.split("+")[0] != "0.32.0":
        raise RuntimeError("Cache layout capture is qualified for vLLM 0.32.0 only")
    tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=False)
    # Offline, one request at a time. Capture is enabled only AFTER startup.
    llm = LLM(model=args.model, dtype="bfloat16", trust_remote_code=False,
              kv_cache_dtype=args.kv_dtype,
              kv_cache_memory_bytes=1 << 30, max_model_len=4096,
              max_num_seqs=1, max_num_batched_tokens=4096,
              enable_prefix_caching=False, enforce_eager=True,
              compilation_config={"mode": 0}, seed=42,
              attention_config={"backend": "FLASHINFER"},
              worker_extension_cls="capture_kv_vllm.CaptureWorker")
    provenance = {
        "schema": 1, "model_id": args.model_id, "revision": args.revision,
        "kv_dtype": args.kv_dtype, "model_dtype": "bfloat16",
        "vllm": vllm.__version__, "torch": torch.__version__,
        "cuda": torch.version.cuda, "gpu": torch.cuda.get_device_name(0),
        "sampling": "uniformly spaced complete pages from each cache write; valid short tails separately",
        "pages_per_write": args.pages_per_write,
        "capture_point": "FlashInferImpl.do_kv_cache_update, after stock writer",
        "timing_valid": False,
        "scale_policy": "stock vLLM scales, loaded from checkpoint or default 1.0; values recorded per layer",
        "requests_sha256": hashlib.sha256(args.requests.read_bytes()).hexdigest(),
        "script_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "runs": [],
    }
    requests = json.loads(args.requests.read_text())
    if args.limit_requests:
        requests = requests[:args.limit_requests]
    names = [r["id"] for r in requests]
    if (len(set(names)) != len(names)
            or any(not isinstance(n, str) or not re.fullmatch(r"[A-Za-z0-9_-]+", n) for n in names)):
        raise ValueError("Request IDs must be unique safe filename components")
    for request in requests:
        ids = tokenizer.encode(request["text"], add_special_tokens=False)
        ids = ids[:request["max_prompt_tokens"]]
        info = {k: v for k, v in request.items() if k != "text"}
        info["prompt_tokens"] = len(ids)
        info["token_ids_sha256"] = hashlib.sha256(json.dumps(ids).encode()).hexdigest()
        llm.collective_rpc("kv_capture_begin", args=(str(args.output), info, args.pages_per_write))
        result = llm.generate([{"prompt_token_ids": ids}],
                              SamplingParams(temperature=0, max_tokens=16, ignore_eos=True),
                              use_tqdm=False)
        capture = llm.collective_rpc("kv_capture_finish")
        row = {"request": info, "capture": capture,
               "output_token_ids": result[0].outputs[0].token_ids}
        provenance["runs"].append(row)
        (args.output / "manifest.json").write_text(json.dumps(provenance, indent=2) + "\n")
        print(json.dumps(capture), flush=True)
    print("KV_CAPTURE_COMPLETE", flush=True)


if __name__ == "__main__":
    main()
