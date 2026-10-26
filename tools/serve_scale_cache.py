"""Start an explicitly selected eager vLLM cache-on/cache-off experiment."""
from __future__ import annotations

import argparse
import os
from pathlib import Path
import sys


def main():
    """Translate experiment flags to worker-inherited settings, preserving vLLM arguments."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--scale-cache", choices=("on", "off"), required=True)
    parser.add_argument("--scale-representation", choices=("dictionary", "raw"), default="dictionary")
    parser.add_argument("--scale-cache-mib", type=int, default=256)
    parser.add_argument("--scale-cache-policy", choices=("retain", "lru"), default="retain")
    parser.add_argument("--dictionary-library", type=Path)
    parser.add_argument("--moe-support", type=Path, help="Existing qualified MxWave model_plugin/moe_runtime directory")
    args, forwarded = parser.parse_known_args()
    if args.scale_representation == "raw" and args.scale_cache != "off":
        parser.error("The raw-scale control requires --scale-cache off")
    if forwarded and forwarded[0] == "--":
        forwarded = forwarded[1:]
    needs_library = args.scale_representation == "dictionary"
    if (not forwarded or forwarded[0] != "serve" or "--enforce-eager" not in forwarded
            or not 0 <= args.scale_cache_mib <= 65536
            or (needs_library and (args.dictionary_library is None or not args.dictionary_library.is_file()))):
        parser.error("Require a dictionary decoder for dictionary mode, 0..65536 MiB, and 'serve MODEL --enforce-eager'")
    extension = "zipserv_vllm.scale_cache_plugin.ScaleCacheWorker"
    if "--worker-extension-cls" in forwarded:
        i = forwarded.index("--worker-extension-cls")
        if i + 1 == len(forwarded) or forwarded[i + 1] != extension:
            parser.error("Use the ScaleCacheWorker extension for this isolated experiment")
    else:
        forwarded.extend(["--worker-extension-cls", extension])
    os.environ.update(ZIPSERV_SCALE_DICTIONARY="1",
                      ZIPSERV_SCALE_REPRESENTATION=args.scale_representation,
                      ZIPSERV_SCALE_CACHE="1" if args.scale_cache == "on" else "0",
                      ZIPSERV_SCALE_CACHE_MIB=str(args.scale_cache_mib),
                      ZIPSERV_SCALE_CACHE_POLICY=args.scale_cache_policy)
    if args.dictionary_library is not None:
        os.environ["ZIPSERV_SCALE_DICTIONARY_LIBRARY"] = str(args.dictionary_library.resolve())
    if "VLLM_PLUGINS" in os.environ:
        selected = os.environ["VLLM_PLUGINS"].split(",")
        os.environ["VLLM_PLUGINS"] = ",".join(dict.fromkeys([*selected, "zipserv_scale_cache"]))
    if args.moe_support:
        if not all((args.moe_support / name).is_file() for name in ("model_plugin.py", "moe_runtime.py")):
            parser.error("--moe-support must contain the qualified model_plugin.py and moe_runtime.py")
        path = str(args.moe_support.resolve())
        os.environ["ZIPSERV_SCALE_BACKEND_MODULE"] = "model_plugin"
        sys.path.insert(0, path)
        os.environ["PYTHONPATH"] = path + os.pathsep + os.environ.get("PYTHONPATH", "")
    if args.scale_representation == "dictionary" and os.environ.get("MXWAVE_MOE_MODE") != "aligned":
        parser.error("Configure the existing aligned-nibble backend: MXWAVE_MOE_MODE=aligned")
    sys.argv = ["vllm", *forwarded]
    from vllm.entrypoints.cli.main import main as vllm_main
    vllm_main()


if __name__ == "__main__":
    main()
