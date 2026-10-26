# Copyright 2026 Kyrylo Bilchenko
# SPDX-License-Identifier: Apache-2.0
"""Experimental vLLM launch with independent BF16, FP8 and FP4 storage controls."""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import sys


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bf16-weights", choices=("off", "on"), default="off")
    parser.add_argument("--fp8-weights", choices=("off", "on"), default="off")
    parser.add_argument("--mxfp4-scales", choices=("off", "on"), default="off")
    parser.add_argument("--bf16-library", type=Path)
    parser.add_argument("--fp8-library", type=Path)
    parser.add_argument("--mxfp4-library", type=Path)
    parser.add_argument("--bf16-target", action="append", default=[], help="Reviewed module-name pattern; repeatable")
    parser.add_argument("--fp8-target", action="append", default=[], help="Reviewed module-name pattern; repeatable")
    parser.add_argument("--min-weight-saving-percent", type=float, default=1)
    parser.add_argument("--min-scale-saving-percent", type=float, default=1)
    parser.add_argument("--audit", action="store_true")
    args, forwarded = parser.parse_known_args()
    if forwarded and forwarded[0] == "--":
        forwarded = forwarded[1:]
    if not forwarded or forwarded[0] != "serve":
        parser.error("Forward 'serve MODEL [vLLM options]' after --")
    os.environ.update(ZIPSERV_MIXED_BF16=args.bf16_weights, ZIPSERV_MIXED_FP8=args.fp8_weights,
        ZIPSERV_MXFP4_SCALES=args.mxfp4_scales, ZIPSERV_BF16_TARGETS=json.dumps(args.bf16_target),
        ZIPSERV_FP8_TARGETS=json.dumps(args.fp8_target),
        ZIPSERV_MIXED_MIN_SAVING_PERCENT=str(args.min_weight_saving_percent),
        ZIPSERV_MXFP4_MIN_SAVING_PERCENT=str(args.min_scale_saving_percent))
    for path, variable in ((args.bf16_library, "ZIPSERV_LIBRARY"), (args.fp8_library, "ZIPSERV_FP8_LIBRARY"),
                           (args.mxfp4_library, "ZIPSERV_MXFP4_LIBRARY")):
        if path is not None:
            os.environ[variable] = str(path.expanduser().resolve())
    from zipserv_vllm.mixed_plugin import settings as mixed_settings
    from zipserv_vllm.mxfp4_plugin import settings as scale_settings
    try:
        mixed_settings(os.environ)
        scale_settings(os.environ)
    except ValueError as error:
        parser.error(str(error))
    if "VLLM_PLUGINS" in os.environ:
        names = [name.strip() for name in os.environ["VLLM_PLUGINS"].split(",") if name.strip()]
        os.environ["VLLM_PLUGINS"] = ",".join(dict.fromkeys([*names, "zipserv_mixed_weights", "zipserv_mxfp4_scales"]))
    if args.audit:
        if any(value.startswith("--worker-extension-cls") for value in forwarded):
            parser.error("--audit cannot replace an existing worker extension")
        forwarded += ["--worker-extension-cls", "zipserv_vllm.mixed_backend.MixedWorker"]
    print("Experimental lossless weight storage selected. Savings thresholds filter storage only; "
          "measure speed and quality for these targets. FP4 code bytes and FP8 scale bytes remain unchanged.", file=sys.stderr)
    sys.argv = ["vllm", *forwarded]
    from vllm.entrypoints.cli.main import main as vllm_main
    vllm_main()


if __name__ == "__main__":
    main()
