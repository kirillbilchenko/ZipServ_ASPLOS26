# Copyright 2026 Kyrylo Bilchenko
# SPDX-License-Identifier: Apache-2.0
"""Launch vLLM with explicitly enabled or disabled lossless MXFP4 scale storage."""
from __future__ import annotations

import argparse
import os
from pathlib import Path
import sys


def main() -> None:
    """Translate flags into settings inherited by every model worker."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mxfp4-scales", choices=("on", "off"), required=True)
    parser.add_argument("--mxfp4-library", type=Path)
    parser.add_argument("--min-scale-saving-percent", type=float, default=1.0,
                        help="Minimum estimated allocation saving for each eligible scale tensor")
    parser.add_argument("--scale-audit", action="store_true",
                        help="Install the optional read-only per-worker audit extension")
    args, forwarded = parser.parse_known_args()
    if forwarded and forwarded[0] == "--":
        forwarded = forwarded[1:]
    if not forwarded or forwarded[0] != "serve":
        parser.error("Forward 'serve MODEL [vLLM options]' after --")
    os.environ["ZIPSERV_MXFP4_SCALES"] = args.mxfp4_scales
    os.environ["ZIPSERV_MXFP4_MIN_SAVING_PERCENT"] = str(args.min_scale_saving_percent)
    if args.mxfp4_library is not None:
        os.environ["ZIPSERV_MXFP4_LIBRARY"] = str(args.mxfp4_library.expanduser().resolve())
    from zipserv_vllm.mxfp4_plugin import settings
    try:
        settings(os.environ)
    except ValueError as error:
        parser.error(str(error))
    if "VLLM_PLUGINS" in os.environ:
        selected = [name.strip() for name in os.environ["VLLM_PLUGINS"].split(",") if name.strip()]
        os.environ["VLLM_PLUGINS"] = ",".join(dict.fromkeys([*selected, "zipserv_mxfp4_scales"]))
    extension = "zipserv_vllm.mxfp4_backend.ScaleWorker"
    if args.scale_audit:
        # A caller's existing worker extension must compose this mixin itself.
        supplied = [arg for arg in forwarded if arg.startswith("--worker-extension-cls")]
        if supplied:
            parser.error("--scale-audit cannot replace an existing worker extension")
        forwarded.extend(["--worker-extension-cls", extension])
    sys.argv = ["vllm", *forwarded]
    from vllm.entrypoints.cli.main import main as vllm_main
    vllm_main()


if __name__ == "__main__":
    main()
