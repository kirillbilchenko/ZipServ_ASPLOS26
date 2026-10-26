"""Source-checkout compatibility entry point for the shared CPU reference codec."""
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "integrations" / "vllm"))
from zipserv_vllm.low_precision_codec import *  # noqa: F403
