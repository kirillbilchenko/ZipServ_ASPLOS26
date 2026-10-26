"""Check opt-in and mutually exclusive scale ownership without Torch or vLLM."""
from __future__ import annotations

from pathlib import Path
import sys
import tempfile
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "integrations/vllm"))
from zipserv_vllm.mxfp4_plugin import settings


class SettingsTests(unittest.TestCase):
    def test_inert_and_off(self) -> None:
        self.assertIsNone(settings({}))
        self.assertEqual(settings({"ZIPSERV_MXFP4_SCALES": "off"}).mode, "off")

    def test_on_requires_a_real_library_path(self) -> None:
        with self.assertRaises(ValueError):
            settings({"ZIPSERV_MXFP4_SCALES": "on"})
        with tempfile.NamedTemporaryFile() as library:
            parsed = settings({"ZIPSERV_MXFP4_SCALES": "on", "ZIPSERV_MXFP4_LIBRARY": library.name})
            self.assertEqual(parsed.library, Path(library.name).resolve())
            self.assertEqual(parsed.min_saving_percent, 1)

    def test_invalid_modes_and_savings(self) -> None:
        for value in ("1", "auto", "", "ON"):
            with self.assertRaises(ValueError):
                settings({"ZIPSERV_MXFP4_SCALES": value})
        for value in ("nan", "inf", "0", "100", "-1", "invalid"):
            with self.assertRaises(ValueError):
                settings({"ZIPSERV_MXFP4_SCALES": "off", "ZIPSERV_MXFP4_MIN_SAVING_PERCENT": value})

    def test_double_scale_owners_rejected(self) -> None:
        for key, value in (("ZIPSERV_SCALE_DICTIONARY", "1"), ("MXWAVE_GLM53_ENABLE", "1"),
                           ("MXWAVE_MOE_MODE", "aligned"), ("MXWAVE_MOE_MODE", "raw")):
            with self.assertRaises(ValueError):
                settings({"ZIPSERV_MXFP4_SCALES": "off", key: value})


if __name__ == "__main__":
    unittest.main()
