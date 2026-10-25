"""Prevent comparisons with edited questions or the wrong speculative runtime."""
import copy
import json
from pathlib import Path
import sys
import unittest
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "tools"))
import eval_exact_ct_gsm8k as exact


class ProtocolTests(unittest.TestCase):
    def setUp(self):
        self.parent = json.dumps({"cases": [{"question": "fixed"}], "limits": [], "temperature": 0,
                                  "concurrency": 4, "seed": 42, "max_tokens": 1024}).encode()
        self.launch = {"image": exact.IMAGE, "rank": 0, "mode": "off", "returncode": 0,
            "preflight": False, "cmd": ["serve", "/model", "--max-model-len", "500000",
            "--kv-cache-memory-bytes", str(6 * 1024**3), "--max-num-seqs", "32",
            "--max-num-batched-tokens", "6919", "--kv-cache-dtype", "fp8_e4m3",
            "--enable-prefix-caching", "--worker-extension-cls", "ct_weight_control.ControlWorker",
            "--speculative-config", '{"method":"dflash"}'], "host_config": {},
            "sources": {"runtime.py": "hash"}, "tools": {},
            "env": {"MXWAVE_COMPARE_ARM": "A", "GLM_DENSE_FAST": "1"}}

    def test_question_edit_and_scoring_change_are_rejected(self):
        with patch.object(exact, "PARENT_SHA", exact.sha(self.parent)):
            protocol = exact.freeze(self.parent, self.launch)
            exact.validate(protocol, self.parent, self.launch)
            for key, value in (("cases", [{"question": "changed"}]), ("temperature", 1)):
                edited = {**protocol, key: value}
                with self.assertRaises(ValueError):
                    exact.validate(edited, self.parent, self.launch)

    def test_weight_arm_can_change_but_compute_source_cannot(self):
        with patch.object(exact, "PARENT_SHA", exact.sha(self.parent)):
            protocol = exact.freeze(self.parent, self.launch)
            other = copy.deepcopy(self.launch)
            other["mode"] = "ct"
            other["env"]["MXWAVE_COMPARE_ARM"] = "B"
            exact.validate(protocol, self.parent, other)
            other["sources"]["runtime.py"] = "changed"
            with self.assertRaises(ValueError):
                exact.validate(protocol, self.parent, other)

    def test_missing_speculation_and_wrong_image_are_rejected(self):
        other = copy.deepcopy(self.launch)
        other["cmd"] = other["cmd"][:-2]
        with self.assertRaises(ValueError):
            exact.contract(other)
        other = {**self.launch, "image": "vllm032"}
        with self.assertRaises(ValueError):
            exact.contract(other)

    def test_changed_parent_is_rejected(self):
        with self.assertRaises(ValueError):
            exact.freeze(b"edited dataset", self.launch)


if __name__ == "__main__":
    unittest.main()
