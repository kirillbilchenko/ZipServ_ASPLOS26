"""Avoid double-counting aliased weights in incremental CT saving estimates."""
from pathlib import Path
import sys
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "tools"))
from ct_storage_audit import storage_totals


class StorageTotalsTests(unittest.TestCase):
    def test_target_draft_alias_is_counted_once(self):
        rows = [{"storage_id": 0, "storage_bytes": 4096, "tensor_bytes": 2048},
                {"storage_id": 0, "storage_bytes": 4096, "tensor_bytes": 4096},
                {"storage_id": 1, "storage_bytes": 128, "tensor_bytes": 128}]
        result = storage_totals(rows)
        self.assertEqual(result["unique_storage_bytes"], 4224)
        self.assertEqual(result["logical_tensor_bytes_with_aliases"], 6272)
        self.assertEqual(result["storages"], 2)

    def test_conflicting_alias_sizes_are_rejected(self):
        with self.assertRaises(ValueError):
            storage_totals([{"storage_id": 0, "storage_bytes": 4, "tensor_bytes": 4},
                            {"storage_id": 0, "storage_bytes": 8, "tensor_bytes": 4}])


if __name__ == "__main__":
    unittest.main()
