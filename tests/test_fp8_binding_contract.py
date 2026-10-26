"""CPU-only checks of the native input-domain contract; GPU tests use real buffers."""
from __future__ import annotations

import importlib.util
from pathlib import Path
import sys
import types
import unittest
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'tools'))
spec = importlib.util.spec_from_file_location('fp8_marlin_contract_test', ROOT / 'tools/fp8_marlin.py')
binding = importlib.util.module_from_spec(spec)
# These tests never execute tensor operations. Avoid requiring a CUDA/PyTorch
# installation just to check whether an unsafe library or payload is rejected.
with patch.dict(sys.modules, {'torch': types.ModuleType('torch')}):
    spec.loader.exec_module(binding)


class Function:
    def __init__(self, value): self.value = value
    def __call__(self, *args): return self.value


class Library:
    def __init__(self, abi=1, compact=None, codec=3, compressed=1):
        for name, value in {'abi': abi, 'compressed': compressed, 'codec': codec,
                            'last_error': b'', 'run': 0, 'decode': 0}.items():
            setattr(self, 'zipserv_marlin_fp8_' + name, Function(value))
        if compact is not None:
            self.zipserv_marlin_fp8_compact_only = Function(compact)


def load(lib, **kwargs):
    with patch.object(binding.ctypes, 'CDLL', return_value=lib):
        return binding.load_library(Path('/unused/native.so'), compressed=True, codec='byteplane', **kwargs)


def owner(raw_tiles=0):
    item = object.__new__(binding.Weights)
    item.data = object()
    item.compressed = True
    item.library = load(Library())
    item.accounting = {'selected': 'byteplane', 'raw_fallback_tiles': raw_tiles}
    return item


class BindingContractTests(unittest.TestCase):
    def test_legacy_mixed_and_explicit_compact_abis(self):
        self.assertFalse(load(Library()).zipserv_compact_only)
        self.assertFalse(load(Library(compact=0)).zipserv_compact_only)
        self.assertTrue(load(Library(abi=2, compact=1)).zipserv_compact_only)

    def test_invalid_abi_capability_combinations_fail_closed(self):
        for abi, capability in ((2, None), (2, 0), (1, 1), (2, 2), (3, 1)):
            with self.subTest(abi=abi, capability=capability), self.assertRaises(ValueError):
                load(Library(abi=abi, compact=capability))
        with self.assertRaises(ValueError): load(Library(abi=2, compact=1, codec=2))

    def test_compact_rebinding_preserves_payload_and_checks_domain(self):
        compact = load(Library(abi=2, compact=1))
        original = owner()
        candidate = original.with_library(compact)
        self.assertIs(candidate.data, original.data)
        self.assertIs(candidate.library, compact)
        self.assertIsNot(candidate.accounting, original.accounting)
        self.assertFalse(original.library.zipserv_compact_only)
        for raw_tiles in (1, 8, None):
            with self.subTest(raw_tiles=raw_tiles), self.assertRaisesRegex(ValueError, 'requires zero raw fallback tiles'):
                owner(raw_tiles).with_library(compact)

    def test_format_mismatch_cannot_rebind(self):
        wrong = load(Library())
        wrong.zipserv_codec_id = 2
        with self.assertRaisesRegex(ValueError, 'format does not match'):
            owner().with_library(wrong)


if __name__ == '__main__':
    unittest.main()
