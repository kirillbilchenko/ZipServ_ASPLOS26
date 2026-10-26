# Copyright 2026 Kyrylo Bilchenko
# SPDX-License-Identifier: Apache-2.0
"""Read-only inventory and bounded native-KDA fixtures from the existing CT recipe.

This observer never changes a weight or its consumer. Logical quantization is
reported separately from a tensor's storage dtype: Marlin FP8 codes are int32.
"""
from __future__ import annotations

import hashlib
import inspect
import json
from pathlib import Path

from ct_weight_control import ControlWorker


def type_name(value):
    return type(value).__module__ + "." + type(value).__qualname__


def storage_totals(rows):
    """Count one physical storage once, including aliases across target and draft."""
    seen = {}
    for row in rows:
        key = row["storage_id"]
        size = row["storage_bytes"]
        if type(size) is not int or size < 0:
            raise ValueError("Invalid storage size")
        if key in seen and seen[key] != size:
            raise ValueError("Conflicting sizes for the same storage")
        seen[key] = size
    return {"storages": len(seen), "unique_storage_bytes": sum(seen.values()),
            "tensor_references": len(rows), "logical_tensor_bytes_with_aliases": sum(r["tensor_bytes"] for r in rows)}


def tensor_hash(tensor):
    import torch
    return hashlib.sha256(tensor.detach().contiguous().view(torch.uint8).cpu().numpy()).hexdigest()


class CoverageWorker(ControlWorker):
    """Retain CT's original reports; expose additive read-only inspection RPCs."""

    def zipserv_ct_storage_inventory(self):
        import torch

        recipe = self.ct_weight_control_report()
        if recipe["mode"] != "ct" or recipe["status"] != "verified":
            raise ValueError("Inventory the current CT recipe with its compression retained")
        kda = {row["name"]: row for row in recipe["ct_recipe"]["kda"]["layers"]}
        model = self.model_runner.model
        draft = self.model_runner.speculator.model
        ids, tensors, modules, skipped = {}, [], [], []

        def record(value, owner, name, origin):
            if not isinstance(value, torch.Tensor):
                return
            if value.device.type == "meta" or value.layout != torch.strided:
                skipped.append({"owner": owner, "name": name, "reason": "meta or non-strided tensor"})
                return
            storage = value.untyped_storage()
            identity = (str(value.device), storage._cdata)
            key = ids.setdefault(identity, len(ids))
            tensors.append({"owner": owner, "name": name, "origin": origin,
                "storage_id": key, "storage_bytes": storage.nbytes(), "tensor_bytes": value.nbytes,
                "offset": value.storage_offset(), "shape": list(value.shape), "stride": list(value.stride()),
                "dtype": str(value.dtype), "device": str(value.device), "contiguous": value.is_contiguous()})

        for scope, root in (("target", model), ("draft", draft)):
            for prefix, module in root.named_modules():
                owner = scope + ":" + prefix
                method = getattr(module, "quant_method", None)
                scheme = getattr(module, "scheme", getattr(method, "scheme", None))
                kernel = getattr(scheme, "kernel", None)
                weight = getattr(module, "weight", None)
                entry = {"owner": owner, "class": type_name(module),
                    "quant_method": type_name(method), "scheme": type_name(scheme), "kernel": type_name(kernel),
                    "min_latency_gemm": bool(getattr(module, "_use_min_latency_gemm", False)),
                    "is_converted_kda": scope == "target" and prefix in kda}
                if entry["is_converted_kda"]:
                    entry["declared_kda"] = {k: kda[prefix][k] for k in
                        ("shape", "source_sha256", "rounded_sha256", "resident")}
                if method is not None:
                    try:
                        source = Path(inspect.getfile(type(method)))
                        entry.update(method_source=str(source), method_source_sha256=hashlib.sha256(source.read_bytes()).hexdigest())
                    except (OSError, TypeError):
                        entry["method_source"] = None
                registered = set()
                for origin, values in (("parameter", module.named_parameters(recurse=False, remove_duplicate=False)),
                                       ("buffer", module.named_buffers(recurse=False, remove_duplicate=False))):
                    for name, value in values:
                        registered.add(name)
                        record(value, owner, name, origin)
                # Include direct unregistered tensors, e.g. legacy Marlin workspaces.
                for name, value in vars(module).items():
                    if name not in registered and isinstance(value, torch.Tensor):
                        record(value, owner, name, "direct_attribute")
                if registered or isinstance(weight, torch.Tensor) or entry["is_converted_kda"]:
                    modules.append(entry)
        by_dtype = {}
        for dtype in sorted({row["dtype"] for row in tensors}):
            by_dtype[dtype] = storage_totals([row for row in tensors if row["dtype"] == dtype])
        report = {"status": "complete", "rank": recipe["rank"], "modules": modules,
            "tensors": tensors, "totals": storage_totals(tensors), "by_storage_dtype": by_dtype,
            "skipped": skipped, "ct_recipe": recipe,
            "limits": ["Raw storage dtype is not logical quantization; FP8 Marlin codes appear as int32.",
                       "Per-dtype subtotals can overlap when a storage has views of different dtypes; use the unique total.",
                       "Includes registered parameters/buffers and direct tensor attributes in target/draft module trees.",
                       "Nested Python containers, external native allocations, global head/mHC twins, KV and activation caches require separate accounting.",
                       "Potential coverage only; no additional compression, measured saving, or speed result."]}
        target = Path("/reports") / f"storage-inventory-r{recipe['rank']}.json"
        with target.open("x") as stream:
            json.dump(report, stream, indent=2)
            stream.write("\n")
        return report

    def zipserv_ct_capture_kda(self):
        """Capture each native KDA shard once for tests outside the serving process."""
        import torch
        from safetensors.torch import save_file

        recipe = self.ct_weight_control_report()
        records = recipe["ct_recipe"]["kda"]["layers"]
        if recipe["mode"] != "ct" or len(records) != 68:
            raise ValueError("Require the complete unchanged CT KDA conversion")
        root = Path("/reports") / f"native-kda-r{recipe['rank']}"
        root.mkdir()
        rows = []
        for index, record in enumerate(records):
            layer = self.model_runner.model.get_submodule(record["name"])
            weight, scales = layer.weight, layer.weight_scale
            n, k = weight.shape[1] // 4, weight.shape[0] * 16
            if (weight.dtype != torch.int32 or scales.dtype != torch.float8_e8m0fnu
                    or not weight.is_contiguous() or not scales.is_contiguous()
                    or tuple(scales.shape) != (k // 32, n)
                    or type(layer.quant_method).__name__ != "KdaMxfp8MarlinMethod"):
                raise ValueError("Unexpected native KDA layout or consumer")
            # Each CPU copy is released before the next matrix; no full model clone.
            cpu_weight = weight.detach().cpu().contiguous()
            cpu_scale = scales.detach().view(torch.uint8).cpu().contiguous()
            path = root / f"{index:03d}.safetensors"
            save_file({"weight": cpu_weight, "scale_bytes": cpu_scale}, str(path),
                      metadata={"representation": "CT native Marlin MXFP8", "name": record["name"],
                                "logical_shape": json.dumps(record["shape"])})
            rows.append({"name": record["name"], "path": str(path), "logical_shape": record["shape"],
                "native_shape": [n, k], "weight_sha256": tensor_hash(cpu_weight),
                "scale_sha256": tensor_hash(cpu_scale), "weight_bytes": weight.nbytes,
                "scale_bytes": scales.nbytes, "rounded_sha256": record["rounded_sha256"],
                "source_sha256": record["source_sha256"], "file_sha256": hashlib.sha256(path.read_bytes()).hexdigest()})
            del cpu_weight, cpu_scale
        report = {"status": "complete", "rank": recipe["rank"], "layers": rows,
                  "weight_bytes": sum(row["weight_bytes"] for row in rows),
                  "scale_bytes": sum(row["scale_bytes"] for row in rows)}
        (root / "manifest.json").write_text(json.dumps(report, indent=2) + "\n")
        return report
