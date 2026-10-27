"""Resolve the active model stack (L1 core plus L2, L3, L4 adapters) and refuse bad ones.

Every layer must be signed by a key the host trusts for that layer, descend from the
same publisher checkpoint, be trained against the running core version, and have
passed its gate. The result is a runtime recipe (llama.cpp arguments); nothing here
loads a model.
"""
from __future__ import annotations

from dataclasses import dataclass
import os

from .manifest import Manifest, ManifestError, sha256_file

ORDER = ("L1", "L2", "L3", "L4")


@dataclass(frozen=True)
class TrustPolicy:
    """Trusted host configuration: which key ids may sign which layer, and the keys."""
    keys: dict           # key_id -> verifier with .verify(manifest)
    allowed: dict        # layer -> frozenset of key_ids

    def check(self, manifest):
        if manifest.key_id not in self.allowed.get(manifest.layer, frozenset()):
            raise ManifestError(f"untrusted_key_for_{manifest.layer}")
        key = self.keys.get(manifest.key_id)
        if key is None or not key.verify(manifest):
            raise ManifestError(f"bad_signature_{manifest.layer}")


@dataclass(frozen=True)
class Layer:
    manifest: Manifest
    path: str
    scale: float = 1.0


@dataclass(frozen=True)
class Stack:
    layers: tuple

    @property
    def core(self):
        return self.layers[0]

    def describe(self):
        """For "how do I know this?": which layers shaped the answer, by name and version."""
        return [{"layer": l.manifest.layer, "name": l.manifest.name, "version": l.manifest.version, "scale": l.scale}
                for l in self.layers]

    def llama_cpp_args(self, mmproj_path=None):
        args = ["-m", self.core.path]
        if mmproj_path:
            args += ["--mmproj", mmproj_path]
        for layer in self.layers[1:]:
            args += ["--lora-scaled", layer.path, f"{layer.scale:g}"]
        return args


def resolve(layers, trust, verify_files=True):
    """layers: iterable of Layer in any order; exactly one L1, at most one of each adapter layer."""
    by = {}
    for layer in layers:
        m = layer.manifest.validate()
        if m.layer in by:
            raise ManifestError(f"duplicate_{m.layer}")
        if not 0.0 < layer.scale <= 2.0:
            raise ManifestError("adapter_scale_out_of_range")
        by[m.layer] = layer
    if "L1" not in by:
        raise ManifestError("missing_core")
    core = by["L1"].manifest
    for name in ORDER:
        if name not in by:
            continue
        layer, m = by[name], by[name].manifest
        trust.check(m)
        if m.gate_decision != "passed":
            raise ManifestError(f"{name}_has_not_passed_gate")
        if m.base_sha256 != core.base_sha256:
            raise ManifestError(f"{name}_descends_from_another_base")
        if m.core_version != core.version:
            raise ManifestError(f"{name}_trained_for_another_core")
        if verify_files:
            if not os.path.isfile(layer.path):
                raise ManifestError(f"{name}_artifact_missing")
            if sha256_file(layer.path) != m.artifact_sha256:
                raise ManifestError(f"{name}_artifact_hash_mismatch")
    product = by.get("L2")
    for name in ("L3", "L4"):
        if name in by and product is not None and by[name].manifest.product != product.manifest.product:
            raise ManifestError(f"{name}_built_for_another_product")
    if "L4" in by and "L2" not in by:
        raise ManifestError("personal_adapter_requires_its_product_adapter")
    if "L3" in by and "L4" in by and by["L4"].manifest.organization != by["L3"].manifest.organization:
        raise ManifestError("personal_adapter_belongs_to_another_organization")
    if "L4" in by and by["L4"].manifest.organization and "L3" not in by:
        raise ManifestError("organization_member_requires_organization_adapter")
    return Stack(tuple(by[n] for n in ORDER if n in by))
