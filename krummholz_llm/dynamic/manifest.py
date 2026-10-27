"""Layer manifests: strict schema, canonical bytes and pluggable signatures.

A manifest describes one artifact (merged core or LoRA adapter) and how it was made.
Signatures cover the canonical JSON of every field except `signature`. Verifiers are
trusted host configuration; a manifest can never name its own trust.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass, field, fields
import hashlib
import hmac
import json
import re

SCHEMA = "krummholz-layer-manifest-v1"
LAYERS = ("L1", "L2", "L3", "L4")
LAYER_KIND = {"L1": "core", "L2": "product", "L3": "organization", "L4": "personal"}
PRODUCTS = frozenset({"acadia", "katahdin", "atlantic"})
# Adapters may only touch the language model; vision tower and projector stay shared.
LM_TARGETS = frozenset({"q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"})
SHA = re.compile(r"[0-9a-f]{64}")
VERSION = re.compile(r"[0-9]{1,4}\.[0-9]{1,4}\.[0-9]{1,6}")
NAME = re.compile(r"[a-z0-9][a-z0-9._-]{0,63}")


class ManifestError(ValueError):
    pass


@dataclass(frozen=True)
class LoraConfig:
    rank: int
    alpha: int
    targets: tuple[str, ...]

    def validate(self):
        if type(self.rank) is not int or not 1 <= self.rank <= 256:
            raise ManifestError("lora_rank_out_of_range")
        if type(self.alpha) is not int or not 1 <= self.alpha <= 1024:
            raise ManifestError("lora_alpha_out_of_range")
        if not self.targets or not set(self.targets) <= LM_TARGETS or len(set(self.targets)) != len(self.targets):
            raise ManifestError("lora_targets_must_be_language_model_projections")


@dataclass(frozen=True)
class Manifest:
    layer: str
    name: str
    version: str
    base_sha256: str          # the publisher checkpoint (L0) everything descends from
    core_version: str         # Krummholz core this layer was trained against
    artifact_sha256: str      # the GGUF or adapter file this manifest describes
    trainer_version: str
    seed: int
    created_at: str
    key_id: str               # which trusted key must verify this manifest
    parent_version: str | None = None   # previous version of the same layer (L4 chain)
    product: str | None = None          # L2: which product; L3/L4: product they sit on
    organization: str | None = None     # L3, and L4 for members of an organization
    lora: LoraConfig | None = None      # None only for L1 (merged core)
    examples_sha256: str | None = None
    example_count: int = 0
    eval_results: dict = field(default_factory=dict)
    gate_decision: str = "not_run"      # not_run | passed | failed
    schema: str = SCHEMA
    signature: str = ""

    def validate(self):
        if self.schema != SCHEMA:
            raise ManifestError("unknown_manifest_schema")
        if self.layer not in LAYERS:
            raise ManifestError("unknown_layer")
        for value, rule, reason in ((self.name, NAME, "invalid_name"), (self.version, VERSION, "invalid_version"),
                                    (self.core_version, VERSION, "invalid_core_version"),
                                    (self.base_sha256, SHA, "invalid_base_sha256"),
                                    (self.artifact_sha256, SHA, "invalid_artifact_sha256"), (self.key_id, NAME, "invalid_key_id")):
            if type(value) is not str or not rule.fullmatch(value):
                raise ManifestError(reason)
        if self.parent_version is not None and (type(self.parent_version) is not str or not VERSION.fullmatch(self.parent_version)):
            raise ManifestError("invalid_parent_version")
        if self.examples_sha256 is not None and not SHA.fullmatch(self.examples_sha256):
            raise ManifestError("invalid_examples_sha256")
        if type(self.seed) is not int or type(self.example_count) is not int or self.example_count < 0:
            raise ManifestError("invalid_counts")
        if self.gate_decision not in {"not_run", "passed", "failed"}:
            raise ManifestError("invalid_gate_decision")
        if self.layer == "L1":
            if self.lora is not None or self.version != self.core_version:
                raise ManifestError("core_is_merged_and_versions_itself")
        else:
            if self.lora is None:
                raise ManifestError("adapter_requires_lora_config")
            self.lora.validate()
        if self.layer in {"L2", "L3", "L4"} and self.product not in PRODUCTS:
            raise ManifestError("adapter_requires_known_product")
        if self.layer == "L3" and not self.organization:
            raise ManifestError("organization_adapter_requires_organization")
        if self.layer == "L4" and self.examples_sha256 is None:
            raise ManifestError("personal_adapter_requires_example_log_hash")
        return self

    def unsigned_bytes(self):
        value = asdict(self)
        value.pop("signature")
        return canonical(value)

    def to_json(self):
        return json.dumps(asdict(self), sort_keys=True, ensure_ascii=False, indent=1)

    @staticmethod
    def from_json(raw):
        value = json.loads(raw, object_pairs_hook=_no_duplicates)
        if not isinstance(value, dict) or set(value) - {f.name for f in fields(Manifest)}:
            raise ManifestError("unexpected_manifest_fields")
        lora = value.get("lora")
        if lora is not None:
            if not isinstance(lora, dict) or set(lora) != {"rank", "alpha", "targets"}:
                raise ManifestError("invalid_lora_config")
            value["lora"] = LoraConfig(lora["rank"], lora["alpha"], tuple(lora["targets"]))
        try:
            return Manifest(**value).validate()
        except TypeError as error:
            raise ManifestError("missing_manifest_fields") from error


def _no_duplicates(items):
    out = {}
    for key, value in items:
        if key in out:
            raise ManifestError("duplicate_json_key")
        out[key] = value
    return out


def canonical(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode("utf-8")


def sha256_file(path, chunk=1 << 20):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(chunk), b""):
            h.update(block)
    return h.hexdigest()


class HmacKey:
    """Symmetric device key: the device signs and verifies its own personal (L4) layers."""

    def __init__(self, key_id, secret):
        if not NAME.fullmatch(key_id) or len(secret) < 32:
            raise ManifestError("weak_or_invalid_device_key")
        self.key_id, self._secret = key_id, secret

    def sign(self, manifest):
        return "hmac-sha256:" + hmac.new(self._secret, manifest.unsigned_bytes(), hashlib.sha256).hexdigest()

    def verify(self, manifest):
        return hmac.compare_digest(manifest.signature, self.sign(manifest))


class Ed25519PublicKey:
    """Release key for Krummholz (L1, L2) and organization (L3) layers. Needs `cryptography`;
    without it every verification fails closed with a clear reason instead of passing."""

    def __init__(self, key_id, public_bytes):
        if not NAME.fullmatch(key_id) or len(public_bytes) != 32:
            raise ManifestError("invalid_release_key")
        self.key_id, self._raw = key_id, public_bytes

    def verify(self, manifest):
        if not manifest.signature.startswith("ed25519:"):
            return False
        try:
            from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PublicKey as _Key
            from cryptography.exceptions import InvalidSignature
        except ImportError as error:
            raise ManifestError("ed25519_verifier_unavailable") from error
        try:
            _Key.from_public_bytes(self._raw).verify(bytes.fromhex(manifest.signature[8:]), manifest.unsigned_bytes())
            return True
        except (InvalidSignature, ValueError):
            return False


def signed(manifest, key):
    from dataclasses import replace
    unsigned = replace(manifest, signature="")
    return replace(unsigned, signature=key.sign(unsigned))
