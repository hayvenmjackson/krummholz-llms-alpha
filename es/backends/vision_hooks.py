"""Apply and restore a candidate's vision delta inside the serving engine, between generation batches.

The text candidates differ only in their LoRA, which vLLM routes per request. The vision candidates
differ in the vision tower's own weights, which are one global state per engine. So for every
candidate whose image prompts are about to run, the driver:

    controller.apply(c)            # one RPC: every worker writes W0 + delta(c) into its weights
    llm.generate(image requests)   # each request carries controller.request_fields(shas)
    controller.restore()           # one RPC: every worker copies the pinned originals back

How the RPC reaches the model (vLLM V1, offline LLM):

  * Preferred: start the engine with `worker_extension_cls="es.backends.vision_hooks.VisionWorkerExtension"`
    (an LLM keyword, set through backend.engine.extra_kwargs in config/atlantic-v9-vision.yaml). vLLM
    mixes the class into its Worker, so `llm.collective_rpc("vision_apply", args=...)` runs the method
    inside the worker process next to the model, which the worker reaches with `self.get_model()`
    (or `self.model_runner.model`).
  * Fallback without the extension: `llm.collective_rpc(rpc_apply, args=...)` sends the module-level
    function (pickled by reference to this module, so the worker must be able to import `es`; the
    engine core inherits the driver's sys.path) and vLLM calls it as `rpc_apply(worker, *args)`.
  * `llm.apply_model(fn)` also works for one-off reads, but it gives the bare model with no place to
    keep the pinned originals between calls, so the hooks keep their state on the worker object.

Only small control data crosses the RPC: the search-space directory path once, then the FP32
coefficient bytes (8 KB for 2,048 coordinates) and a tag per candidate. The worker loads the bases
from disk and pins the original touched tensors once.

Cache coherence. vLLM caches vision encoder outputs by the image's multimodal hash and caches KV
blocks by token ids plus the multimodal hashes of the images they contain; neither key knows the
tower's weights. Two policies, both exact:

  * reset (used when the engine has `reset_encoder_cache`, vLLM main as of 2026-11): after every
    apply and restore the controller clears the encoder cache (`llm.llm_engine.reset_encoder_cache()`)
    and the prefix cache (`llm.reset_prefix_cache()`). Image requests carry the image sha256 as their
    multimodal uuid, so the processor cache of preprocessed pixels is reused across candidates. With
    one candidate per wave, nothing useful survives a wave boundary anyway (every wave has its own
    LoRA, and vLLM keys cached blocks by LoRA), so the resets cost almost nothing.
  * keyed (older vLLM): the multimodal uuid names the vision state, "<image sha256>@<tag>", and a
    `cache_salt` "vision-<tag>" is added, so no encoder output or KV block can be shared across tower
    states without any reset. Costs one image preprocessing per candidate and image.

Text-only requests carry neither field: they never run the tower, so the vision state cannot affect
them. The preflight in VISION.md (a large delta must change image replies; restore must reproduce
the base replies token for token) checks the chosen policy on the installed vLLM.

The Hugging Face backend runs in-process, so HFVisionController patches the model directly. The mock
backend gets RecordingVisionController, which only records the state for loop tests.

Fail closed: attach refuses (and changes nothing) if any touched tensor is missing, sharded, fused
differently, quantized (FP8 base), or differs from the snapshot bytes the space was built from.
"""
from __future__ import annotations

import contextlib
import hashlib
import time
from pathlib import Path
from typing import Dict, Iterator, List, Mapping, Optional, Sequence

import numpy as np

from ..vision import DEFAULT_VISION, VisionPatcher, VisionSpace, deep_merge

WORKER_EXTENSION = "es.backends.vision_hooks.VisionWorkerExtension"
_ATTR = "_krummholz_vision_patcher"


# ======================================================================= worker side
def worker_model(worker):
    """The nn.Module a vLLM worker serves (GPU worker: get_model(); older: model_runner.model)."""
    import torch
    fn = getattr(worker, "get_model", None)
    if callable(fn):
        m = fn()
        if m is not None:
            return m
    runner = getattr(worker, "model_runner", None)
    if runner is not None:
        fn = getattr(runner, "get_model", None)
        if callable(fn):
            return fn()
        if getattr(runner, "model", None) is not None:
            return runner.model
    if isinstance(worker, torch.nn.Module):
        return worker
    raise RuntimeError("cannot find the model inside this vLLM worker")


def _patcher(worker) -> VisionPatcher:
    p = getattr(worker, _ATTR, None)
    if p is None:
        raise RuntimeError("vision hooks are not attached on this worker (call vision_attach first)")
    return p


def _rank(worker) -> int:
    return int(getattr(worker, "rank", 0) or 0)


def rpc_attach(worker, space_dir: str, originals: str = "auto", compute_dtype: str = "float64",
               verify_on_attach: bool = True, audit_restore: bool = True) -> dict:
    old = getattr(worker, _ATTR, None)
    if old is not None:
        old.detach()
        setattr(worker, _ATTR, None)
    space = VisionSpace.load(space_dir)
    patcher = VisionPatcher(worker_model(worker), space, originals=originals, compute_dtype=compute_dtype,
                            verify_on_attach=verify_on_attach, audit_restore=audit_restore)
    setattr(worker, _ATTR, patcher)
    out = patcher.status()
    out["rank"] = _rank(worker)
    return out


def rpc_apply(worker, coeffs: bytes, tag: str, expect_fingerprint: str) -> dict:
    p = _patcher(worker)
    if p.space.fingerprint() != expect_fingerprint:
        raise RuntimeError("worker vision space differs from the driver's; refusing to apply")
    c = np.frombuffer(coeffs, dtype="<f4")
    t0 = time.perf_counter()
    got = p.apply(c, tag)
    return {"rank": _rank(worker), "tag": got, "seconds": time.perf_counter() - t0}


def rpc_restore(worker) -> dict:
    p = _patcher(worker)
    t0 = time.perf_counter()
    p.restore()
    return {"rank": _rank(worker), "tag": p.tag, "seconds": time.perf_counter() - t0}


def rpc_status(worker) -> dict:
    p = getattr(worker, _ATTR, None)
    if p is None:
        return {"rank": _rank(worker), "attached": False}
    out = p.status()
    out.update(rank=_rank(worker), attached=True)
    return out


def rpc_mismatches(worker) -> dict:
    return {"rank": _rank(worker), "mismatches": _patcher(worker).mismatches()}


def rpc_sha256(worker) -> dict:
    """sha256 of every touched tensor as served now (copies them to host; for audits, not per candidate)."""
    p = _patcher(worker)
    return {"rank": _rank(worker), "tag": p.tag, "sha256": p.sha256s()}


def rpc_detach(worker) -> dict:
    p = getattr(worker, _ATTR, None)
    if p is not None:
        p.detach()
        setattr(worker, _ATTR, None)
    return {"rank": _rank(worker), "attached": False}


class VisionWorkerExtension:
    """Mixed into vLLM's Worker with LLM(worker_extension_cls=WORKER_EXTENSION).

    Called by name: llm.collective_rpc("vision_apply", args=(coeff_bytes, tag, fingerprint)).
    """

    def vision_attach(self, space_dir, originals="auto", compute_dtype="float64", verify_on_attach=True,
                      audit_restore=True):
        return rpc_attach(self, space_dir, originals, compute_dtype, verify_on_attach, audit_restore)

    def vision_apply(self, coeffs, tag, expect_fingerprint):
        return rpc_apply(self, coeffs, tag, expect_fingerprint)

    def vision_restore(self):
        return rpc_restore(self)

    def vision_status(self):
        return rpc_status(self)

    def vision_mismatches(self):
        return rpc_mismatches(self)

    def vision_sha256(self):
        return rpc_sha256(self)

    def vision_detach(self):
        return rpc_detach(self)


_RPC = {"attach": rpc_attach, "apply": rpc_apply, "restore": rpc_restore, "status": rpc_status,
        "mismatches": rpc_mismatches, "sha256": rpc_sha256, "detach": rpc_detach}


# ======================================================================= driver side
class VisionController:
    """Driver-side interface shared by the vLLM, Hugging Face and recording controllers."""

    def __init__(self, space: VisionSpace, runtime: Optional[Mapping] = None):
        self.space = space
        self.runtime = deep_merge(DEFAULT_VISION["runtime"], dict(runtime or {}))
        self.tag = "base"
        self.state = "detached"
        self.applies = 0
        self.restores = 0
        self.apply_seconds = 0.0

    def attach(self) -> dict:
        raise NotImplementedError

    def _apply(self, c32: np.ndarray, tag: str) -> None:
        raise NotImplementedError

    def _restore(self) -> None:
        raise NotImplementedError

    def apply(self, c, tag: Optional[str] = None) -> str:
        if self.state == "detached":
            raise RuntimeError("vision controller is not attached")
        c32 = self.space.check_vector(c).numpy().astype("<f4")
        tag = tag or self.space.tag(c32)
        t0 = time.perf_counter()
        self._apply(c32, tag)
        self.apply_seconds += time.perf_counter() - t0
        self.tag, self.state = tag, "candidate"
        self.applies += 1
        return tag

    def restore(self) -> None:
        if self.state == "detached":
            raise RuntimeError("vision controller is not attached")
        t0 = time.perf_counter()
        self._restore()
        self.apply_seconds += time.perf_counter() - t0
        self.tag, self.state = "base", "base"
        self.restores += 1

    @contextlib.contextmanager
    def candidate(self, c, tag: Optional[str] = None) -> Iterator[str]:
        """apply(c) ... restore(), restoring even if generation raises."""
        t = self.apply(c, tag)
        try:
            yield t
        finally:
            self.restore()

    def request_fields(self, image_sha256s: Sequence[str]) -> dict:
        """Extra fields for one image request under the current vision state (vLLM only)."""
        return {}

    def status(self) -> dict:
        return {"kind": type(self).__name__, "state": self.state, "tag": self.tag, "applies": self.applies,
                "restores": self.restores, "apply_restore_seconds": round(self.apply_seconds, 3),
                "fingerprint": self.space.fingerprint(), "dim": self.space.dim}

    def close(self) -> None:
        pass


class VLLMVisionController(VisionController):
    """Drives the worker hooks through llm.collective_rpc and keeps vLLM's caches coherent."""

    def __init__(self, llm, space_dir, runtime: Optional[Mapping] = None):
        super().__init__(VisionSpace.load(space_dir), runtime)
        self.llm = llm
        self.space_dir = str(Path(space_dir).resolve())
        engine = getattr(llm, "llm_engine", None)
        self.use_extension = bool(self.runtime["use_worker_extension"]) and self._extension_installed()
        can_reset = (callable(getattr(engine, "reset_encoder_cache", None))
                     and callable(getattr(llm, "reset_prefix_cache", None)))
        policy = self.runtime["cache_policy"]
        if policy == "auto":
            policy = "reset" if can_reset else "keyed"
        if policy not in ("reset", "keyed"):
            raise ValueError("cache_policy must be auto, reset or keyed")
        if policy == "reset" and not can_reset:
            raise RuntimeError("cache_policy reset needs llm.llm_engine.reset_encoder_cache and "
                               "llm.reset_prefix_cache (newer vLLM); use keyed")
        self.cache_policy = policy
        self.encoder_resets = 0

    def _extension_installed(self) -> bool:
        try:
            cls = self.llm.llm_engine.vllm_config.parallel_config.worker_extension_cls
        except AttributeError:
            return False
        return isinstance(cls, str) and cls.endswith("VisionWorkerExtension")

    def _rpc(self, name: str, *args) -> List[dict]:
        if self.use_extension:
            res = self.llm.collective_rpc(f"vision_{name}", args=args)
        else:
            res = self.llm.collective_rpc(_RPC[name], args=args)
        return list(res)

    def _invalidate(self) -> None:
        if self.cache_policy == "reset":
            self.llm.llm_engine.reset_encoder_cache()
            if self.llm.reset_prefix_cache() is False:
                raise RuntimeError("vLLM refused to reset the prefix cache (requests still running?)")
            self.encoder_resets += 1

    def attach(self) -> dict:
        rt = self.runtime
        res = self._rpc("attach", self.space_dir, rt["originals"], rt["compute_dtype"], bool(rt["verify_on_attach"]),
                        bool(rt["audit_restore"]))
        fps = {r["fingerprint"] for r in res}
        if fps != {self.space.fingerprint()}:
            raise RuntimeError(f"workers loaded different vision spaces: {sorted(fps)}")
        self.state, self.tag = "base", "base"
        self._invalidate()
        return {"workers": res, "cache_policy": self.cache_policy, "use_extension": self.use_extension}

    def _apply(self, c32: np.ndarray, tag: str) -> None:
        res = self._rpc("apply", c32.tobytes(), tag, self.space.fingerprint())
        if {r["tag"] for r in res} != {tag}:
            raise RuntimeError(f"workers report vision tags {res}, expected {tag}")
        self._invalidate()

    def _restore(self) -> None:
        res = self._rpc("restore")
        if {r["tag"] for r in res} != {"base"}:
            raise RuntimeError(f"workers did not restore the base tower: {res}")
        self._invalidate()

    def request_fields(self, image_sha256s: Sequence[str]) -> dict:
        shas = [str(s) for s in image_sha256s]
        if self.cache_policy == "keyed":
            return {"multi_modal_uuids": {"image": [f"{s}@{self.tag}" for s in shas]},
                    "cache_salt": f"vision-{self.tag}"}
        return {"multi_modal_uuids": {"image": shas}}

    def served_sha256(self) -> Dict[str, str]:
        res = self._rpc("sha256")
        first = res[0]["sha256"]
        for r in res[1:]:
            if r["sha256"] != first:
                raise RuntimeError("workers serve different vision weights")
        return first

    def mismatches(self) -> List[str]:
        out: List[str] = []
        for r in self._rpc("mismatches"):
            out.extend(r["mismatches"])
        return sorted(set(out))

    def status(self) -> dict:
        s = super().status()
        s.update(cache_policy=self.cache_policy, use_extension=self.use_extension, encoder_resets=self.encoder_resets)
        return s

    def close(self) -> None:
        if self.state != "detached":
            self._rpc("detach")
            self.state = "detached"


class HFVisionController(VisionController):
    """In-process fallback for the Hugging Face backend (and CPU tests)."""

    def __init__(self, model, space: VisionSpace, runtime: Optional[Mapping] = None):
        super().__init__(space, runtime)
        self.model = model
        self.patcher: Optional[VisionPatcher] = None

    def attach(self) -> dict:
        rt = self.runtime
        self.patcher = VisionPatcher(self.model, self.space, originals=rt["originals"],
                                     compute_dtype=rt["compute_dtype"], verify_on_attach=bool(rt["verify_on_attach"]),
                                     audit_restore=bool(rt["audit_restore"]))
        self.state, self.tag = "base", "base"
        return self.patcher.status()

    def _apply(self, c32: np.ndarray, tag: str) -> None:
        self.patcher.apply(c32, tag)

    def _restore(self) -> None:
        self.patcher.restore()

    def served_sha256(self) -> Dict[str, str]:
        return self.patcher.sha256s()

    def mismatches(self) -> List[str]:
        return self.patcher.mismatches()

    def close(self) -> None:
        if self.patcher is not None:
            self.patcher.detach()
            self.patcher = None
        self.state = "detached"


class RecordingVisionController(VisionController):
    """No model: records the vision state so mock-backend loop tests can read it (self.current)."""

    def __init__(self, space: VisionSpace, runtime: Optional[Mapping] = None):
        super().__init__(space, runtime)
        self.current = np.zeros(space.dim, dtype=np.float32)
        self.history: List[tuple] = []

    def attach(self) -> dict:
        self.state, self.tag = "base", "base"
        return {"kind": "recording", "dim": self.space.dim}

    def _apply(self, c32: np.ndarray, tag: str) -> None:
        self.current = c32.copy()
        self.history.append(("apply", tag))

    def _restore(self) -> None:
        self.current = np.zeros(self.space.dim, dtype=np.float32)
        self.history.append(("restore", "base"))


def make_vision_controller(backend, space_dir, runtime: Optional[Mapping] = None) -> VisionController:
    """Pick the controller for a backend object: vLLM (has .llm), HF (has .model or .base_model), mock."""
    if getattr(backend, "llm", None) is not None:
        return VLLMVisionController(backend.llm, space_dir, runtime)
    model = getattr(backend, "model", None) or getattr(backend, "base_model", None)
    if model is not None:
        return HFVisionController(model, VisionSpace.load(space_dir), runtime)
    return RecordingVisionController(VisionSpace.load(space_dir), runtime)


# ======================================================================= requests
def vision_engine_kwargs(vision_cfg: Mapping, images_per_prompt: int = 1, max_pixels: Optional[int] = None) -> dict:
    """LLM keyword arguments a vision run adds to the text engine's (merge into engine_kwargs)."""
    rt = deep_merge(DEFAULT_VISION["runtime"], dict((vision_cfg or {}).get("runtime") or {}))
    out = {"limit_mm_per_prompt": {"image": int(images_per_prompt), "video": 0}}
    if rt["use_worker_extension"]:
        out["worker_extension_cls"] = WORKER_EXTENSION
    if max_pixels:
        out["mm_processor_kwargs"] = {"size": {"longest_edge": int(max_pixels), "shortest_edge": 65536}}
    return out


def image_messages(text: str, n_images: int = 1, system: Optional[str] = None) -> List[dict]:
    """Chat messages whose user turn holds image parts then the text (the Qwen template renders
    each image part as <|vision_start|><|image_pad|><|vision_end|>; vLLM expands the pad token)."""
    msgs: List[dict] = []
    if system:
        msgs.append({"role": "system", "content": system})
    msgs.append({"role": "user", "content": [{"type": "image"} for _ in range(int(n_images))]
                 + [{"type": "text", "text": text}]})
    return msgs


def image_request(prompt_token_ids: Sequence[int], images: Sequence, fields: Mapping) -> dict:
    """One vLLM TokensPrompt for an image item under the controller's current vision state."""
    req = {"prompt_token_ids": list(prompt_token_ids), "multi_modal_data": {"image": list(images)}}
    req.update(fields)
    return req


def load_image_verified(path, sha256: str):
    """Open an image file only if its bytes match the manifest sha256; returns an RGB PIL image."""
    from PIL import Image
    import io
    data = Path(path).read_bytes()
    got = hashlib.sha256(data).hexdigest()
    if got != sha256:
        raise ValueError(f"image {path} sha256 {got[:12]} does not match the manifest {sha256[:12]}")
    img = Image.open(io.BytesIO(data))
    img.load()
    return img.convert("RGB")
