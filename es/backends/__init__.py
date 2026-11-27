"""Generation backends. Import concrete backends lazily: vLLM is only present on the GPU pod."""
from .base import (AdapterRef, Backend, Prompt, Reply, check_think_tags_survive, make_reply, render_prompt,
                   split_public)


def make_backend(cfg: dict, svd=None) -> Backend:
    """Build a backend from the `backend` config section.

    svd is the LoraSVD (needed only by the mock backend, which decodes candidate
    adapters back into singular values).
    """
    kind = cfg.get("kind")
    if kind == "vllm":
        from .vllm_backend import VLLMBackend
        return VLLMBackend(cfg)
    if kind == "hf":
        from .hf_backend import HFBackend
        return HFBackend(cfg)
    if kind == "mock":
        from .mock_backend import MockBackend
        return MockBackend(cfg, svd)
    raise ValueError(f"unknown backend kind {kind!r}")


__all__ = ["AdapterRef", "Backend", "Prompt", "Reply", "check_think_tags_survive", "make_backend", "make_reply",
           "render_prompt", "split_public"]
