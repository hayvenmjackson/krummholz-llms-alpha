"""Evolution strategies over the singular values of a LoRA adapter (ESSA-style).

Modules:
  essa      SVD of each LoRA delta, candidate construction, PEFT adapter writing
  es_loop   generation loop, fitness shaping, optimizer, journal, checkpoint/resume
  fitness   grader adapter interface, retention checks, candidate scoring
  data      training pool and dev loading, deterministic batch sampling
  backends  generation backends (vLLM multi-LoRA, Hugging Face reference, mock)

Evaluation items never live in this repository; configs reference them by path.
"""

__all__ = ["essa", "es_loop", "fitness", "data"]
