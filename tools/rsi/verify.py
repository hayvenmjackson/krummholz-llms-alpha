"""External checks for the RSI loops: math answers compared with the reference, code run against unit tests.

Nothing here is a model. These checks are what keeps a self-improvement loop honest: the policy can write, grade and
filter its own data, but a gain only counts when these functions agree on sealed problems it never trained on.

  extract(text)            final answer: last \\boxed{} or, failing that, the last number
  math_ok(text, ref)       True when the extracted answer equals the reference (math_verify when installed)
  code_block(text)         the last ```python block (or the whole reply when there is none)
  run_tests(code, tests)   (passed, detail): the code and its asserts in a fresh, isolated Python with a time limit
"""
from __future__ import annotations

import os
import re
import subprocess
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "v7"))
from grpo import final_answer, same  # noqa: E402  (the v7 GRPO reward's checks, so both stages agree)

FENCE = re.compile(r"```(?:python|py)?[ \t]*\n(.*?)```", re.S | re.I)


def extract(text: str) -> str:
    return final_answer(text or "")


def norm(ans: str) -> str:
    """A canonical form for counting votes: strips spaces, $, trailing period and \\text{}, and 2.50 -> 2.5."""
    a = re.sub(r"\\text\{([^{}]*)\}", r"\1", str(ans)).replace("$", "").replace(" ", "").rstrip(".").replace(",", "")
    try:
        f = float(a)
        return str(int(f)) if f == int(f) else repr(f)
    except (ValueError, OverflowError):
        return a.lower()


def math_ok(text: str, ref) -> bool:
    pred = extract(text)
    return bool(pred) and (same(pred, str(ref)) or norm(pred) == norm(str(ref)))


def code_block(text: str) -> str:
    blocks = FENCE.findall(text or "")
    return (blocks[-1] if blocks else (text or "")).strip("\n").rstrip()  # keep indentation: a reply may be a body only


def _limits():  # Linux pods only: memory and CPU caps for the child, no core files
    import resource
    resource.setrlimit(resource.RLIMIT_AS, (6 << 30, 6 << 30))  # 2 GB broke numpy imports (HumanEval+ tests use it)
    resource.setrlimit(resource.RLIMIT_CPU, (20, 20))
    resource.setrlimit(resource.RLIMIT_CORE, (0, 0))


def run_tests(code: str, tests: list[str] | str, timeout: float = 10.0) -> tuple[bool, str]:
    """Run model-written code plus its tests in `python -I` inside a temp dir, with no inherited environment.
    This runs the policy's own code on a throwaway pod; it is isolation for robustness, not a security sandbox."""
    tests = tests if isinstance(tests, str) else "\n".join(tests)
    src = code + "\n\n" + tests + "\nprint('__RSI_PASS__')\n"
    with tempfile.TemporaryDirectory() as d:
        p = Path(d) / "t.py"
        p.write_text(src, encoding="utf-8")
        env = {"PATH": os.environ.get("PATH", ""), "PYTHONHASHSEED": "0", "PYTHONDONTWRITEBYTECODE": "1",
               "OPENBLAS_NUM_THREADS": "1", "OMP_NUM_THREADS": "1", "MKL_NUM_THREADS": "1"}
        if os.name == "nt":
            env["SYSTEMROOT"] = os.environ.get("SYSTEMROOT", "")
        try:
            r = subprocess.run([sys.executable, "-I", str(p)], cwd=d, env=env, capture_output=True, text=True,
                               timeout=timeout, preexec_fn=_limits if os.name == "posix" else None)
        except subprocess.TimeoutExpired:
            return False, "timeout"
        ok = r.returncode == 0 and "__RSI_PASS__" in r.stdout
        return ok, "" if ok else (r.stderr.strip().splitlines() or ["exit %d" % r.returncode])[-1][:200]


def humaneval_tests(prompt: str, test: str) -> str:
    """HumanEval+ ships `check(candidate)`; call it on the function the prompt defines (its last def)."""
    names = re.findall(r"^def\s+(\w+)\s*\(", prompt, re.M)
    return test + f"\ncheck({names[-1]})\n" if names else test
