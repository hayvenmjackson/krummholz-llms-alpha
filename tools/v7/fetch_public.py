"""Download and subset v7's public datasets onto the volume (licenses checked November 5, 2026; see the v7 plan).

  python fetch_public.py --out /workspace/krummholz-v7/public

Writes <name>.jsonl per source with a license field, MANIFEST.json (rows, license, sha256) and SEALED.json: hashes of
every sealed eval question, which training builders must check against so no eval item reaches a training path.
Gated sets (xlam, HarmBench, wildguardmix) are skipped here; they need Hayven's terms acceptance and a separate step.
Large sets are streamed and subset on the pod; nothing large passes through a laptop.
"""
import argparse, hashlib, json, re, traceback
from pathlib import Path

from datasets import load_dataset

def qhash(text):
    return hashlib.sha256(re.sub(r"\W+", " ", text.lower()).strip().encode()).hexdigest()


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--out", required=True); a = ap.parse_args()
    out = Path(a.out); out.mkdir(parents=True, exist_ok=True)
    manifest, sealed = [], {}

    def save(name, license, rows, use, seal_key=None):
        p = out / f"{name}.jsonl"; n = 0
        with p.open("w", encoding="utf-8") as f:
            for r in rows:
                f.write(json.dumps({**r, "license": license}, ensure_ascii=False) + "\n"); n += 1
                if seal_key:
                    sealed.setdefault(name, []).append(qhash(r[seal_key]))
        manifest.append({"name": name, "license": license, "use": use, "rows": n, "sha256": hashlib.sha256(p.read_bytes()).hexdigest()})
        print("SAVED", name, n, flush=True)

    def attempt(name, fn):
        if (out / f"{name}.jsonl").exists():
            print("HAVE", name); return
        try:
            fn()
        except Exception:
            print("FETCH_FAILED", name, flush=True); traceback.print_exc()

    attempt("gsm8k_train", lambda: save("gsm8k_train", "MIT", ({"question": r["question"], "answer": r["answer"].split("####")[-1].strip(), "solution": r["answer"]}
                                                               for r in load_dataset("openai/gsm8k", "main", split="train")), "RL problems"))
    attempt("gsm8k_test", lambda: save("gsm8k_test", "MIT", ({"question": r["question"], "answer": r["answer"].split("####")[-1].strip()}
                                                             for r in load_dataset("openai/gsm8k", "main", split="test")), "sealed eval", "question"))

    def math_train():
        rows = []
        for cfg in ("algebra", "counting_and_probability", "geometry", "intermediate_algebra", "number_theory", "prealgebra", "precalculus"):
            rows += [{"question": r["problem"], "solution": r["solution"], "level": r["level"], "type": cfg}
                     for r in load_dataset("EleutherAI/hendrycks_math", cfg, split="train")]
        save("hendrycks_math_train", "MIT", rows, "RL problems")
    attempt("hendrycks_math_train", math_train)
    attempt("math500", lambda: save("math500", "MIT", ({"question": r["problem"], "answer": r["answer"]}
                                                       for r in load_dataset("HuggingFaceH4/MATH-500", split="test")), "sealed eval", "question"))
    attempt("deepmath", lambda: save("deepmath", "MIT", ({"question": r["question"], "answer": r["final_answer"], "difficulty": r.get("difficulty")}
                                                         for r in load_dataset("zwhe99/DeepMath-103K", split="train")), "RL problems with verified answers"))

    def omr():
        ds = load_dataset("nvidia/OpenMathReasoning", split="cot", streaming=True)
        rows = []
        for r in ds:
            if r.get("problem_type") == "has_answer_extracted" and r.get("generated_solution") and len(r["generated_solution"]) < 6000:
                rows.append({"question": r["problem"], "answer": r.get("expected_answer"), "solution": r["generated_solution"],
                             "generator": r.get("generation_model")})
            if len(rows) >= 50000:
                break
        save("openmathreasoning_50k", "CC BY 4.0", rows, "math SFT (open-weight solutions)")
    attempt("openmathreasoning_50k", omr)

    def ocr():
        ds = load_dataset("nvidia/OpenCodeReasoning", "split_0", split="split_0", streaming=True)
        rows = []
        for r in ds:
            if r.get("solution") and len(r.get("output", "")) < 8000:
                rows.append({"question": r["input"], "solution": r["solution"], "reasoning": r.get("output"), "source": r.get("source")})
            if len(rows) >= 30000:
                break
        save("opencodereasoning_30k", "CC BY 4.0", rows, "code SFT (open-weight outputs)")
    attempt("opencodereasoning_30k", ocr)
    attempt("mbpp_train", lambda: save("mbpp_train", "CC BY 4.0", ({"question": r["text"], "code": r["code"], "tests": r["test_list"]}
                                                                   for r in load_dataset("google-research-datasets/mbpp", split="train")), "code RL with tests"))
    attempt("mbpp_test", lambda: save("mbpp_test", "CC BY 4.0", ({"question": r["text"], "tests": r["test_list"]}
                                                                 for r in load_dataset("google-research-datasets/mbpp", split="test")), "sealed eval", "question"))
    attempt("humanevalplus", lambda: save("humanevalplus", "Apache 2.0", ({"question": r["prompt"], "task_id": r["task_id"], "test": r["test"]}
                                                                          for r in load_dataset("evalplus/humanevalplus", split="test")), "sealed eval", "question"))
    attempt("xstest", lambda: save("xstest", "CC BY 4.0", ({"question": r["prompt"], "type": r.get("type"), "label": r.get("label")}
                                                           for r in load_dataset("Paul/XSTest", split="train")), "sealed eval (over-refusal)", "question"))
    attempt("helpsteer3", lambda: save("helpsteer3", "CC BY 4.0", ({"context": r["context"], "response1": r["response1"], "response2": r["response2"],
                                                                     "preference": r["overall_preference"], "domain": r.get("domain"), "language": r.get("language")}
                                                                    for r in load_dataset("nvidia/HelpSteer3", "preference", split="train")),
                                                       "human preference labels (response sources pending counsel)"))
    attempt("hermes_fc", lambda: save("hermes_fc", "Apache 2.0", ({"conversations": r["conversations"], "category": r.get("category")}
                                                                  for r in load_dataset("NousResearch/hermes-function-calling-v1", "func_calling_singleturn", split="train")),
                                                     "tool-calling formats (generator pending counsel)"))

    prev = json.loads((out / "MANIFEST.json").read_text()) if (out / "MANIFEST.json").exists() else []
    names = {m["name"] for m in manifest}
    (out / "MANIFEST.json").write_text(json.dumps([m for m in prev if m["name"] not in names] + manifest, indent=1))
    old = json.loads((out / "SEALED.json").read_text()) if (out / "SEALED.json").exists() else {}
    (out / "SEALED.json").write_text(json.dumps({**old, **sealed}))
    print("PUBLIC_DONE", len(manifest), "sets;", sum(len(v) for v in sealed.values()), "sealed hashes", flush=True)


if __name__ == "__main__":
    main()
