"""Publish release GGUFs (v6 beta, v7 Lite) to Hugging Face from the network volume, one model repo per model.

Runs on a pod with the volume at $VOL (default /workspace) and HF_TOKEN from a RunPod secret, which is never printed.
Each repo is created private and gets its model card from this repository as README.md, then the files in the card's
Files table from the model's folder under the release's root on the volume, $VOL/krummholz-v6 or $VOL/krummholz-v7 (only LICENSE-base.txt may come from the folder of the run
a fix started from). Files already on Hugging Face with the same contents are skipped, so a rerun resumes. A repo is
made public only when every file there matches the volume: the SHA-256 of each GGUF, which for the weight files must
also be the one SHA256SUMS-gguf recorded when they were quantized, and the git blob of every other file. The report
goes to <release root>/hf-publish.json.

    python3 tools/vl/publish_hf.py [--release v6|v7] [--owner krummholz-ai] [--private]
"""
import argparse
import hashlib
import json
import os
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
V6 = Path(os.environ.get("VOL", "/workspace")) / "krummholz-v6"
# Hugging Face repo: (model card, folders on the volume, release folder first); smallest upload first
MODELS = {
    "atlantic-lite-4b-v6.3-GGUF": ("atlantic-lite-4b-v6.3", ["atlantic-lite-4b-fix3", "atlantic-lite-4b"]),
    "atlantic-27b-v6-GGUF": ("atlantic-27b-v6", ["atlantic-27b"]),
    "core-35b-a3b-v6.3-GGUF": ("core-35b-a3b-v6.3", ["core-35b-a3b-fix3", "core-35b-a3b"]),
}
# v7: Atlantic Lite after RLAIF round 1 (tools/v7/job_rlaif.sh). The weights come from the RLAIF folder; the base license
# and the vision projector (which neither v7 training nor RLAIF touched) come from the v7 training run it continued.
V7 = V6.parent / "krummholz-v7"
MODELS_V7 = {
    "atlantic-lite-4b-v7-GGUF": ("atlantic-lite-4b-v7", ["rlaif1/atlantic-lite-4b", "atlantic-lite-4b"]),
}
RELEASES = {"v6": (V6, MODELS), "v7": (V7, MODELS_V7)}
FROM_EARLIER_RUN = {"LICENSE-base.txt", "mmproj-krummholz-atlantic-lite-4b-v7-f16.gguf"}
FILE_ROW = re.compile(r"^\| `([^`]+)` \|", re.M)


def files_table(card):
    """File names in the card's Files table, in order."""
    return FILE_ROW.findall(card.split("\n## Files\n", 1)[1].split("\n## ", 1)[0])


def locate(name, folders, root=None):
    """A file from the release folder; only the base license may come from the run a fix started from."""
    root = V6 if root is None else root
    search = folders if name in FROM_EARLIER_RUN else folders[:1]
    for folder in search:
        if (root / folder / name).is_file():
            return root / folder / name
    raise FileNotFoundError(f"{name} not in {', '.join(search)} under {root}")


def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while chunk := f.read(1 << 24):
            h.update(chunk)
    return h.hexdigest()


def git_blob(path):
    data = Path(path).read_bytes()
    return hashlib.sha1(b"blob %d\0" % len(data) + data).hexdigest()


def recorded_sums(path):
    """SHA256SUMS-gguf as {file name: hashes in the order recorded}."""
    sums = {}
    for line in Path(path).read_text(encoding="utf-8").splitlines():
        if line.strip():
            digest, name = line.split(maxsplit=1)
            sums.setdefault(name.lstrip("*"), []).append(digest)
    return sums


def publish(api, repo, card_path, folders, public, root=None):
    from huggingface_hub.hf_api import RepoFile

    local = {"README.md": card_path}
    local |= {name: locate(name, folders, root) for name in files_table(card_path.read_text(encoding="utf-8"))}
    want = {name: (path.stat().st_size, sha256(path) if name.endswith(".gguf") else git_blob(path))
            for name, path in local.items()}
    sums = recorded_sums(local["SHA256SUMS-gguf"])
    api.create_repo(repo, private=True, exist_ok=True)

    def remote():
        return {f.path: (f.size, f.lfs.sha256 if f.lfs else f.blob_id)
                for f in api.list_repo_tree(repo, expand=True) if isinstance(f, RepoFile)}

    have = remote()
    for name, path in local.items():
        if have.get(name) == want[name]:
            print("SKIP", repo, name, flush=True)
            continue
        print("UPLOAD", repo, name, want[name][0], flush=True)
        api.upload_file(path_or_fileobj=str(path), path_in_repo=name, repo_id=repo, commit_message=f"Add {name}")
    have = remote()
    files = [{"file": name, "from": str(local[name]), "bytes": want[name][0],
              "sha256" if name.endswith(".gguf") else "git_blob": want[name][1],
              "ok": have.get(name) == want[name] and (name not in sums or sums[name][-1] == want[name][1])}
             for name in local]
    problems = [f"{f['file']} differs on Hugging Face or from SHA256SUMS-gguf" for f in files if not f["ok"]]
    problems += [f"SHA256SUMS-gguf lists {n}, which is not published" for n in sorted(set(sums) - set(local))]
    problems += [f"unexpected file {n}" for n in sorted(set(have) - set(local) - {".gitattributes"})]
    if public and not problems:
        api.update_repo_settings(repo, private=False)
    return {"url": f"https://huggingface.co/{repo}", "ok": not problems, "public": public and not problems,
            "problems": problems, "files": files}


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--release", choices=sorted(RELEASES), default="v6")
    ap.add_argument("--owner", default="krummholz-ai")
    ap.add_argument("--private", action="store_true", help="leave every repo private")
    args = ap.parse_args()
    from huggingface_hub import HfApi  # imported here so the tests run without it

    api = HfApi()
    who = api.whoami()
    role = who.get("auth", {}).get("accessToken", {}).get("role")
    print("SIGNED_IN", who["name"], "token role", role, flush=True)
    if args.owner not in {who["name"]} | {o["name"] for o in who.get("orgs", [])}:
        raise SystemExit(f"TOKEN_ACCOUNT the token belongs to {who['name']}, not {args.owner}")
    if role == "read":
        raise SystemExit("TOKEN_READ_ONLY the token cannot upload")
    root, models = RELEASES[args.release]
    report = {}
    for name, (card, folders) in models.items():
        repo = f"{args.owner}/{name}"
        try:
            report[repo] = publish(api, repo, ROOT / "docs" / "model-cards" / f"{card}.md", folders, not args.private, root)
        except Exception as e:  # this repo stays private; the others still go up
            report[repo] = {"url": f"https://huggingface.co/{repo}", "ok": False, "public": False,
                            "problems": [f"{type(e).__name__}: {e}"]}
        print("REPO_OK" if report[repo]["ok"] else "REPO_FAIL", repo, *report[repo]["problems"], flush=True)
        (root / "hf-publish.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    done = all(r["ok"] for r in report.values())
    print("PUBLISH_DONE" if done else "PUBLISH_INCOMPLETE", flush=True)
    return 0 if done else 1


if __name__ == "__main__":
    raise SystemExit(main())
