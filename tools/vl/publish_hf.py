"""Publish the v6 beta GGUFs to Hugging Face from the network volume, one model repo per model.

Runs on a pod with the volume at $VOL (default /workspace) and HF_TOKEN from a RunPod secret, which is never printed.
Each repo is created private and gets its model card from this repository as README.md, then the files in the card's
Files table from the model's folder under $VOL/krummholz-v6 (only LICENSE-base.txt may come from the folder of the run
a fix started from). Files already on Hugging Face with the same contents are skipped, so a rerun resumes. A repo is
made public only when every file there matches the volume: the SHA-256 of each GGUF, which for the weight files must
also be the one SHA256SUMS-gguf recorded when they were quantized, and the git blob of every other file. The report
goes to $VOL/krummholz-v6/hf-publish.json.

    python3 tools/vl/publish_hf.py [--owner krummholz-ai] [--private]
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
FROM_EARLIER_RUN = {"LICENSE-base.txt"}
FILE_ROW = re.compile(r"^\| `([^`]+)` \|", re.M)


def files_table(card):
    """File names in the card's Files table, in order."""
    return FILE_ROW.findall(card.split("\n## Files\n", 1)[1].split("\n## ", 1)[0])


def locate(name, folders):
    """A file from the release folder; only the base license may come from the run a fix started from."""
    search = folders if name in FROM_EARLIER_RUN else folders[:1]
    for folder in search:
        if (V6 / folder / name).is_file():
            return V6 / folder / name
    raise FileNotFoundError(f"{name} not in {', '.join(search)} under {V6}")


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


def publish(api, repo, card_path, folders, public):
    from huggingface_hub.hf_api import RepoFile

    local = {"README.md": card_path}
    local |= {name: locate(name, folders) for name in files_table(card_path.read_text(encoding="utf-8"))}
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
    report = {}
    for name, (card, folders) in MODELS.items():
        repo = f"{args.owner}/{name}"
        try:
            report[repo] = publish(api, repo, ROOT / "docs" / "model-cards" / f"{card}.md", folders, not args.private)
        except Exception as e:  # this repo stays private; the others still go up
            report[repo] = {"url": f"https://huggingface.co/{repo}", "ok": False, "public": False,
                            "problems": [f"{type(e).__name__}: {e}"]}
        print("REPO_OK" if report[repo]["ok"] else "REPO_FAIL", repo, *report[repo]["problems"], flush=True)
        (V6 / "hf-publish.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    done = all(r["ok"] for r in report.values())
    print("PUBLISH_DONE" if done else "PUBLISH_INCOMPLETE", flush=True)
    return 0 if done else 1


if __name__ == "__main__":
    raise SystemExit(main())
