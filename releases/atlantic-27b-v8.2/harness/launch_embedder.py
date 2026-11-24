"""Start a CPU-only, loopback BGE server from the pinned F32 embedding file."""

import argparse
from hashlib import file_digest
from pathlib import Path
import subprocess


BGE_NAME = "bge-small-en-v1.5"
BGE_SHA256 = "bf40c42ad7d89382e9ba7376d5c4b73f6b556cb541fab37aaa1da9c320149b65"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--llama-server", required=True, type=Path)
    parser.add_argument("--model-file", type=Path, default=Path(__file__).resolve().parent / "assets" / f"{BGE_NAME}-f32.gguf")
    parser.add_argument("--port", type=int, default=8089)
    parser.add_argument("--threads", type=int, default=2)
    args = parser.parse_args()
    with args.model_file.open("rb") as file:
        if file_digest(file, "sha256").hexdigest() != BGE_SHA256:
            raise SystemExit("BGE file does not match the pinned F32 model")
    command = [str(args.llama_server.resolve(strict=True)), "--model", str(args.model_file.resolve()),
               "--alias", BGE_NAME, "--embedding", "--pooling", "cls", "--n-gpu-layers", "0",
               "--host", "127.0.0.1", "--port", str(args.port), "--threads", str(args.threads),
               "--parallel", "1", "--log-disable"]
    raise SystemExit(subprocess.call(command))


if __name__ == "__main__":
    main()
