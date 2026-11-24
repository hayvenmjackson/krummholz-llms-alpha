"""Local binding for the released Lite v8.1 request and reply judge."""

from dataclasses import dataclass
import hashlib
import json
import math
from pathlib import Path
import struct
import subprocess
import threading
from urllib.parse import urlsplit
from urllib.request import HTTPRedirectHandler, ProxyHandler, Request, build_opener


HERE = Path(__file__).resolve().parent
MAX_TEXT_BYTES = 32_768
UNAVAILABLE = "The safety check is unavailable, so I can't show a reply right now."


@dataclass(frozen=True)
class Decision:
    allowed: bool
    stopped_by: str
    rule: str
    text: str
    deferred: bool = False


UNAVAILABLE_DECISION = Decision(False, "judge-unavailable", "", UNAVAILABLE)
UNAVAILABLE_PROBE = ("judge-unavailable", None)


class NoRedirects(HTTPRedirectHandler):
    def redirect_request(self, request, fp, code, msg, headers, newurl):
        return None


def local_origin(url):
    parsed = urlsplit(url)
    if (parsed.scheme != "http" or parsed.hostname != "127.0.0.1" or not parsed.port
            or parsed.username or parsed.password or parsed.path not in {"", "/"}
            or parsed.query or parsed.fragment):
        raise ValueError("A numeric loopback HTTP origin with a port is required")
    return url.rstrip("/")


def post_json(url, data, *, timeout, headers=None, max_bytes=2_000_000):
    request = Request(url, json.dumps(data, ensure_ascii=False).encode("utf-8"),
                      {"Content-Type": "application/json", **(headers or {})})
    opener = build_opener(ProxyHandler({}), NoRedirects())
    with opener.open(request, timeout=timeout) as response:
        body = response.read(max_bytes + 1)
    if len(body) > max_bytes:
        raise ValueError("Oversized local response")
    return json.loads(body)


def verify_assets(assets, *, runtime_only=False):
    for line in (HERE / "SHA256SUMS").read_text(encoding="ascii").splitlines():
        expected, relative = line.split("  ", 1)
        if runtime_only and not relative.endswith((".tsv", ".kzj", ".kze")):
            continue
        actual = hashlib.sha256((assets / relative).read_bytes()).hexdigest()
        if actual != expected:
            raise ValueError(f"Invalid released gate asset: {relative}")


def released_threshold(assets, stage):
    with (assets / "judge" / "models" / f"{stage}.kze").open("rb") as source:
        header = source.read(20)
    if len(header) != 20 or header[:4] != b"KZE1":
        raise ValueError("Invalid released KZE threshold header")
    threshold = struct.unpack_from("<f", header, 16)[0]
    if not math.isfinite(threshold):
        raise ValueError("Invalid released threshold")
    return threshold


class HardGate:
    def __init__(self, *, binary, embedding_url="http://127.0.0.1:8089",
                 assets=HERE / "upstream", process_timeout=90.0, embedding_timeout=8.0,
                 request_threshold=None, deferred_reply_threshold=None):
        self.binary = Path(binary).resolve(strict=True)
        self.assets = Path(assets).resolve(strict=True)
        verify_assets(self.assets)
        self.embedding_url = local_origin(embedding_url) + "/v1/embeddings"
        self.process_timeout = process_timeout
        self.embedding_timeout = embedding_timeout
        if request_threshold is not None or deferred_reply_threshold is not None:
            if (type(request_threshold) not in {int, float} or not math.isfinite(request_threshold)
                    or request_threshold <= released_threshold(self.assets, "request")
                    or type(deferred_reply_threshold) not in {int, float}
                    or not math.isfinite(deferred_reply_threshold)
                    or deferred_reply_threshold >= released_threshold(self.assets, "reply")):
                raise ValueError("Tuning requires both a higher request threshold and a lower deferred reply threshold")
        self.request_threshold = request_threshold
        self.deferred_reply_threshold = deferred_reply_threshold

    def before(self, request):
        return self._run("before", request, threshold=self.request_threshold)

    def after(self, request, reply, *, deferred=False):
        if type(deferred) is not bool or (deferred and self.deferred_reply_threshold is None):
            return UNAVAILABLE_DECISION
        threshold = self.deferred_reply_threshold if deferred else None
        return self._run("after", request, reply, threshold=threshold)

    def request_probe(self, request):
        """Return a rule outcome or judge score for in-process calibration."""
        return self._run("request-probe", request)

    def reply_probe(self, request, reply):
        """Return an output rule outcome or judge score for in-process calibration."""
        return self._run("reply-probe", request, reply)

    def embedding(self, text):
        result = post_json(self.embedding_url, {"model": "bge-small-en-v1.5", "input": text},
                           timeout=self.embedding_timeout, max_bytes=50_000)
        if result["model"] != "bge-small-en-v1.5" or len(result["data"]) != 1:
            raise ValueError("Wrong embedding model or response count")
        values = result["data"][0]["embedding"]
        if (len(values) != 384 or any(type(number) not in {float, int} or not math.isfinite(number)
                                      for number in values)
                or not 0.999 <= math.fsum(number * number for number in values) <= 1.001):
            raise ValueError("Invalid BGE embedding")
        return values

    def _run(self, stage, *texts, threshold=None):
        probe = stage.endswith("-probe")
        unavailable = UNAVAILABLE_PROBE if probe else UNAVAILABLE_DECISION
        try:
            payload = [text.encode("utf-8") for text in texts]
            if any(len(text) > MAX_TEXT_BYTES or b"\0" in text for text in payload):
                return unavailable
            verify_assets(self.assets, runtime_only=True)
            command = [str(self.binary), str(self.assets), stage]
            if threshold is not None:
                command.append(str(threshold))
            with subprocess.Popen(command, stdin=subprocess.PIPE,
                                  stdout=subprocess.PIPE, stderr=subprocess.DEVNULL) as process:
                timer = threading.Timer(self.process_timeout, process.kill)
                timer.daemon = True
                timer.start()
                try:
                    return self._exchange(process, payload, stage, threshold, unavailable)
                finally:
                    timer.cancel()
                    if process.poll() is None:
                        process.kill()
        except (OSError, TypeError, AttributeError, ValueError, OverflowError, subprocess.SubprocessError):
            return unavailable

    def _exchange(self, process, payload, stage, threshold, unavailable):
        process.stdin.write(b"".join(text.hex().encode("ascii") + b"\n" for text in payload))
        process.stdin.flush()
        while line := process.stdout.readline(66_000):
            fields = line.rstrip(b"\r\n").decode("ascii").split("\t")
            if len(fields) == 2 and fields[0] == "EMBED":
                window = bytes.fromhex(fields[1]).decode("utf-8", errors="replace")
                try:
                    embedding = " ".join(format(value, ".9g") for value in self.embedding(window))
                except (OSError, TypeError, KeyError, IndexError, ValueError, OverflowError):
                    embedding = ""
                process.stdin.write(embedding.encode("ascii") + b"\n")
                process.stdin.flush()
                continue
            if stage.endswith("-probe"):
                rule_stage = "input-rule" if stage == "request-probe" else "output-rule"
                if len(fields) == 3 and fields[0] == "SCORE" and process.wait(timeout=2) == 0:
                    if fields[1] == "scored":
                        score = float(fields[2])
                        if math.isfinite(score):
                            return "scored", score
                    elif fields[1] == rule_stage and not fields[2]:
                        return rule_stage, None
                break
            if len(fields) in {5, 6} and fields[0] == "RESULT" and fields[1] in {"0", "1"}:
                _, allowed, stopped_by, rule, text = fields[:5]
                deferred = fields[5:] == ["deferred"]
                denied_stages = ({"input-rule", "request-judge"} if stage == "before" else {"output-rule", "judge"})
                denied_stages.add("judge-unavailable")
                if (allowed == "1" and (stopped_by or rule)) or (allowed == "0" and stopped_by not in denied_stages):
                    break
                if len(fields) == 6 and not (deferred and stage == "before" and threshold is not None
                                            and allowed == "1" and not stopped_by and not rule and not text):
                    break
                if process.wait(timeout=2) == 0:
                    return Decision(allowed == "1", stopped_by, rule, bytes.fromhex(text).decode("utf-8"), deferred)
            break
        return unavailable
