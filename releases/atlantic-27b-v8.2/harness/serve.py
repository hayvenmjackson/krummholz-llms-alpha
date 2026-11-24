"""Non-streaming OpenAI-compatible text endpoint with the Lite v8.1 safety gate."""

import argparse
import json
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import os
from pathlib import Path
import re
from urllib.parse import urlsplit

from hard_gate import Decision, HardGate, MAX_TEXT_BYTES, UNAVAILABLE, local_origin, post_json


THINK_BLOCK = re.compile(r"<think\b[^>]*>.*?</think\s*>", re.I | re.S)
THINK_TAG = re.compile(r"</?think\b[^>]*>", re.I)
INKLING_THINK_BLOCK = re.compile(r"(?:<\|message_model\|>)?<\|content_thinking\|>.*?<\|end_message\|>", re.S)
INKLING_TOOL = re.compile(r"(?:<\|message_model\|>[^<]*)?<\|content_invoke_tool_json\|>(.*?)<\|end_message\|>", re.S)
INKLING_TEXT_MARKERS = re.compile(r"<\|(?:message_model|content_text|end_message|content_model_end_sampling)\|>")
INKLING_PRIVATE_MARKER = re.compile(r"(?:<\|content_thinking\|>|<\|channel\|>\s*analysis)", re.I)
INLINE_TOOL_MARKER = re.compile(r"<\s*/?\s*tool_call(?:\s|>)", re.I)
READONLY_TOOL_VERSION = "krummholz-readonly-tools-v2"
ALLOWED_ROLES = {"system", "developer", "user", "assistant", "tool"}
REQUEST_FIELDS = {"model", "messages", "stream", "n", "max_tokens", "max_completion_tokens", "temperature",
                  "top_p", "top_k", "min_p", "seed", "stop", "tools", "tool_choice", "parallel_tool_calls",
                  "reasoning_effort", "chat_template_kwargs", "response_format", "presence_penalty", "frequency_penalty"}


def as_json(value):
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def tool_names(tools):
    if not isinstance(tools, list):
        raise ValueError("Tools must be an array of function definitions")
    names = set()
    for tool in tools:
        if tool["type"] != "function":
            raise ValueError("Only function tools are supported")
        name = tool["function"]["name"]
        if not isinstance(name, str) or not re.fullmatch(r"[A-Za-z0-9_-]{1,64}", name) or name in names:
            raise ValueError("Invalid tool name")
        names.add(name)
    return names


def safe_tool_calls(calls, names):
    if not isinstance(calls, list):
        raise ValueError("Invalid tool calls")
    result = []
    for call in calls:
        function = call["function"]
        name, arguments, identifier = function["name"], function["arguments"], call["id"]
        if (call["type"] != "function" or name not in names or not isinstance(identifier, str)
                or not re.fullmatch(r"[A-Za-z0-9_-]{1,128}", identifier)
                or not isinstance(arguments, str) or not isinstance(json.loads(arguments), dict)
                or INKLING_PRIVATE_MARKER.search(arguments) or THINK_TAG.search(arguments)):
            raise ValueError("Unrecognized or malformed tool call")
        result.append({"id": identifier, "type": "function",
                       "function": {"name": name, "arguments": arguments}})
    return result


def request_text(messages, tools):
    if not isinstance(messages, list) or not messages:
        raise ValueError("Messages must be a nonempty array")
    texts = []
    for message in messages:
        role = message["role"]
        if role not in ALLOWED_ROLES:
            raise ValueError("Unknown message role")
        content = message.get("content")
        if isinstance(content, list):
            if any(part["type"] != "text" for part in content):
                raise ValueError("This safety gate only supports text messages")
            content = "\n".join(part["text"] for part in content)
        if not isinstance(content, str) and not (content is None and message.get("tool_calls")):
            raise ValueError("Expected text messages")
        texts.append((role, content or ""))
        metadata = {key: message[key] for key in ("tool_calls", "tool_call_id", "name") if key in message}
        if set(message) - {"role", "content", *metadata}:
            raise ValueError("Unsupported message fields")
        if metadata:
            texts.append(("tool metadata", as_json(metadata)))
    if not any(role == "user" and content.strip() for role, content in texts):
        raise ValueError("A user text message is required")
    if tools:
        texts.append(("tool definitions", as_json(tools)))
    if len(texts) == 1 and texts[0][0] == "user":
        return texts[0][1]
    return "\n\n".join(f"[{role}]\n{text}" for role, text in texts)


def gate_prompt(body, tools):
    prompt = request_text(body["messages"], tools)
    options = {key: body[key] for key in ("response_format", "chat_template_kwargs", "stop", "tool_choice")
               if key in body}
    if options:
        prompt += "\n\n[options]\n" + as_json(options)
    return prompt


def visible_content(content):
    if content is None:
        return None
    if not isinstance(content, str):
        raise ValueError("Expected text output")
    content = INKLING_THINK_BLOCK.sub("", THINK_BLOCK.sub("", content))
    if match := re.search(r"</think\s*>", content, flags=re.I):
        content = content[match.end():]
    if INKLING_PRIVATE_MARKER.search(content) or THINK_TAG.search(content):
        raise ValueError("Unterminated private output segment")
    content = INKLING_TOOL.sub(canonical_inline_call, content)
    if "<|content_invoke_tool_json|>" in content:
        raise ValueError("Unterminated tool output segment")
    return INKLING_TEXT_MARKERS.sub("", content).strip()


def canonical_inline_call(match):
    raw = json.loads(match[1])
    name, args = raw["name"], raw["args"]
    if not isinstance(name, str) or not re.fullmatch(r"[A-Za-z0-9_-]{1,64}", name) or not isinstance(args, dict):
        raise ValueError("Malformed inline tool call")
    return f"<tool_call>{as_json({'version': READONLY_TOOL_VERSION, 'name': name, 'arguments': args})}</tool_call>"


def visible_annotations(annotations):
    result = []
    for annotation in annotations:
        if annotation.get("type") != "url_citation":
            continue
        citation = annotation["url_citation"]
        parsed = urlsplit(citation["url"])
        if parsed.scheme not in {"http", "https"} or not parsed.hostname:
            raise ValueError("Malformed citation URL")
        fields = {"url": citation["url"]}
        if isinstance(citation.get("title"), str):
            fields["title"] = visible_content(citation["title"])
        for key in ("start_index", "end_index"):
            if type(citation.get(key)) is int:
                fields[key] = citation[key]
        result.append({"type": "url_citation", "url_citation": fields})
    return result


def completion(model, content, *, finish_reason="stop", tool_calls=None, annotations=None, source=None):
    message = {"role": "assistant", "content": content}
    if tool_calls:
        message["tool_calls"] = tool_calls
    if annotations:
        message["annotations"] = annotations
    source = source or {}
    result = {"id": source.get("id", "chatcmpl-local-gate"), "object": "chat.completion", "model": model,
              "choices": [{"index": 0, "message": message, "finish_reason": finish_reason}]}
    if type(source.get("created")) is int:
        result["created"] = source["created"]
    usage = source.get("usage")
    usage = usage if isinstance(usage, dict) else {}
    result["usage"] = {key: usage[key] if type(usage.get(key)) is int and usage[key] >= 0 else 0
                       for key in ("prompt_tokens", "completion_tokens", "total_tokens")}
    return result


class GuardedCompletion:
    def __init__(self, gate, upstream_url, model, upstream_token=None, *, system_prompt=None):
        self.gate = gate
        self.upstream_url = local_origin(upstream_url) + "/v1/chat/completions"
        self.model = model
        self.headers = {"Authorization": f"Bearer {upstream_token}"} if upstream_token else {}
        if system_prompt is not None and (
            not system_prompt.strip() or "\0" in system_prompt or len(system_prompt.encode("utf-8")) > 8192
        ):
            raise ValueError("Invalid server-owned system prompt")
        self.system_prompt = system_prompt

    def refuse(self, decision, *, source=None, deferred=False):
        status = 503 if decision.stopped_by == "judge-unavailable" else 200
        code = {"input-rule": "before-rule", "request-judge": "before-judge", "output-rule": "reply-rule",
                "judge": "reply-judge", "judge-unavailable": "withheld",
                "deferred-tool": "deferred-tool"}[decision.stopped_by]
        if deferred and code in {"reply-rule", "reply-judge"}:
            code += "-deferred"
        usage = {"usage": source.get("usage")} if source else None
        return status, completion(self.model, decision.text, source=usage), code

    def complete(self, body):
        if (not isinstance(body, dict) or set(body) - REQUEST_FIELDS or body.get("model") != self.model
                or body.get("stream", False) is not False or body.get("n", 1) != 1):
            raise ValueError("Send one non-streaming response request for the configured model")
        tools = body.get("tools", [])
        names = tool_names(tools)
        prompt = gate_prompt(body, tools)
        if self.system_prompt and any(message["role"] in {"system", "developer"} for message in body["messages"]):
            raise ValueError("The server supplies the system prompt for this endpoint")
        if len(prompt.encode("utf-8")) > MAX_TEXT_BYTES:
            raise ValueError("Text exceeds the safety gate input limit")
        before = self.gate.before(prompt)
        if not before.allowed:
            return self.refuse(before)
        try:
            upstream_body = body
            if self.system_prompt:
                upstream_body = {
                    **body,
                    "messages": [{"role": "system", "content": self.system_prompt}, *body["messages"]],
                }
            raw = post_json(self.upstream_url, upstream_body, timeout=180, headers=self.headers)
            choice, = raw["choices"]
            finish = choice["finish_reason"]
            if finish not in {"stop", "length", "tool_calls"}:
                raise ValueError("Unsupported response completion status")
            message = choice["message"]
            content = visible_content(message.get("content"))
            calls = safe_tool_calls(message.get("tool_calls", []), names)
            annotations = visible_annotations(message.get("annotations", []))
            if not content and not calls:
                raise ValueError("No checked final content")
            if finish == "tool_calls" and not calls:
                raise ValueError("Tool-call finish without a checked function call")
            if before.deferred and (calls or INLINE_TOOL_MARKER.search(content or "")):
                return self.refuse(Decision(False, "deferred-tool", "", "I can't help with that."), source=raw)
            visible = [content or ""]
            if calls:
                visible.append(as_json(calls))
            if annotations:
                visible.append(as_json(annotations))
            visible_reply = "\n".join(visible)
            after = (self.gate.after(prompt, visible_reply, deferred=True) if before.deferred
                     else self.gate.after(prompt, visible_reply))
            if not after.allowed:
                return self.refuse(after, source=raw, deferred=before.deferred)
            identifier = raw.get("id")
            if not isinstance(identifier, str) or not re.fullmatch(r"[A-Za-z0-9_-]{1,128}", identifier):
                raw.pop("id", None)
            return 200, completion(self.model, content, finish_reason=finish, tool_calls=calls,
                                   annotations=annotations, source=raw), "allow-deferred" if before.deferred else "allow"
        except (OSError, KeyError, IndexError, TypeError, ValueError):
            return 502, completion(self.model, UNAVAILABLE), "withheld"


class Handler(BaseHTTPRequestHandler):
    def log_message(self, format_string, *args):
        pass

    def send_json(self, status, result, decision=None):
        if not isinstance(result, dict):
            result = {"error": {"message": result}}
        wire = json.dumps(result, ensure_ascii=False).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Cache-Control", "no-store")
        self.send_header("Content-Length", str(len(wire)))
        if decision:
            self.send_header("X-Krummholz-Gate", decision)
        self.end_headers()
        self.wfile.write(wire)

    def do_POST(self):
        if self.path != "/v1/chat/completions":
            self.send_json(404, "Endpoint unavailable")
            return
        try:
            size = int(self.headers.get("Content-Length", "0"))
            if not 1 <= size <= 250_000:
                raise ValueError("Request size limit")
            document = json.loads(self.rfile.read(size))
            status, reply, decision = self.server.guarded.complete(document)
        except (KeyError, TypeError, ValueError, RecursionError):
            status, reply, decision = 400, "Invalid local non-streaming text request", "invalid-request"
        self.send_json(status, reply, decision)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--binary", required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--upstream-url", default="http://127.0.0.1:8000")
    parser.add_argument("--embedding-url", default="http://127.0.0.1:8089")
    parser.add_argument("--port", type=int, default=8090)
    parser.add_argument("--request-threshold", type=float)
    parser.add_argument("--deferred-reply-threshold", type=float)
    parser.add_argument("--system-prompt-file", type=Path, help="Server-owned instructions sent to the model after request screening.")
    args = parser.parse_args()
    try:
        gate = HardGate(binary=args.binary, embedding_url=args.embedding_url,
                        request_threshold=args.request_threshold,
                        deferred_reply_threshold=args.deferred_reply_threshold)
    except ValueError as error:
        parser.error(str(error))
    if not gate.before("What is the capital of Maine?").allowed:
        raise SystemExit("Released judge failed its startup check")
    try:
        system_prompt = (args.system_prompt_file.read_text(encoding="utf-8-sig").strip()
                         if args.system_prompt_file else None)
        guarded = GuardedCompletion(gate, args.upstream_url, args.model, os.getenv("INKLING_GATE_UPSTREAM_TOKEN"),
                                    system_prompt=system_prompt)
    except (OSError, UnicodeError, ValueError) as error:
        parser.error(str(error))
    with ThreadingHTTPServer(("127.0.0.1", args.port), Handler) as server:
        server.guarded = guarded
        server.serve_forever()


if __name__ == "__main__":
    main()
