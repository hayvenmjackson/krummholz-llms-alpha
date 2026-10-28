"""Provider-independent, read-only tool boundary; no live network adapter ships here.

Adapter code and caller policy are trusted host configuration. Model proposals,
search results and source text never grant capabilities. Evidence handles identify
request-local observations, not authenticated publishers or semantically true claims.
"""
from __future__ import annotations

import contextlib
from dataclasses import asdict, dataclass
import datetime as dt
import hashlib
import ipaddress
import json
import math
import multiprocessing
import os
import re
import time
from urllib.parse import parse_qsl, unquote, urljoin, urlsplit, urlunsplit
import uuid

VERSION = "krummholz-readonly-tools-v1"
VERSION_V2 = "krummholz-readonly-tools-v2"  # adds calculate (code, not the model) and the dated Krummholz Library
WEB_TOOLS = frozenset({"search_public_web", "fetch_public_page"})
V2_TOOLS = WEB_TOOLS | {"calculate", "search_library", "fetch_library"}
CONTROL = re.compile(r"[\x00-\x1f\x7f]")
CREDENTIAL_PARAMETER = re.compile(r"(?:token|apikey|authorization|password|secret|signature|credential)|^(?:key|auth|sig|session|sessionid|code)$", re.I)


class BrokerError(ValueError):
    pass


def _pairs(items):
    result = {}
    for key, value in items:
        if key in result:
            raise BrokerError("duplicate_json_key")
        result[key] = value
    return result


def strict_json(raw, maximum):
    if not isinstance(raw, str) or len(raw.encode("utf-8", errors="strict")) > maximum:
        raise BrokerError("json_size_limit")
    try:
        return json.loads(raw, object_pairs_hook=_pairs,
                          parse_constant=lambda _: (_ for _ in ()).throw(BrokerError("nonfinite_json")))
    except (ValueError, RecursionError) as error:
        raise BrokerError("invalid_json") from error


def keys(value, required):
    if type(value) is not dict or set(value) != set(required):
        raise BrokerError("unexpected_fields")


def text(value, maximum, *, empty=False, multiline=False):
    checked = value.replace("\n", "").replace("\r", "").replace("\t", "") if multiline and isinstance(value, str) else value
    if type(value) is not str or (not empty and not value.strip()) or len(value) > maximum or CONTROL.search(checked):
        raise BrokerError("invalid_text")
    try:
        value.encode("utf-8", errors="strict")
    except UnicodeError as error:
        raise BrokerError("invalid_unicode") from error
    return value


def public_ip(value):
    try:
        address = ipaddress.ip_address(value)
        if isinstance(address, ipaddress.IPv6Address) and address.ipv4_mapped is not None:
            address = address.ipv4_mapped
        return address.is_global and not (address.is_multicast or address.is_reserved or address.is_unspecified)
    except ValueError:
        return False


@dataclass(frozen=True)
class Limits:
    max_calls: int = 6
    total_seconds: float = 30
    call_seconds: float = 10
    max_results: int = 5
    max_query_chars: int = 512
    max_body_bytes: int = 65536
    total_body_bytes: int = 196608
    max_redirects: int = 3
    max_url_chars: int = 2048

    def validate(self):
        bounds = {"max_calls": (1, 16), "max_results": (1, 10), "max_query_chars": (1, 2048),
                  "max_body_bytes": (1, 262144), "total_body_bytes": (1, 1048576),
                  "max_redirects": (0, 3), "max_url_chars": (1, 4096)}
        for name, (minimum, maximum) in bounds.items():
            value = getattr(self, name)
            if type(value) is not int or not minimum <= value <= maximum:
                raise BrokerError("invalid_host_limits")
        for name, maximum in (("total_seconds", 120), ("call_seconds", 30)):
            value = getattr(self, name)
            if type(value) not in (int, float) or not math.isfinite(value) or not 0 < value <= maximum:
                raise BrokerError("invalid_host_limits")


@dataclass(frozen=True)
class PublicScope:
    """Exact public queries and hosts authorized by the host, never by a model."""
    profile: str
    allowed_queries: frozenset[str]
    allowed_hosts: frozenset[str]
    require_fresh_retrieval: bool = False
    max_retrieval_age_seconds: float = 300
    allowed_url_query_parameters: frozenset[str] = frozenset()


@dataclass(frozen=True)
class Evidence:
    source_id: str
    result_id: str
    url: str
    title: str
    text: str
    text_sha256: str
    fetched_at: str
    fetched_monotonic: float
    provider_mode: str
    media_type: str
    redirects: tuple[str, ...]
    untrusted_content: bool = True


def _adapter_process(pipe, adapter, operation, request, max_bytes):
    """Only JSON crosses back; no provider logs/exception details reach the model."""
    try:
        # Provider credentials, if later needed, belong to its trusted host adapter
        # configuration; do not inherit unrelated cloud/model/account credentials.
        keep = {"PATH", "SYSTEMROOT", "WINDIR", "SYSTEMDRIVE", "TEMP", "TMP", "LANG", "LC_ALL",
                "LD_LIBRARY_PATH", "DYLD_LIBRARY_PATH", "SSL_CERT_FILE", "SSL_CERT_DIR"}
        for name in list(os.environ):
            if name.upper() not in keep:
                del os.environ[name]
        with open(os.devnull, "w", encoding="utf-8") as sink, contextlib.redirect_stdout(sink), contextlib.redirect_stderr(sink):
            try:
                value = getattr(adapter, operation)(request)
                encoded = json.dumps({"ok": True, "value": value}, ensure_ascii=False, allow_nan=False).encode("utf-8")
                if len(encoded) > max_bytes:
                    encoded = b'{"ok":false,"error":"provider_response_too_large"}'
            except BaseException:
                encoded = b'{"ok":false,"error":"provider_error"}'
            pipe.send_bytes(encoded)
    finally:
        pipe.close()


class ProcessRunner:
    """One bounded child per adapter operation; no lingering timed-out threads."""
    def run(self, adapter, operation, request, timeout, maximum):
        context = multiprocessing.get_context("spawn")
        reader, writer = context.Pipe(duplex=False)
        process = context.Process(target=_adapter_process, args=(writer, adapter, operation, request, maximum), daemon=True)
        started = time.monotonic()
        try:
            process.start()
            writer.close()
            remaining = timeout - (time.monotonic() - started)
            if remaining <= 0 or not reader.poll(remaining):
                raise BrokerError("provider_timeout")
            payload = reader.recv_bytes(maxlength=maximum).decode("utf-8", errors="strict")
            envelope = strict_json(payload, maximum)
            if type(envelope) is not dict or envelope.get("ok") is not True:
                raise BrokerError(envelope.get("error", "provider_error") if type(envelope) is dict else "provider_error")
            keys(envelope, ("ok", "value"))
            return envelope["value"]
        except (EOFError, OSError, UnicodeError, RuntimeError, AttributeError, TypeError) as error:
            raise BrokerError("provider_transport_error") from error
        finally:
            reader.close()
            writer.close()
            if process.pid is not None:
                if process.is_alive():
                    process.terminate()
                process.join(timeout=0.25)
                if process.is_alive():
                    process.kill()
                    process.join(timeout=0.25)
                if process.is_alive():
                    raise BrokerError("provider_could_not_be_reaped")
                process.close()


class ToolBroker:
    def __init__(self, scope, *, search_adapter=None, fetch_adapter=None, provider_mode="fixture",
                 limits=Limits(), runner=None, monotonic=time.monotonic, utc_now=None, library=None):
        limits.validate()
        if not isinstance(scope, PublicScope) or scope.profile not in {"acadia", "katahdin", "atlantic"}:
            raise BrokerError("invalid_host_scope")
        if (type(scope.allowed_queries) is not frozenset or type(scope.allowed_hosts) is not frozenset
                or type(scope.allowed_url_query_parameters) is not frozenset):
            raise BrokerError("immutable_host_scope_required")
        if type(scope.require_fresh_retrieval) is not bool or type(scope.max_retrieval_age_seconds) not in (int, float):
            raise BrokerError("invalid_freshness_scope")
        if not math.isfinite(scope.max_retrieval_age_seconds) or not 0 < scope.max_retrieval_age_seconds <= 86400:
            raise BrokerError("invalid_freshness_scope")
        for query in scope.allowed_queries:
            text(query, limits.max_query_chars)
        if len(scope.allowed_queries) > 100 or len(scope.allowed_hosts) > 100:
            raise BrokerError("host_scope_too_large")
        for host in scope.allowed_hosts:
            if type(host) is not str or host != host.lower() or not re.fullmatch(r"[a-z0-9.-]+", host):
                raise BrokerError("invalid_host_allowlist")
        if len(scope.allowed_url_query_parameters) > 20:
            raise BrokerError("host_scope_too_large")
        for parameter in scope.allowed_url_query_parameters:
            if (type(parameter) is not str or not re.fullmatch(r"[a-z][a-z0-9_]{0,39}", parameter)
                    or CREDENTIAL_PARAMETER.search(parameter.replace("_", ""))):
                raise BrokerError("invalid_public_url_parameter_scope")
        if provider_mode not in {"fixture", "live"}:
            raise BrokerError("invalid_provider_mode")
        self.scope, self.limits, self.mode = scope, limits, provider_mode
        self.search_adapter, self.fetch_adapter = search_adapter, fetch_adapter
        self.runner = runner if runner is not None else ProcessRunner()
        self.clock = monotonic
        self.utc_now = utc_now if utc_now is not None else lambda: dt.datetime.now(dt.timezone.utc)
        self.started, self.calls, self.body_bytes = self.clock(), 0, 0
        self.run_id = uuid.uuid4().hex
        self._results, self._evidence, self._events = {}, {}, []
        self.library, self._library_results, self._library_evidence = library, {}, {}
        self._disabled = False

    def _url(self, value):
        value = text(value, self.limits.max_url_chars)
        if value != value.strip() or CONTROL.search(unquote(value)) or "\\" in value:
            raise BrokerError("invalid_public_url")
        try:
            parsed = urlsplit(value)
            host = parsed.hostname
            if (parsed.scheme != "https" or not host or parsed.username is not None or parsed.password is not None
                    or parsed.port not in (None, 443) or parsed.fragment or host not in self.scope.allowed_hosts):
                raise BrokerError("url_outside_public_scope")
            if host == "localhost" or host.endswith((".localhost", ".local", ".internal", ".lan", ".home")):
                raise BrokerError("private_network_target")
            try:
                address = ipaddress.ip_address(host)
            except ValueError:
                address = None
            if address is not None and not public_ip(str(address)):
                raise BrokerError("private_network_target")
            parameters = parse_qsl(parsed.query, keep_blank_values=True, max_num_fields=20)
            if ";" in unquote(parsed.query):
                raise BrokerError("ambiguous_url_query")
            if parsed.query and (not parameters or any(name not in self.scope.allowed_url_query_parameters for name, _ in parameters)):
                raise BrokerError("url_query_parameters_not_authorized")
            if any(CREDENTIAL_PARAMETER.search(re.sub(r"[^a-z0-9]", "", name.casefold())) for name, _ in parameters):
                raise BrokerError("credential_bearing_url")
            return urlunsplit(("https", parsed.netloc, parsed.path or "/", parsed.query, ""))
        except ValueError as error:
            if isinstance(error, BrokerError):
                raise
            raise BrokerError("invalid_public_url") from error

    def _remaining(self, call_started):
        return min(self.limits.call_seconds - (self.clock() - call_started),
                   self.limits.total_seconds - (self.clock() - self.started))

    def _operation(self, adapter, operation, request, call_started):
        if adapter is None:
            raise BrokerError("provider_unavailable")
        remaining = self._remaining(call_started)
        if remaining <= 0:
            raise BrokerError("time_budget_exhausted")
        request = {**request, "timeout_seconds": remaining}
        maximum = self.limits.max_body_bytes + 10 * (self.limits.max_url_chars + 2400) + 4096
        value = self.runner.run(adapter, operation, request, remaining, maximum)
        if self._remaining(call_started) <= 0:
            raise BrokerError("provider_timeout")
        return value

    def invoke(self, raw_proposal):
        """One JSON proposal in, one bounded data response out; never executes model text."""
        if self._disabled:
            return {"status": "denied", "reason": "broker_disabled", "side_effecting_actions": []}
        if self.calls >= self.limits.max_calls or self.clock() - self.started >= self.limits.total_seconds:
            return {"status": "denied", "reason": "budget_exhausted", "side_effecting_actions": []}
        self.calls += 1
        started, name = self.clock(), "unparsed"
        try:
            proposal = strict_json(raw_proposal, 8192)
            keys(proposal, ("version", "name", "arguments"))
            if proposal["version"] not in (VERSION, VERSION_V2):
                raise BrokerError("unknown_tool_protocol")
            name = text(proposal["name"], 64)
            if name not in (WEB_TOOLS if proposal["version"] == VERSION else V2_TOOLS):
                raise BrokerError("tool_unavailable")
            arguments = proposal["arguments"]
            if name == "calculate":
                from krummholz_llm import calculate
                keys(arguments, ("expression",))
                response = calculate.run({"expression": text(arguments["expression"], calculate.MAX_CHARS)})
            elif name == "search_library":
                keys(arguments, ("query",))
                response = self._search_library(text(arguments["query"], self.limits.max_query_chars))
            elif name == "fetch_library":
                keys(arguments, ("result_id",))
                response = self._fetch_library(text(arguments["result_id"], 100))
            elif name == "search_public_web":
                keys(arguments, ("query",))
                query = text(arguments["query"], self.limits.max_query_chars)
                if query not in self.scope.allowed_queries:
                    raise BrokerError("public_query_not_authorized")
                response = self._search(query, started)
            else:
                keys(arguments, ("result_id",))
                result_id = text(arguments["result_id"], 100)
                if result_id not in self._results:
                    raise BrokerError("unknown_request_local_result")
                response = self._fetch(result_id, started)
            response.update(provider_mode=self.mode, side_effecting_actions=[])
        except (BrokerError, UnicodeError, TypeError) as error:
            reason = str(error) if isinstance(error, BrokerError) else "invalid_proposal"
            if reason == "provider_could_not_be_reaped":
                self._disabled = True
            response = {"status": "denied" if reason in {"tool_unavailable", "public_query_not_authorized"} else "unresolved",
                        "reason": reason, "provider_mode": self.mode, "side_effecting_actions": []}
        self._events.append({"call": self.calls, "tool": name if name in V2_TOOLS else "unavailable",
                             "status": response["status"], "reason": response.get("reason"),
                             "elapsed_seconds": max(0, self.clock() - started)})
        return response

    def _search_library(self, query):
        if self.library is None:
            raise BrokerError("library_unavailable")
        results = []
        for doc_id, _ in self.library.search(query, k=self.limits.max_results):
            result_id = f"library-{self.run_id}-{len(self._library_results) + 1}"
            self._library_results[result_id] = doc_id
            results.append(self.library.hit(doc_id, result_id))
        return {"status": "search_results" if results else "no_results", "results": results, "library_as_of": self.library.as_of}

    def _fetch_library(self, result_id):
        if self.library is None:
            raise BrokerError("library_unavailable")
        if result_id not in self._library_results:
            raise BrokerError("unknown_request_local_result")
        source_id = f"library/{self.run_id}-{len(self._library_evidence) + 1}"
        evidence = self.library.evidence(self._library_results[result_id], source_id)
        self._library_evidence[source_id] = evidence
        return {"status": "evidence_observed", "evidence": {**evidence, "current_facts": "as_of_library_date"}}

    def _search(self, query, started):
        payload = self._operation(self.search_adapter, "search", {"query": query, "max_results": self.limits.max_results}, started)
        keys(payload, ("results",))
        rows = payload["results"]
        if type(rows) is not list or len(rows) > self.limits.max_results:
            raise BrokerError("invalid_search_results")
        staged = []
        for row in rows:
            keys(row, ("url", "title", "snippet"))
            staged.append({"url": self._url(row["url"]), "title": text(row["title"], 300),
                           "snippet": text(row["snippet"], 2048, empty=True)})
        results = []
        for row in staged:
            result_id = f"result-{self.run_id}-{len(self._results) + 1}"
            self._results[result_id] = row
            results.append({"result_id": result_id, **row, "untrusted_content": True, "citable": False})
        return {"status": "search_results" if results else "no_results", "results": results}

    def _fetch(self, result_id, started):
        discovery = self._results[result_id]
        url, redirects = discovery["url"], []
        for hop in range(self.limits.max_redirects + 1):
            url = self._url(url)
            hostname = urlsplit(url).hostname
            resolved = self._operation(self.fetch_adapter, "resolve", {"hostname": hostname}, started)
            keys(resolved, ("addresses",))
            addresses = resolved["addresses"]
            if (type(addresses) is not list or not 1 <= len(addresses) <= 8
                    or any(type(ip) is not str or not public_ip(ip) for ip in addresses)):
                raise BrokerError("private_or_invalid_dns_answer")
            remaining_bytes = min(self.limits.max_body_bytes, self.limits.total_body_bytes - self.body_bytes)
            if remaining_bytes <= 0:
                raise BrokerError("body_budget_exhausted")
            fetched = self._operation(self.fetch_adapter, "fetch", {
                "url": url, "addresses": sorted(set(addresses)), "tls_server_name": hostname,
                "max_body_bytes": remaining_bytes, "follow_redirects": False,
                "send_credentials": False, "send_cookies": False}, started)
            keys(fetched, ("status", "location", "text", "media_type"))
            status = fetched["status"]
            if type(status) is not int:
                raise BrokerError("invalid_fetch_status")
            body = fetched["text"]
            if type(body) is not str:
                raise BrokerError("invalid_fetch_body")
            size = len(body.encode("utf-8", errors="strict"))
            if size > remaining_bytes:
                raise BrokerError("body_size_limit")
            self.body_bytes += size
            if status in {301, 302, 303, 307, 308}:
                if hop == self.limits.max_redirects:
                    raise BrokerError("redirect_budget_exhausted")
                location = text(fetched["location"], self.limits.max_url_chars)
                next_url = self._url(urljoin(url, location))
                if next_url == url or next_url in redirects:
                    raise BrokerError("redirect_loop")
                redirects.append(url)
                url = next_url
                continue
            if status != 200:
                raise BrokerError("upstream_http_error")
            if fetched["location"] is not None or fetched["media_type"] not in {"text/plain", "text/html", "application/json"}:
                raise BrokerError("unsupported_fetch_content")
            if not body.strip() or "\x00" in body:
                raise BrokerError("empty_or_binary_fetch_body")
            now = self.utc_now()
            if not isinstance(now, dt.datetime) or now.tzinfo is None:
                raise BrokerError("invalid_host_clock")
            source_id = f"{self.scope.profile}/web-{self.run_id}-{len(self._evidence) + 1}"
            evidence = Evidence(source_id, result_id, url, discovery["title"], body, hashlib.sha256(body.encode()).hexdigest(),
                                now.astimezone(dt.timezone.utc).isoformat(), self.clock(), self.mode, fetched["media_type"], tuple(redirects))
            self._evidence[source_id] = evidence
            return {"status": "evidence_observed", "evidence": self._view(evidence)}
        raise BrokerError("redirect_budget_exhausted")

    def _view(self, evidence):
        value = asdict(evidence)
        age = max(0, self.clock() - evidence.fetched_monotonic)
        value.pop("fetched_monotonic")
        value.update(retrieval_age_seconds=age, retrieval_fresh=age <= self.scope.max_retrieval_age_seconds,
                     publisher_identity="not_authenticated_by_source_id", semantic_support="not_determined",
                     current_facts="not_established_by_retrieval_time")
        return value

    def ledger(self):
        return {"version": VERSION, "provider_mode": self.mode, "calls_used": self.calls,
                "body_bytes_used": self.body_bytes, "sources": [self._view(row) for row in self._evidence.values()],
                "events": [dict(row) for row in self._events], "semantic_review": "not_performed",
                "training_performed": False}

    def assess_citations(self, citations):
        """Checks evidence availability, literal quotes and retrieval age, never truth."""
        reasons = []
        if type(citations) is not list or not citations or len(citations) > 32:
            return {"status": "unresolved", "reasons": ["no_bounded_citations"], "semantic_support": "not_determined"}
        for citation in citations:
            try:
                keys(citation, ("source_id", "quote"))
                source_id = text(citation["source_id"], 100)
                quote = text(citation["quote"], 4096, multiline=True)
                library_text = self._library_evidence.get(source_id, {}).get("text")
                if library_text is not None:  # dated library release: literal quote check, no live-freshness rule
                    if quote not in library_text:
                        raise BrokerError("quote_not_in_observed_source")
                    continue
                evidence = self._evidence.get(source_id)
                if evidence is None:
                    raise BrokerError("unknown_request_local_source")
                if quote not in evidence.text:
                    raise BrokerError("quote_not_in_observed_source")
                if self.scope.require_fresh_retrieval:
                    if evidence.provider_mode != "live":
                        raise BrokerError("fixture_is_not_live_retrieval")
                    if self.clock() - evidence.fetched_monotonic > self.scope.max_retrieval_age_seconds:
                        raise BrokerError("stale_retrieval")
            except (BrokerError, UnicodeError, TypeError) as error:
                reasons.append(str(error) if isinstance(error, BrokerError) else "invalid_citation")
        return {"status": "unresolved" if reasons else "evidence_ready_for_semantic_review", "reasons": reasons,
                "semantic_support": "not_determined", "current_facts": "not_established_by_retrieval_time"}
