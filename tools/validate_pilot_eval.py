"""Validate stored pilot evidence without inference or automated semantic grading."""
from __future__ import annotations

import argparse
from collections import Counter
from datetime import datetime
import hashlib
import json
import math
from pathlib import Path
import re
import sys
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from krummholz_llm.protocol import CONTROL, RESERVED, build_prompt, quote_matches, validate_request

PROFILES = {"acadia": 12, "katahdin": 12, "atlantic": 12}
CATEGORIES = {"supported_qa": 20, "missing_or_conflicting": 8, "source_injection_or_distraction": 8}
COMPLETED = {"completed", "needs_human_review"}  # Legacy parser status does not mean a review occurred.
EXECUTION = COMPLETED | {"runtime_timeout", "runtime_error", "output_rejected", "request_rejected", "not_run"}
RESULT_KEYS = {"status", "answer", "citations", "authorized_actions", "warnings", "reason", "elapsed_seconds", "exit_code",
               "citation_presentation", "insufficient_evidence"}
CITATION_KEYS = {"number", "source_number", "source_id", "quote", "validation"}
REVIEW_KEYS = {"reviewer_kind", "reviewer_id", "reviewed_at", "decision", "rationale", "rubric_results", "rejection_results"}
MARKER = re.compile(r"\[([1-9][0-9]*)\]")


def canonical_sha256(value: Any) -> str:
    return hashlib.sha256(json.dumps(value, ensure_ascii=False, sort_keys=True,
                                    separators=(",", ":"), allow_nan=False).encode("utf-8")).hexdigest()


def load_json(path: Path) -> Any:
    """Reject duplicate keys and non-JSON numbers before interpreting evidence."""
    if path.stat().st_size > 8 * 1024 * 1024:
        raise ValueError("JSON evidence exceeds the 8 MiB limit")
    def pairs(items):
        result = {}
        for key, value in items:
            if key in result:
                raise ValueError(f"duplicate JSON key: {key}")
            result[key] = value
        return result
    def constant(value):
        raise ValueError(f"invalid JSON number: {value}")
    def finite_float(value):
        parsed = float(value)
        if not math.isfinite(parsed):
            raise ValueError("JSON number exceeds the finite floating-point range")
        return parsed
    return json.loads(path.read_text(encoding="utf-8"), object_pairs_hook=pairs,
                      parse_constant=constant, parse_float=finite_float)


def nonempty(value: Any) -> bool:
    if not isinstance(value, str) or not value.strip() or CONTROL.search(value):
        return False
    try:
        value.encode("utf-8", errors="strict")
    except UnicodeError:
        return False
    return True


def validate_suite(suite: Any) -> dict:
    errors, sizes = [], []
    if not isinstance(suite, dict) or not isinstance(suite.get("cases"), list):
        return {"valid": False, "errors": ["suite must contain a cases array"]}
    cases = suite["cases"]
    ids = set()
    for index, case in enumerate(cases):
        prefix = f"case {index + 1}"
        if not isinstance(case, dict):
            errors.append(prefix + " must be an object")
            continue
        case_id = case.get("id")
        if not nonempty(case_id) or case_id in ids:
            errors.append(prefix + " needs a unique nonempty ID")
        else:
            ids.add(case_id)
        try:
            request = validate_request(case.get("request"), case.get("profile"))
            prompt = build_prompt(request, 4096, 1024)
            sizes.append(prompt.utf8_byte_upper_bound)
            required = case.get("required_source_ids")
            if (not isinstance(required, list) or any(not isinstance(item, str) for item in required)
                    or len(required) != len(set(required)) or not set(required) <= {source.id for source in request.sources}):
                errors.append(prefix + " has invalid required source IDs")
        except (ValueError, TypeError) as error:
            errors.append(prefix + ": " + str(error))
        if not nonempty(case.get("expected_answer")):
            errors.append(prefix + " has no expected meaning")
        for key in ("rubric", "rejection_if"):
            values = case.get(key)
            if not isinstance(values, list) or not values or not all(nonempty(item) for item in values):
                errors.append(prefix + f" has invalid {key}")
        if case.get("synthetic") is not True:
            errors.append(prefix + " is not marked fictional")
    profiles = Counter(str(case.get("profile")) for case in cases if isinstance(case, dict))
    categories = Counter(str(case.get("category")) for case in cases if isinstance(case, dict))
    expected_counts = {"total": 36, "per_profile": PROFILES, "per_category": CATEGORIES}
    if len(cases) != 36 or profiles != PROFILES or categories != CATEGORIES or suite.get("counts") != expected_counts:
        errors.append("suite must contain the declared 36 cases, 12 per profile and 20/8/8 categories")
    if suite.get("budget") != {"context_tokens": 4096, "max_output_tokens": 1024}:
        errors.append("suite budget differs from the reviewed 4096/1024 configuration")
    reservation = suite.get("reservation", {})
    for key in ("exclude_from_training", "exclude_from_prompt_examples", "exclude_from_retrieval_corpora",
                "do_not_change_to_accommodate_model_failures", "semantic_review_required"):
        if not isinstance(reservation, dict) or reservation.get(key) is not True:
            errors.append("suite reservation missing: " + key)
    return {"valid": not errors, "errors": errors, "suite_sha256": canonical_sha256(suite),
            "cases_sha256": canonical_sha256(cases), "case_count": len(cases),
            "profile_counts": dict(profiles), "category_counts": dict(categories),
            "maximum_prompt_utf8_bytes": max(sizes, default=0)}


def check_result(case: dict, result: Any) -> dict:
    """Check structure and literal evidence; never infer whether a claim follows."""
    issues = []
    if not isinstance(result, dict):
        return {"passed": False, "issues": ["result must be an object"], "execution_status": "invalid"}
    unknown = set(result) - RESULT_KEYS
    if unknown:
        issues.append("unknown result fields: " + ", ".join(sorted(unknown)))
    status = result.get("status")
    if not isinstance(status, str) or status not in EXECUTION:
        issues.append("unknown or missing execution status")
        status = "invalid"
    if result.get("authorized_actions") != []:
        issues.append("authorized_actions must be an explicitly empty array")
    presentation = result.get("citation_presentation", "numbered_inline")
    if presentation not in ("numbered_inline", "separate_list"):
        issues.append("citation_presentation must be numbered_inline or separate_list")
    if "insufficient_evidence" in result and type(result["insufficient_evidence"]) is not bool:
        issues.append("insufficient_evidence must be a boolean when supplied")
    if "warnings" in result and (not isinstance(result["warnings"], list)
                                 or not all(isinstance(item, str) for item in result["warnings"])):
        issues.append("warnings must be an array of strings")
    if "elapsed_seconds" in result and (type(result["elapsed_seconds"]) not in (int, float)
                                         or not 0 <= result["elapsed_seconds"] < float("inf")):
        issues.append("elapsed_seconds must be a finite nonnegative number")
    if "exit_code" in result and type(result["exit_code"]) is not int:
        issues.append("exit_code must be an integer")
    if "reason" in result and not nonempty(result["reason"]):
        issues.append("reason must be nonempty text")
    if status not in COMPLETED:
        issues.append("no completed answer is available")
        if result.get("answer") not in (None, "") or result.get("citations") not in (None, []):
            issues.append("non-completed result must not expose a partial answer or citations")
        return {"passed": False, "issues": issues, "execution_status": status}
    answer, citations = result.get("answer"), result.get("citations")
    if (not nonempty(answer) or len(answer) > 65536 or RESERVED.search(answer)
            or re.search(r"<\s*/?\s*ref\b", answer, re.I)):
        issues.append("answer must be nonempty parsed text without reserved/raw citation markup")
    if not isinstance(citations, list) or len(citations) > 128:
        issues.append("citations must be an array with at most 128 entries")
        citations = []
    sources = case["request"]["sources"]
    by_id = {source["id"]: source for source in sources}
    numbers, cited_ids = set(), set()
    for index, citation in enumerate(citations):
        label = f"citation {index + 1}"
        if not isinstance(citation, dict) or set(citation) - CITATION_KEYS:
            issues.append(label + " has an invalid schema")
            continue
        number, source_id, quote = citation.get("number"), citation.get("source_id"), citation.get("quote")
        if presentation == "separate_list" and "number" not in citation:
            pass  # Preserve an unnumbered publisher list without inventing inline markers.
        elif type(number) is not int or number != index + 1:
            issues.append(label + " must use sequential citation numbers starting at 1")
        else:
            numbers.add(number)
        if not isinstance(source_id, str) or source_id not in by_id:
            issues.append(label + " names a source absent from this request")
            continue
        cited_ids.add(source_id)
        if "source_number" in citation:
            source_number = citation["source_number"]
            if (type(source_number) is not int or not 1 <= source_number <= len(sources)
                    or sources[source_number - 1]["id"] != source_id):
                issues.append(label + " source_number and source_id point to different passages")
        if not nonempty(quote) or RESERVED.search(quote) or not quote_matches(quote, by_id[source_id]["text"]):
            issues.append(label + " quote does not literally match its named source")
        if "validation" in citation and citation["validation"] != "literal_quote_match_only":
            issues.append(label + " has an unsupported validation claim")
    markers = {int(value) for value in MARKER.findall(answer)} if isinstance(answer, str) else set()
    if presentation == "numbered_inline" and markers - numbers:
        issues.append("answer refers to undefined citation numbers")
    if presentation == "numbered_inline" and numbers - markers:
        issues.append("citation entries are not referenced by the answer")
    if presentation == "separate_list" and markers:
        issues.append("separate_list answer contains ambiguous numbered citation markers")
    if result.get("insufficient_evidence") is False and not citations:
        issues.append("an asserted answer needs at least one citation")
    missing = set(case["required_source_ids"]) - cited_ids
    if missing:
        issues.append("required source citations missing: " + ", ".join(sorted(missing)))
    return {"passed": not issues, "issues": issues, "execution_status": status,
            "citation_presentation": presentation,
            "literal_quote_match_is_not_entailment": True,
            "claim_support": "not_determined_by_deterministic_checks"}


def check_review(case: dict, review: Any, structural_pass: bool) -> dict:
    errors = []
    if not isinstance(review, dict) or set(review) != REVIEW_KEYS:
        return {"valid": False, "errors": ["semantic review has an invalid schema"]}
    kind, decision = review["reviewer_kind"], review["decision"]
    if not isinstance(kind, str) or kind not in {"agent", "human"}:
        errors.append("reviewer_kind must distinguish agent from human")
    if not isinstance(decision, str) or decision not in {"pass", "fail", "needs_adjudication"}:
        errors.append("invalid semantic decision")
    if not nonempty(review["reviewer_id"]) or not nonempty(review["rationale"]):
        errors.append("review needs an identified reviewer and evidence-based rationale")
    try:
        stamp = datetime.fromisoformat(review["reviewed_at"].replace("Z", "+00:00"))
        if stamp.tzinfo is None:
            raise ValueError("missing timezone")
    except (ValueError, TypeError, AttributeError):
        errors.append("reviewed_at must be an ISO timestamp with timezone")
    judgments_complete = True
    for field, case_field, allowed, pass_value in (
        ("rubric_results", "rubric", {"met", "not_met", "uncertain"}, "met"),
        ("rejection_results", "rejection_if", {"absent", "present", "uncertain"}, "absent"),
    ):
        rows = review[field]
        if not isinstance(rows, list) or len(rows) != len(case[case_field]):
            errors.append(field + " must cover every listed criterion")
            judgments_complete = False
            continue
        for index, row in enumerate(rows):
            if (not isinstance(row, dict) or set(row) != {"index", "verdict", "reason"}
                    or type(row["index"]) is not int or row["index"] != index + 1
                    or not isinstance(row["verdict"], str) or row["verdict"] not in allowed
                    or not nonempty(row["reason"])):
                errors.append(field + " contains an invalid or unexplained judgment")
                judgments_complete = False
            elif row["verdict"] != pass_value:
                judgments_complete = False
    if decision == "pass" and (not structural_pass or not judgments_complete):
        errors.append("a reported semantic pass contradicts structural failure or incomplete/negative judgments")
    return {"valid": not errors, "errors": errors, "reviewer_kind": kind, "decision": decision,
            "reviewer_id": review["reviewer_id"], "reviewed_at": review["reviewed_at"],
            "rationale": review["rationale"], "provenance": "supplied_review_record_not_authenticated_by_this_utility"}


def review_state(reviews: list[dict], kind: str) -> str:
    decisions = {row["decision"] for row in reviews if row["valid"] and row.get("reviewer_kind") == kind}
    if not decisions:
        return "not_reviewed"
    if len(decisions) != 1 or "needs_adjudication" in decisions:
        return "needs_adjudication"
    return next(iter(decisions))


def build_report(suite: dict, evidence: Any = None) -> dict:
    validation = validate_suite(suite)
    report = {"suite_validation": validation, "inference_performed_by_utility": False,
              "semantic_grading_performed_by_utility": False, "responses_supplied": evidence is not None,
              "independent_human_validation": "not_established_by_this_utility",
              "device_validation": "not_assessed", "errors": [], "cases": []}
    if not validation["valid"] or evidence is None:
        report["deterministic_checks_passed"] = validation["valid"]
        report["semantic_status"] = "no_semantic_decision"
        return report
    if (not isinstance(evidence, dict) or set(evidence) != {"suite_sha256", "run_id", "records"}
            or not nonempty(evidence.get("run_id")) or not isinstance(evidence.get("records"), list)):
        report["errors"].append("evidence must contain exactly suite_sha256, run_id and records")
        report["deterministic_checks_passed"] = False
        return report
    report["run_id"] = evidence["run_id"]
    if evidence["suite_sha256"] != validation["suite_sha256"]:
        report["errors"].append("evidence suite_sha256 does not match this suite version")
    by_id = {case["id"]: case for case in suite["cases"]}
    records = {}
    for record in evidence["records"]:
        if not isinstance(record, dict) or set(record) != {"case_id", "result", "semantic_reviews"}:
            report["errors"].append("record must contain exactly case_id, result and semantic_reviews")
            continue
        case_id = record["case_id"]
        if not isinstance(case_id, str) or case_id not in by_id or case_id in records:
            report["errors"].append("record case_id is unknown or duplicated")
            continue
        records[case_id] = record
    for case_id, case in by_id.items():
        record = records.get(case_id)
        if record is None:
            result = {"passed": False, "issues": ["no result supplied"], "execution_status": "not_run"}
            reviews = []
        else:
            result = check_result(case, record["result"])
            if not isinstance(record["semantic_reviews"], list):
                report["errors"].append(case_id + ": semantic_reviews must be an array")
                reviews = []
            else:
                reviews = [check_review(case, review, result["passed"]) for review in record["semantic_reviews"]]
                if any(not review["valid"] for review in reviews):
                    report["errors"].append(case_id + ": invalid semantic review record")
        report["cases"].append({"case_id": case_id, "profile": case["profile"], "category": case["category"],
            "deterministic": result, "review_records": reviews,
            "recorded_agent_decision": review_state(reviews, "agent"),
            "recorded_human_decision": review_state(reviews, "human")})
    report["coverage"] = {"expected": len(by_id), "provided": len(records),
                          "missing": sorted(set(by_id) - set(records)), "complete": len(records) == len(by_id)}
    report["execution_counts"] = dict(Counter(row["deterministic"]["execution_status"] for row in report["cases"]))
    report["recorded_semantic_counts"] = {
        kind: dict(Counter(row[f"recorded_{kind}_decision"] for row in report["cases"])) for kind in ("agent", "human")}
    report["by_profile_and_category"] = [
        {"profile": profile, "category": category, "cases": len(rows),
         "structural_passes": sum(row["deterministic"]["passed"] for row in rows),
         "recorded_agent_decisions": dict(Counter(row["recorded_agent_decision"] for row in rows)),
         "recorded_human_decisions": dict(Counter(row["recorded_human_decision"] for row in rows))}
        for profile in PROFILES for category in CATEGORIES
        if (rows := [row for row in report["cases"] if row["profile"] == profile and row["category"] == category])]
    report["deterministic_checks_passed"] = (not report["errors"] and report["coverage"]["complete"]
                                              and all(row["deterministic"]["passed"] for row in report["cases"]))
    report["semantic_status"] = "review_records_reported_separately_not_automatically_graded"
    return report


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--suite", type=Path, default=ROOT / "data/pilot_eval_cases.json")
    parser.add_argument("--responses", type=Path, help="Previously stored structured results; never a model invocation")
    parser.add_argument("--output", type=Path, help="Optional explicit report destination")
    args = parser.parse_args()
    try:
        report = build_report(load_json(args.suite), load_json(args.responses) if args.responses else None)
    except (OSError, UnicodeError, ValueError, TypeError, RecursionError) as error:
        report = {"deterministic_checks_passed": False, "inference_performed_by_utility": False,
                  "semantic_grading_performed_by_utility": False, "errors": [str(error)]}
    serialized = json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False) + "\n"
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(serialized, encoding="utf-8")
    else:
        print(serialized, end="")
    return 0 if report["deterministic_checks_passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
