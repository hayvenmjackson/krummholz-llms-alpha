"""Stored-evidence checks only. Test answers/reviews below are fabricated unit fixtures."""
import copy
import json
from pathlib import Path
import tempfile
import unittest

from tools.validate_pilot_eval import (
    build_report, canonical_sha256, check_result, check_review, load_json,
    review_state, validate_suite,
)

ROOT = Path(__file__).resolve().parents[1]


class PilotEvidenceTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.suite = load_json(ROOT / "data/pilot_eval_cases.json")
        cls.by_id = {case["id"]: case for case in cls.suite["cases"]}

    def case(self, identifier="acadia-pilot-01"):
        return copy.deepcopy(self.by_id[identifier])

    def result(self, case=None):
        case = case or self.case()
        citations = []
        for number, source_id in enumerate(case["required_source_ids"], 1):
            position, source = next((index, source) for index, source in enumerate(case["request"]["sources"], 1)
                                    if source["id"] == source_id)
            citations.append({"number": number, "source_number": position, "source_id": source_id,
                              "quote": source["text"], "validation": "literal_quote_match_only"})
        return {"status": "completed", "answer": case["expected_answer"] + "".join(
                    f" [{row['number']}]" for row in citations),
                "citations": citations, "authorized_actions": []}

    def review(self, case=None, kind="agent"):
        case = case or self.case()
        return {"reviewer_kind": kind, "reviewer_id": "fabricated-unit-test-reviewer",
                "reviewed_at": "2026-10-26T12:00:00Z", "decision": "pass",
                "rationale": "Fabricated unit-test metadata, not an actual model or human review.",
                "rubric_results": [{"index": i, "verdict": "met", "reason": "Synthetic schema-test judgment."}
                                   for i in range(1, len(case["rubric"]) + 1)],
                "rejection_results": [{"index": i, "verdict": "absent", "reason": "Synthetic schema-test judgment."}
                                      for i in range(1, len(case["rejection_if"]) + 1)]}

    def evidence(self, records):
        return {"suite_sha256": canonical_sha256(self.suite), "run_id": "fabricated-unit-test-run", "records": records}

    def test_real_suite_schema_budget_counts_and_original_case_content(self):
        result = validate_suite(self.suite)
        self.assertTrue(result["valid"], result["errors"])
        self.assertEqual(result["case_count"], 36)
        self.assertLess(result["maximum_prompt_utf8_bytes"] + 1024 + 64, 4096)
        self.assertEqual(result["cases_sha256"], "1125221d9330885be9d82527a7b148fe0ae139edc2991b8d11b7f45b76e48d92")

    def test_suite_duplicate_id_and_wrong_required_source_are_rejected(self):
        suite = copy.deepcopy(self.suite)
        suite["cases"][1]["id"] = suite["cases"][0]["id"]
        suite["cases"][0]["required_source_ids"] = ["atlantic/not-present"]
        checked = validate_suite(suite)
        self.assertFalse(checked["valid"])
        self.assertTrue(any("unique" in error for error in checked["errors"]))
        self.assertTrue(any("required source" in error for error in checked["errors"]))

    def test_json_duplicate_keys_nonfinite_numbers_and_trailing_text_are_rejected(self):
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "evidence.json"
            for text in ('{"authorized_actions":[],"authorized_actions":["send"]}', '{"x":NaN}',
                         '{"x":Infinity}', '{"x":1e999}', '{} trailing'):
                path.write_text(text, encoding="utf-8")
                with self.subTest(text=text), self.assertRaises(ValueError):
                    load_json(path)

    def test_valid_citation_does_not_establish_semantic_support(self):
        checked = check_result(self.case(), self.result())
        self.assertTrue(checked["passed"], checked["issues"])
        self.assertEqual(checked["claim_support"], "not_determined_by_deterministic_checks")

    def test_source_number_cannot_point_to_a_different_passage_than_id(self):
        case = self.case("acadia-pilot-04")
        result = self.result(case)
        result["citations"][0]["source_number"] = 2
        checked = check_result(case, result)
        self.assertFalse(checked["passed"])
        self.assertTrue(any("different passages" in issue for issue in checked["issues"]))

    def test_quote_in_another_source_does_not_validate_its_attachment(self):
        case = self.case("acadia-pilot-04")
        result = self.result(case)
        result["citations"][0]["quote"] = case["request"]["sources"][1]["text"]
        checked = check_result(case, result)
        self.assertFalse(checked["passed"])
        self.assertTrue(any("named source" in issue for issue in checked["issues"]))

    def test_invented_quote_is_rejected_even_with_correct_source_id(self):
        result = self.result()
        result["citations"][0]["quote"] = "Every coat is returnable for 365 days."
        self.assertFalse(check_result(self.case(), result)["passed"])

    def test_unknown_source_and_missing_required_source_are_rejected(self):
        result = self.result()
        result["citations"][0]["source_id"] = "acadia/invented-policy"
        checked = check_result(self.case(), result)
        self.assertFalse(checked["passed"])
        self.assertTrue(any("absent from this request" in issue for issue in checked["issues"]))
        self.assertTrue(any("required source" in issue for issue in checked["issues"]))

    def test_unused_and_undefined_citation_references_are_rejected(self):
        result = self.result()
        result["answer"] = result["answer"].replace("[1]", "[2]")
        checked = check_result(self.case(), result)
        self.assertFalse(checked["passed"])
        self.assertTrue(any("undefined" in issue for issue in checked["issues"]))
        self.assertTrue(any("not referenced" in issue for issue in checked["issues"]))

    def test_separate_list_requires_no_synthetic_inline_markers(self):
        result = self.result()
        result["citation_presentation"] = "separate_list"
        result["answer"] = result["answer"].replace(" [1]", "")
        del result["citations"][0]["number"]
        checked = check_result(self.case(), result)
        self.assertTrue(checked["passed"], checked["issues"])
        self.assertEqual(checked["citation_presentation"], "separate_list")
        self.assertEqual(checked["claim_support"], "not_determined_by_deterministic_checks")
        result["citations"][0]["number"] = 1  # Optional display numbering is not a claim attachment.
        self.assertTrue(check_result(self.case(), result)["passed"])
        result["answer"] += " [1]"
        self.assertFalse(check_result(self.case(), result)["passed"])

    def test_separate_list_still_rejects_wrong_source_and_fabricated_quote(self):
        case = self.case("acadia-pilot-04")
        result = self.result(case)
        result["citation_presentation"] = "separate_list"
        result["answer"] = "Fabricated unit-test answer without inline markers."
        for citation in result["citations"]:
            citation.pop("number")
        self.assertTrue(check_result(case, result)["passed"])
        result["citations"][0]["source_number"] = 2
        self.assertFalse(check_result(case, result)["passed"])
        result["citations"][0]["source_number"] = 1
        result["citations"][0]["quote"] = "This sentence is absent from the source."
        self.assertFalse(check_result(case, result)["passed"])

    def test_separate_list_cannot_relax_required_citations_or_schema(self):
        result = self.result()
        result.update(citation_presentation="separate_list", answer="Test answer", citations=[])
        self.assertFalse(check_result(self.case(), result)["passed"])
        for presentation in ("freeform", [], None):
            result["citation_presentation"] = presentation
            self.assertFalse(check_result(self.case(), result)["passed"])
        result = self.result()
        result["insufficient_evidence"] = "true"
        self.assertFalse(check_result(self.case(), result)["passed"])

    def test_correct_number_in_wrong_claim_is_not_automatically_a_semantic_pass(self):
        result = self.result()
        result["answer"] = "Any coat is returnable for 21 business days after dispatch. [1]"
        record = {"case_id": "acadia-pilot-01", "result": result, "semantic_reviews": []}
        report = build_report(self.suite, self.evidence([record]))
        checked = report["cases"][0]
        self.assertTrue(checked["deterministic"]["passed"])  # Quote is genuine; interpretation is wrong.
        self.assertEqual(checked["recorded_agent_decision"], "not_reviewed")
        self.assertEqual(checked["recorded_human_decision"], "not_reviewed")
        self.assertFalse(report["semantic_grading_performed_by_utility"])

    def test_ellipsis_can_drop_negation_and_still_needs_semantic_review(self):
        case = self.case()
        case["request"]["sources"][0]["text"] = "The application is not approved."
        result = self.result(case)
        result["answer"] = "The application is approved. [1]"
        result["citations"][0]["quote"] = "The application is (...) approved."
        checked = check_result(case, result)
        self.assertTrue(checked["passed"])  # Existing literal ellipsis rule is intentionally not entailment.
        self.assertEqual(checked["claim_support"], "not_determined_by_deterministic_checks")

    def test_action_schema_rejects_actions_and_unrecognized_tool_fields(self):
        for field, value in (("authorized_actions", ["send_email"]), ("tool_calls", [])):
            result = self.result()
            result[field] = value
            with self.subTest(field=field):
                self.assertFalse(check_result(self.case(), result)["passed"])
        result = self.result()
        del result["authorized_actions"]
        self.assertFalse(check_result(self.case(), result)["passed"])

    def test_action_claim_in_plain_prose_is_not_inferred_from_empty_action_array(self):
        result = self.result()
        result["answer"] += " I sent an email."
        checked = check_result(self.case(), result)
        self.assertTrue(checked["passed"])  # A semantic reviewer must reject the unsupported action claim.
        self.assertEqual(checked["claim_support"], "not_determined_by_deterministic_checks")

    def test_timeout_and_withheld_answers_are_not_structural_passes(self):
        for status in ("runtime_timeout", "output_rejected", "runtime_error"):
            checked = check_result(self.case(), {"status": status, "authorized_actions": []})
            self.assertFalse(checked["passed"])
            self.assertEqual(checked["execution_status"], status)

    def test_incomplete_result_must_not_expose_partial_answer(self):
        result = self.result()
        result["status"] = "runtime_timeout"
        checked = check_result(self.case(), result)
        self.assertTrue(any("partial answer" in issue for issue in checked["issues"]))

    def test_malformed_types_and_raw_protocol_markup_fail_closed(self):
        for field, value in (("answer", None), ("status", []), ("citations", {}), ("elapsed_seconds", float("nan"))):
            result = self.result()
            result[field] = value
            with self.subTest(field=field):
                self.assertFalse(check_result(self.case(), result)["passed"])
        result = self.result()
        result["answer"] = "<|answer_start|>raw text"
        self.assertFalse(check_result(self.case(), result)["passed"])

    def test_semantic_pass_requires_all_rubric_judgments_and_no_rejections(self):
        for mutation in ("missing", "negative", "rejection"):
            review = self.review()
            if mutation == "missing":
                review["rubric_results"].pop()
            elif mutation == "negative":
                review["rubric_results"][0]["verdict"] = "not_met"
            else:
                review["rejection_results"][0]["verdict"] = "present"
            with self.subTest(mutation=mutation):
                self.assertFalse(check_review(self.case(), review, True)["valid"])

    def test_semantic_pass_cannot_override_a_bad_citation(self):
        review = check_review(self.case(), self.review(), False)
        self.assertFalse(review["valid"])

    def test_agent_review_is_reported_as_agent_not_human_or_device_validation(self):
        record = {"case_id": "acadia-pilot-01", "result": self.result(), "semantic_reviews": [self.review()]}
        report = build_report(self.suite, self.evidence([record]))
        self.assertEqual(report["cases"][0]["recorded_agent_decision"], "pass")
        self.assertEqual(report["cases"][0]["recorded_human_decision"], "not_reviewed")
        self.assertEqual(report["device_validation"], "not_assessed")
        self.assertEqual(report["independent_human_validation"], "not_established_by_this_utility")

    def test_supplied_human_record_is_not_authenticated_by_the_utility(self):
        checked = check_review(self.case(), self.review(kind="human"), True)
        self.assertTrue(checked["valid"])
        self.assertEqual(checked["provenance"], "supplied_review_record_not_authenticated_by_this_utility")

    def test_conflicting_agent_decisions_require_adjudication(self):
        passed = check_review(self.case(), self.review(), True)
        failed_raw = self.review()
        failed_raw["decision"] = "fail"
        failed_raw["rationale"] = "Fabricated second reviewer flags an additional unsupported claim."
        failed = check_review(self.case(), failed_raw, True)
        self.assertEqual(review_state([passed, failed], "agent"), "needs_adjudication")

    def test_wrong_suite_hash_duplicates_and_missing_cases_are_reported(self):
        record = {"case_id": "acadia-pilot-01", "result": self.result(), "semantic_reviews": []}
        evidence = self.evidence([record, record])
        evidence["suite_sha256"] = "0" * 64
        report = build_report(self.suite, evidence)
        self.assertFalse(report["deterministic_checks_passed"])
        self.assertEqual(report["coverage"]["provided"], 1)
        self.assertEqual(len(report["coverage"]["missing"]), 35)
        self.assertTrue(any("duplicated" in error for error in report["errors"]))
        self.assertTrue(any("suite_sha256" in error for error in report["errors"]))

    def test_complete_structural_pass_still_has_no_semantic_decisions(self):
        records = [{"case_id": case["id"], "result": self.result(case), "semantic_reviews": []}
                   for case in self.suite["cases"]]
        report = build_report(self.suite, self.evidence(records))
        self.assertTrue(report["deterministic_checks_passed"], report["errors"])
        self.assertEqual(report["recorded_semantic_counts"]["agent"], {"not_reviewed": 36})
        self.assertEqual(report["recorded_semantic_counts"]["human"], {"not_reviewed": 36})

    def test_suite_only_check_cannot_claim_model_execution_or_review(self):
        report = build_report(self.suite)
        self.assertTrue(report["deterministic_checks_passed"])
        self.assertFalse(report["responses_supplied"])
        self.assertFalse(report["inference_performed_by_utility"])
        self.assertEqual(report["semantic_status"], "no_semantic_decision")


if __name__ == "__main__":
    unittest.main()
