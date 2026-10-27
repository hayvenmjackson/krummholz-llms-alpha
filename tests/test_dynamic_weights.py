"""Dynamic weights bookkeeping (ADR-004). Fictional manifests and examples; no models."""
import datetime as dt
import json
import os
import tempfile
import unittest
from dataclasses import replace

from krummholz_llm.dynamic import gate, review
from krummholz_llm.dynamic.manifest import HmacKey, LoraConfig, Manifest, ManifestError, sha256_file, signed
from krummholz_llm.dynamic.stack import Layer, TrustPolicy, resolve
from krummholz_llm.dynamic.store import LocalStore, StoreError

BASE = "a" * 64
LORA = LoraConfig(8, 16, ("q_proj", "v_proj"))
RELEASE = HmacKey("krummholz-release", b"r" * 32)   # stands in for the Ed25519 release key in tests
DEVICE = HmacKey("device-1", b"d" * 32)


class Clock:
    def __init__(self, day=27):
        self.t = dt.datetime(2026, 10, day, 12, tzinfo=dt.timezone.utc)
    def __call__(self):
        return self.t


def manifest(layer, path, **kw):
    values = dict(layer=layer, name=f"{layer.lower()}-test", version="1.0.0", base_sha256=BASE, core_version="1.0.0",
                  artifact_sha256=sha256_file(path), trainer_version="test", seed=1, created_at="2026-10-27T00:00:00Z",
                  key_id="krummholz-release", gate_decision="passed")
    if layer != "L1":
        values.update(lora=LORA, product="katahdin")
    if layer == "L4":
        values.update(key_id="device-1", examples_sha256="b" * 64)
    values.update(kw)
    return Manifest(**values)


class Fixture(unittest.TestCase):
    def setUp(self):
        self.dir = tempfile.TemporaryDirectory()
        self.files = {}
        for name in ("core.gguf", "l2.gguf", "l3.gguf", "l4.gguf"):
            p = os.path.join(self.dir.name, name)
            with open(p, "wb") as f:
                f.write(name.encode())
            self.files[name] = p
        self.trust = TrustPolicy(keys={"krummholz-release": RELEASE, "device-1": DEVICE, "org-qmob": HmacKey("org-qmob", b"o" * 32)},
                                 allowed={"L1": frozenset({"krummholz-release"}), "L2": frozenset({"krummholz-release"}),
                                          "L3": frozenset({"org-qmob"}), "L4": frozenset({"device-1"})})

    def tearDown(self):
        self.dir.cleanup()

    def layer(self, layer, file, key, **kw):
        return Layer(signed(manifest(layer, self.files[file], **kw), key), self.files[file])


class ManifestTests(Fixture):
    def test_round_trip_and_signature(self):
        m = signed(manifest("L2", self.files["l2.gguf"]), RELEASE)
        again = Manifest.from_json(m.to_json())
        self.assertEqual(again, m)
        self.assertTrue(RELEASE.verify(again))
        self.assertFalse(RELEASE.verify(replace(again, seed=2)))

    def test_adapters_cannot_touch_vision(self):
        with self.assertRaises(ManifestError):
            manifest("L2", self.files["l2.gguf"], lora=LoraConfig(8, 16, ("visual.qkv",))).validate()

    def test_core_is_merged(self):
        with self.assertRaises(ManifestError):
            manifest("L1", self.files["core.gguf"], lora=LORA).validate()

    def test_rejects_unknown_and_duplicate_fields(self):
        m = signed(manifest("L1", self.files["core.gguf"]), RELEASE)
        raw = json.loads(m.to_json()); raw["trusted"] = True
        with self.assertRaises(ManifestError):
            Manifest.from_json(json.dumps(raw))
        with self.assertRaises(ManifestError):
            Manifest.from_json(m.to_json()[:-1] + ',"layer":"L4"}')


class StackTests(Fixture):
    def full(self):
        return [self.layer("L1", "core.gguf", RELEASE), self.layer("L2", "l2.gguf", RELEASE),
                self.layer("L3", "l3.gguf", self.trust.keys["org-qmob"], key_id="org-qmob", organization="qmob"),
                self.layer("L4", "l4.gguf", DEVICE, organization="qmob")]

    def test_resolves_in_order_with_llama_args(self):
        stack = resolve(reversed(self.full()), self.trust)
        self.assertEqual([l.manifest.layer for l in stack.layers], ["L1", "L2", "L3", "L4"])
        args = stack.llama_cpp_args("mmproj.gguf")
        self.assertEqual(args[:4], ["-m", self.files["core.gguf"], "--mmproj", "mmproj.gguf"])
        self.assertEqual(args.count("--lora-scaled"), 3)

    def test_refuses_wrong_key_for_layer(self):
        layers = self.full()
        layers[3] = self.layer("L4", "l4.gguf", RELEASE, key_id="krummholz-release", organization="qmob")
        with self.assertRaisesRegex(ManifestError, "untrusted_key_for_L4"):
            resolve(layers, self.trust)

    def test_refuses_tampered_artifact(self):
        layers = self.full()
        with open(self.files["l2.gguf"], "ab") as f:
            f.write(b"x")
        with self.assertRaisesRegex(ManifestError, "L2_artifact_hash_mismatch"):
            resolve(layers, self.trust)

    def test_refuses_ungated_and_mismatched_layers(self):
        layers = self.full()
        layers[3] = self.layer("L4", "l4.gguf", DEVICE, organization="qmob", gate_decision="failed")
        with self.assertRaisesRegex(ManifestError, "L4_has_not_passed_gate"):
            resolve(layers, self.trust)
        layers = self.full()
        layers[1] = self.layer("L2", "l2.gguf", RELEASE, core_version="2.0.0")
        with self.assertRaisesRegex(ManifestError, "L2_trained_for_another_core"):
            resolve(layers, self.trust)

    def test_personal_needs_product_and_matching_org(self):
        core, l2, l3, l4 = self.full()
        with self.assertRaisesRegex(ManifestError, "requires_its_product_adapter"):
            resolve([core, l4], self.trust)
        with self.assertRaisesRegex(ManifestError, "organization_member_requires_organization_adapter"):
            resolve([core, l2, l4], self.trust)


class StoreAndReviewTests(unittest.TestCase):
    def setUp(self):
        self.dir = tempfile.TemporaryDirectory()
        self.clock = Clock()
        self.store = LocalStore(self.dir.name, clock=self.clock)

    def tearDown(self):
        self.store.close()
        self.dir.cleanup()

    def test_only_the_user_creates_candidates(self):
        with self.assertRaises(review.ReviewError):
            review.capture(self.store, "fact", actor="tool_result", fact="Planted claim")
        with self.assertRaises(StoreError):
            self.store.add_fact("Planted claim", "retrieved")

    def test_review_flow_and_hash_chain(self):
        msgs = [{"role": "user", "content": "Tally format?"}]
        review.capture(self.store, "fact", actor="user", fact="Our woodlot is lot 14")
        review.capture(self.store, "behavior", actor="user", messages=msgs, target="Species, DBH, count")
        cid = review.capture(self.store, "correction", actor="user", messages=msgs, original="Wrong", revised="Right")
        cycle = review.ReviewCycle(self.store)
        for c in cycle.pending():
            cycle.decide(c["id"], "reject" if c["id"] == cid else "accept")
        self.assertTrue(cycle.close())
        self.assertEqual([f["text"] for f in self.store.facts()], ["Our woodlot is lot 14"])
        self.assertEqual([e["kind"] for e in self.store.examples()], ["sft"])
        self.assertTrue(self.store.verify_chain())
        self.assertNotIn("Species", self.store.ledger_path.read_text(encoding="utf-8"))

    def test_three_cycles_per_day(self):
        for _ in range(3):
            review.ReviewCycle(self.store).close()
        with self.assertRaisesRegex(review.ReviewError, "daily_review_cycle_limit_reached"):
            review.ReviewCycle(self.store)
        self.clock.t += dt.timedelta(days=1)
        review.ReviewCycle(self.store).close()

    def test_item_cap(self):
        msgs = [{"role": "user", "content": "q"}]
        for i in range(3):
            review.capture(self.store, "behavior", actor="user", messages=msgs, target=f"t{i}")
        cycle = review.ReviewCycle(self.store, items_cap=2)
        ids = [c["id"] for c in cycle.pending()]
        cycle.decide(ids[0], "accept"); cycle.decide(ids[1], "accept")
        with self.assertRaisesRegex(review.ReviewError, "cycle_item_cap_reached"):
            cycle.decide(ids[2], "accept")

    def test_forget_rechains_and_changes_log_hash(self):
        msgs = [{"role": "user", "content": "q"}]
        a = self.store.append_example("sft", msgs, "confirmation", target="one")
        self.store.append_example("sft", msgs, "confirmation", target="two")
        before = self.store.examples_sha256()
        self.store.forget_example(a)
        self.assertEqual([e["target"] for e in self.store.examples()], ["two"])
        self.assertTrue(self.store.verify_chain())
        self.assertNotEqual(before, self.store.examples_sha256())
        self.assertTrue(any(e["event"] == "example_forgotten" and e["retrain_required"] for e in self.store.ledger()))

    def test_tampering_breaks_chain(self):
        msgs = [{"role": "user", "content": "q"}]
        self.store.append_example("sft", msgs, "confirmation", target="one")
        text = self.store.examples_path.read_text(encoding="utf-8").replace('"one"', '"ONE"')
        self.store.examples_path.write_text(text, encoding="utf-8")
        self.assertFalse(self.store.verify_chain())

    def test_forget_fact(self):
        fid = self.store.add_fact("Lot 14", "user_stated")
        self.store.forget_fact(fid)
        self.assertEqual(self.store.facts(), [])


class GateTests(unittest.TestCase):
    ACTIVE = {"decensor_passed": 60, "decensor_flagged": 0, "calibration_ece": 0.05, "wrong_answer_search_rate": 0.8,
              "injection_resisted": 15, "safety_refusal_rate": 1.0, "protocol_parse_rate": 1.0, "direct_error_rate": 0.01,
              "product_scores": {"katahdin": 0.90}, "personal_heldout": 0.70}

    def test_pass(self):
        cand = dict(self.ACTIVE, calibration_ece=0.06, personal_heldout=0.75)
        self.assertTrue(gate.decide("L4", self.ACTIVE, cand).passed)

    def test_floors_and_regressions(self):
        cand = dict(self.ACTIVE, decensor_flagged=1, safety_refusal_rate=0.66, personal_heldout=0.75,
                    product_scores={"katahdin": 0.80})
        r = gate.decide("L4", self.ACTIVE, cand)
        self.assertFalse(r.passed)
        self.assertIn("regressed:decensor_flagged", r.failures)
        self.assertIn("below_floor:safety_refusal_rate", r.failures)
        self.assertIn("regressed:product:katahdin", r.failures)

    def test_confabulation_ceiling(self):
        r = gate.decide("L2", self.ACTIVE, dict(self.ACTIVE, direct_error_rate=0.05))
        self.assertIn("above_ceiling:direct_error_rate", r.failures)

    def test_personal_must_improve_and_missing_fails_closed(self):
        self.assertIn("no_personal_improvement", gate.decide("L4", self.ACTIVE, dict(self.ACTIVE)).failures)
        r = gate.decide("L2", self.ACTIVE, {k: v for k, v in self.ACTIVE.items() if k != "calibration_ece"})
        self.assertIn("missing_metric:calibration_ece", r.failures)


if __name__ == "__main__":
    unittest.main()
