"""Generation loop with the deterministic mock backend: learning, crash/resume determinism,
journal replay, budget and stop guards, config-change refusal."""
import json
import os
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np
import yaml

from es.data import LeakageError
from es.es_loop import ESRun, PausedRun, SimulatedCrash, array_sha256, deep_merge, load_config, main, replay
from es.fitness import CallableGrader, MockGrader
from es.tests.helpers import synthetic_adapter, toy_shapes, write_pairs

REPO = Path(__file__).resolve().parents[2]
VOLATILE = {"time", "gen_seconds", "generate_seconds", "seconds", "updated", "replies_generated_this_session",
            "elapsed_hours", "usd_so_far"}


def strip(rec):
    return {k: v for k, v in rec.items() if k not in VOLATILE}


def read_jsonl(p):
    return [json.loads(x) for x in Path(p).read_text().splitlines() if x.strip()] if Path(p).exists() else []


class FakeClock:
    def __init__(self, step=10.0):
        self.t = 0.0
        self.step = step

    def __call__(self):
        self.t += self.step
        return self.t


class LoopTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.dir = Path(self.tmp.name)
        self.src = synthetic_adapter(self.dir / "src", toy_shapes(), rank=4, seed=1)
        self.pool = write_pairs(self.dir / "pool.jsonl", 20)
        self.dev = write_pairs(self.dir / "dev.jsonl", 6, prefix="d", split="dev")
        ret = [{"id": f"r{i}", "split": "retention", "prompt": "p", "checks": {"any_of": ["SCORE"]}} for i in range(6)]
        self.ret = self.dir / "ret.jsonl"
        self.ret.write_text("".join(json.dumps(r) + "\n" for r in ret))
        os.environ.pop("ES_FAULT", None)
        os.environ.pop("ES_FAULT_MODE", None)

    def tearDown(self):
        os.environ.pop("ES_FAULT", None)
        os.environ.pop("ES_FAULT_MODE", None)
        self.tmp.cleanup()

    def cfg(self, **over):
        base = deep_merge(load_config(None), {
            "run": {"name": "mock", "seed": 5},
            "essa": {"adapter": str(self.src), "parameterization": "relative"},
            "es": {"pairs": 6, "sigma": 0.05, "generations": 6, "awd": 0.01,
                   "optimizer": {"kind": "adam", "lr": 0.02}},
            "data": {"train_pool": str(self.pool), "pairs_per_batch": 4,
                     "retention": {"path": str(self.ret), "per_batch": 2}},
            "fitness": {"grader": {"kind": "mock"}},
            "validation": {"dev": str(self.dev), "every": 3, "pairs": 3},
            "checkpoint": {"keep_last": 2, "export_every": 3},
            "backend": {"kind": "mock", "target_norm": 0.3, "wave_size": 5},
        })
        return deep_merge(base, over)

    def run_to_end(self, run_dir, cfg=None, clock=None):
        run = ESRun(cfg or self.cfg(), run_dir, clock=clock or FakeClock(), log=lambda s: None)
        run.prepare()
        return run, run.run()

    def assert_same_run(self, a, b):
        ja, jb = read_jsonl(a / "journal.jsonl"), read_jsonl(b / "journal.jsonl")
        self.assertEqual(len(ja), len(jb))
        for x, y in zip(ja, jb):
            self.assertEqual(strip(x), strip(y), f"journal gen {x['gen']}")
        self.assertEqual([strip(x) for x in read_jsonl(a / "val.jsonl")], [strip(x) for x in read_jsonl(b / "val.jsonl")])
        fa = (a / "final-adapter" / "adapter_model.safetensors").read_bytes()
        fb = (b / "final-adapter" / "adapter_model.safetensors").read_bytes()
        self.assertEqual(fa, fb)

    # ------------------------------------------------------------------ tests
    def test_es_moves_toward_target(self):
        cfg = self.cfg(es={"pairs": 8, "generations": 25, "awd": 0.0, "optimizer": {"lr": 0.02}},
                       validation={"every": 0}, checkpoint={"export_every": 0})
        run, result = self.run_to_end(self.dir / "learn", cfg)
        self.assertEqual(result["stop_reason"], "generations")
        target = run.backend.target
        self.assertLess(np.linalg.norm(run.state.theta - target), 0.5 * np.linalg.norm(target))
        j = read_jsonl(self.dir / "learn" / "journal.jsonl")
        self.assertGreater(np.mean([e["parent_fitness"] for e in j[-5:]]), np.mean([e["parent_fitness"] for e in j[:5]]))
        self.assertEqual(j[0]["replies"], 17 * 10)  # (2 * 8 + 1) candidates x batch of 10

    def test_resume_after_crash_is_deterministic(self):
        ref = self.dir / "ref"
        self.run_to_end(ref)
        self.assertEqual(replay(ref)["checkpoint_matches"], True)
        for spec in ("after_wave:2:1", "after_wave:0:0", "before_journal:1", "after_journal:3",
                     "before_checkpoint:4", "after_checkpoint:2"):
            with self.subTest(fault=spec):
                d = self.dir / ("crash-" + spec.replace(":", "_"))
                os.environ["ES_FAULT"] = spec
                with self.assertRaises(SimulatedCrash):
                    self.run_to_end(d)
                os.environ.pop("ES_FAULT")
                run, result = self.run_to_end(d)
                self.assertEqual(result["stop_reason"], "generations")
                self.assert_same_run(ref, d)
                r = replay(d)
                self.assertTrue(r["checkpoint_matches"])
                self.assertEqual(r["theta_sha256"], array_sha256(run.state.theta))

    def test_partial_results_are_reused_after_crash(self):
        d = self.dir / "partial"
        os.environ["ES_FAULT"] = "after_wave:1:1"
        with self.assertRaises(SimulatedCrash):
            self.run_to_end(d)
        os.environ.pop("ES_FAULT")
        saved = json.loads((d / "partial" / "gen-000001.json").read_text())
        self.assertEqual(len(saved["scores"]), 10)  # two waves of five candidates
        run = ESRun(self.cfg(), d, clock=FakeClock(), log=lambda s: None)
        run.prepare()
        run.run()
        self.assertFalse((d / "partial" / "gen-000001.json").exists())

    def test_hard_kill_in_subprocess(self):
        ref = self.dir / "ref2"
        self.run_to_end(ref)
        cfg_path = self.dir / "cfg.yaml"
        cfg_path.write_text(yaml.safe_dump(self.cfg()))
        d = self.dir / "killed"
        env = dict(os.environ, ES_FAULT="after_wave:3:0", ES_FAULT_MODE="exit", PYTHONPATH=str(REPO))
        cmd = [sys.executable, "-m", "es.es_loop", "run", "--config", str(cfg_path), "--run-dir", str(d)]
        p = subprocess.run(cmd, cwd=REPO, env=env, capture_output=True, text=True)
        self.assertEqual(p.returncode, 17, p.stderr[-2000:])
        env.pop("ES_FAULT")
        env.pop("ES_FAULT_MODE")
        p = subprocess.run(cmd, cwd=REPO, env=env, capture_output=True, text=True)
        self.assertEqual(p.returncode, 0, p.stderr[-2000:])
        # the mock backend is exact, so a hard kill and resume matches the in-process reference
        self.assert_same_run(ref, d)
        p = subprocess.run([sys.executable, "-m", "es.es_loop", "replay", "--run-dir", str(d)], cwd=REPO, env=env,
                           capture_output=True, text=True)
        self.assertEqual(p.returncode, 0, p.stdout + p.stderr)

    def test_wall_clock_and_budget_guards(self):
        cfg = self.cfg(es={"generations": 50}, validation={"every": 0},
                       budget={"max_wall_hours": 0.05, "reserve_minutes": 0})
        run, result = self.run_to_end(self.dir / "wall", cfg, clock=FakeClock(step=10.0))
        self.assertEqual(result["stop_reason"], "wall_clock")
        self.assertLess(run.state.gen, 50)
        self.assertTrue((self.dir / "wall" / "final-adapter" / "adapter_model.safetensors").exists())
        self.assertEqual(json.loads((self.dir / "wall" / "SUMMARY.json").read_text())["stop_reason"], "wall_clock")
        self.assertLessEqual(run.elapsed_total(), 0.05 * 3600 + 200)
        cfg = self.cfg(es={"generations": 50}, validation={"every": 0},
                       budget={"usd_per_hour": 4.39, "max_usd": 0.2})
        run, result = self.run_to_end(self.dir / "usd", cfg, clock=FakeClock(step=10.0))
        self.assertEqual(result["stop_reason"], "budget_usd")
        status = json.loads((self.dir / "usd" / "status.json").read_text())
        self.assertLessEqual(status["usd_so_far"], 0.2 + 0.1)

    def test_no_start_after_usd(self):
        cfg = self.cfg(es={"generations": 50}, validation={"every": 0},
                       budget={"usd_per_hour": 4.39, "no_start_after_usd": 0.3})
        run, result = self.run_to_end(self.dir / "nostart", cfg, clock=FakeClock(step=10.0))
        self.assertEqual(result["stop_reason"], "budget_usd")
        self.assertGreaterEqual(run.spent_usd(), 0.3)

    def test_dev_gate_and_pace_gate(self):
        d = self.dir / "gate"
        run, result = self.run_to_end(d, self.cfg(gates={"dev_gate_gen": 3, "dev_gate_se": 1e9}))
        self.assertEqual(result["stop_reason"], "dev_gate")
        self.assertEqual(run.state.gen, 3)
        gate = read_jsonl(d / "gates.jsonl")[-1]
        self.assertFalse(gate["dev_gate"]["passed"])
        self.assertEqual(gate["dev_gate"]["clusters"], 3)
        self.assertTrue((d / "final-adapter" / "ES_PROVENANCE.json").exists())
        run, result = self.run_to_end(self.dir / "gate-pass", self.cfg(gates={"dev_gate_gen": 3, "dev_gate_se": -1e9}))
        self.assertEqual(result["stop_reason"], "generations")
        cfg = self.cfg(gates={"pace_gen": 3, "pace_min_generations": 10 ** 6},
                       budget={"usd_per_hour": 4.39, "max_usd": 50})
        run, result = self.run_to_end(self.dir / "pace", cfg)
        self.assertEqual(result["stop_reason"], "pace")

    def test_engineering_pauses(self):
        d = self.dir / "pause"
        mock = MockGrader()

        def flaky(item, reply):
            if str(item.get("variant")) == "absent":
                raise ValueError("grader bug")
            return mock.grade(item, reply)

        run = ESRun(self.cfg(validation={"every": 0}), d, grader=CallableGrader(flaky), clock=FakeClock(),
                    log=lambda s: None)
        run.prepare()
        with self.assertRaises(PausedRun):
            run.run()
        self.assertEqual(json.loads((d / "status.json").read_text())["state"], "paused")
        self.assertFalse((d / "partial" / "gen-000000.json").exists())
        self.assertTrue(list((d / "partial").glob("gen-000000.paused-*.json")))
        self.assertEqual(read_jsonl(d / "journal.jsonl"), [])
        # resuming with a fixed grader recomputes the generation and runs to the end
        run = ESRun(self.cfg(validation={"every": 0}), d, clock=FakeClock(), log=lambda s: None)
        run.prepare()
        self.assertEqual(run.run()["stop_reason"], "generations")

    def test_calibration_recommendation(self):
        cfg = self.cfg(es={"optimizer": {"kind": "sgd", "lr_sigma2_coef": 1.7}}, checkpoint={"save_replies": True})
        run = ESRun(cfg, self.dir / "calib", clock=FakeClock(), log=lambda s: None)
        run.prepare(pin_config=False)
        out = run.calibrate([0.30, 0.02, 0.05], directions=2)
        self.assertEqual([r["sigma"] for r in out["rows"]], [0.02, 0.05, 0.30])
        self.assertEqual(out["recommended_sigma"], 0.02)
        self.assertAlmostEqual(out["recommended_lr"], 1.7 * 0.02 ** 2)
        self.assertEqual(run.state.gen, 0)
        self.assertEqual(len(list((self.dir / "calib" / "replies" / "calibrate").glob("*.jsonl.gz"))), 13)
        # calibration does not pin the config: the run can start with the recommended sigma
        self.assertFalse((self.dir / "calib" / "config.json").exists())
        cfg2 = deep_merge(cfg, {"es": {"sigma": out["recommended_sigma"], "generations": 1}})
        run2 = ESRun(cfg2, self.dir / "calib", clock=FakeClock(), log=lambda s: None)
        run2.prepare()
        self.assertEqual(run2.run()["stop_reason"], "generations")

    def test_stop_file(self):
        d = self.dir / "stopped"
        d.mkdir()
        (d / "STOP").write_text("")
        run, result = self.run_to_end(d)
        self.assertEqual(result["stop_reason"], "stop_file")
        self.assertEqual(run.state.gen, 0)

    def test_config_change_refused_but_budget_change_allowed(self):
        d = self.dir / "cfgchange"
        self.run_to_end(d, self.cfg(es={"generations": 2}))
        with self.assertRaises(RuntimeError):
            ESRun(self.cfg(es={"generations": 2, "sigma": 0.07}), d, log=lambda s: None).prepare()
        ESRun(self.cfg(es={"generations": 2}, budget={"max_usd": 99}), d, log=lambda s: None).prepare()

    def test_validation_is_monitoring_only(self):
        d = self.dir / "val"
        run, _ = self.run_to_end(d)
        val = read_jsonl(d / "val.jsonl")
        self.assertEqual([v["gen"] for v in val], [0, 3, 6])
        self.assertTrue(all(v["roundtrip_rel_err"] < 1e-6 for v in val))
        per_item = json.loads((d / "val" / "gen-000003.json").read_text())
        self.assertEqual(len(per_item["rewards"]), 6 + 0)  # 3 dev pairs
        self.assertEqual(len(per_item["correct"]), 6)
        # the final adapter is the final theta, whatever the validation history says
        from safetensors.torch import load_file
        theta = run.svd.project(load_file(str(d / "final-adapter" / "adapter_model.safetensors"))).numpy()
        self.assertTrue(np.allclose(theta, run.state.theta, atol=1e-5))
        self.assertTrue((d / "exports" / "gen-000003" / "ES_PROVENANCE.json").exists())


    # ------------------------------------------------------------ review fixes
    def write_dev_retention(self, n=4, prefix="dr"):
        path = self.dir / f"{prefix}.jsonl"
        rows = [{"id": f"{prefix}{i}", "split": "retention", "prompt": f"dev retention {i}",
                 "checks": {"any_of": ["SCORE"]}} for i in range(n)]
        path.write_text("".join(json.dumps(r) + "\n" for r in rows))
        return path

    def test_dev_retention_comes_from_its_own_pool(self):
        devret = self.write_dev_retention()
        d = self.dir / "devret"
        run, _ = self.run_to_end(d, self.cfg(validation={"retention": 2, "retention_path": str(devret)}))
        ids = json.loads((d / "val" / "gen-000000.json").read_text())["ids"]
        self.assertEqual(len(ids), 6 + 2)
        self.assertTrue(all(not i.startswith("r") or i.startswith("dr") for i in ids), ids)  # no training retention
        self.assertEqual(sum(i.startswith("dr") for i in ids), 2)
        with self.assertRaises(ValueError):   # dev retention without its own pool is refused
            ESRun(self.cfg(validation={"retention": 2}), self.dir / "devret2", log=lambda s: None).prepare()
        with self.assertRaises(LeakageError):  # pointing the dev pool at the training retention file
            ESRun(self.cfg(validation={"retention": 2, "retention_path": str(self.ret)}), self.dir / "devret3",
                  log=lambda s: None).prepare()
        with self.assertRaises(LeakageError):  # dev pairs that are training pairs
            ESRun(self.cfg(validation={"dev": str(self.pool)}), self.dir / "devret4", log=lambda s: None).prepare()

    def test_roundtrip_pause_is_not_bypassed_on_resume(self):
        d = self.dir / "rt"
        for _ in range(2):
            run = ESRun(self.cfg(), d, clock=FakeClock(), log=lambda s: None)
            run.prepare()
            run.roundtrip_error = lambda: 0.5
            with self.assertRaises(PausedRun):
                run.run()
            self.assertEqual(read_jsonl(d / "val.jsonl"), [])   # nothing recorded for a paused check
        run = ESRun(self.cfg(), d, clock=FakeClock(), log=lambda s: None)
        run.prepare()
        self.assertEqual(run.run()["stop_reason"], "generations")

    def test_dev_check_grader_errors_pause(self):
        mock = MockGrader()

        def flaky(item, reply):
            if str(item.get("group", "")).startswith("ys1-d-"):
                raise ValueError("held-out template the grader cannot parse")
            return mock.grade(item, reply)

        run = ESRun(self.cfg(), self.dir / "deverr", grader=CallableGrader(flaky), clock=FakeClock(),
                    log=lambda s: None)
        run.prepare()
        with self.assertRaises(PausedRun) as ctx:
            run.run()
        self.assertIn("dev check", str(ctx.exception))
        self.assertEqual(read_jsonl(self.dir / "deverr" / "val.jsonl"), [])

    def test_gates_run_off_the_validation_cadence(self):
        # G4 at generation 5 with dev checks every 4: generations 0 and 5 are still validated
        d = self.dir / "offcadence"
        run, result = self.run_to_end(d, self.cfg(validation={"every": 4}, gates={"dev_gate_gen": 5, "dev_gate_se": 1e9}))
        self.assertEqual(result["stop_reason"], "dev_gate")
        self.assertEqual(run.state.gen, 5)
        self.assertEqual([v["gen"] for v in read_jsonl(d / "val.jsonl")], [0, 4, 5])
        # G5 without any dev batch is still evaluated
        cfg = self.cfg(validation={"dev": None, "every": 0}, gates={"pace_gen": 2, "pace_min_generations": 10 ** 6},
                       budget={"usd_per_hour": 4.39, "max_usd": 50, "no_start_after_usd": 40})
        run, result = self.run_to_end(self.dir / "pace-nodev", cfg)
        self.assertEqual(result["stop_reason"], "pace")
        self.assertEqual(read_jsonl(self.dir / "pace-nodev" / "gates.jsonl")[-1]["pace"]["ceiling_usd"], 40.0)

    def test_unevaluable_gate_config_is_refused(self):
        with self.assertRaises(ValueError):
            ESRun(self.cfg(validation={"dev": None}, gates={"dev_gate_gen": 4}), self.dir / "g1",
                  log=lambda s: None).prepare()
        with self.assertRaises(ValueError):   # S5 needs dev retention
            ESRun(self.cfg(gates={"retention_margin": 3}), self.dir / "g2", log=lambda s: None).prepare()

    def test_engine_numerics_cannot_change_on_resume(self):
        d = self.dir / "numerics"
        self.run_to_end(d, self.cfg(es={"generations": 2}))
        with self.assertRaises(RuntimeError):
            ESRun(self.cfg(es={"generations": 2}, backend={"engine": {"fp8_base": True}}), d,
                  log=lambda s: None).prepare()
        ESRun(self.cfg(es={"generations": 2}, backend={"engine": {"max_loras": 2}}), d, log=lambda s: None).prepare()

    def test_billed_time_survives_a_crash(self):
        d = self.dir / "billing"
        os.environ["ES_FAULT"] = "after_wave:1:1"
        with self.assertRaises(SimulatedCrash):
            self.run_to_end(d)
        os.environ.pop("ES_FAULT")
        billed = json.loads((d / "billing.json").read_text())["elapsed_s"]
        ckpt = json.loads((d / "checkpoints" / "ckpt-000001" / "state.json").read_text())["elapsed_s"]
        self.assertGreater(billed, ckpt)
        run = ESRun(self.cfg(), d, clock=FakeClock(), log=lambda s: None)
        run.prepare()
        self.assertEqual(run.state.elapsed_s, billed)
        self.assertEqual(run.run()["stop_reason"], "generations")
        self.assert_same_run(self.dir / "billing", d)

    def test_hard_stop_inside_a_generation(self):
        cfg = self.cfg(validation={"every": 0}, budget={"usd_per_hour": 4.39, "max_usd": 0.05})
        d = self.dir / "hardkill"
        run, result = self.run_to_end(d, cfg, clock=FakeClock(step=10.0))
        self.assertEqual(result["stop_reason"], "budget_hard_kill")
        self.assertEqual(run.state.gen, 0)
        self.assertEqual(read_jsonl(d / "journal.jsonl"), [])
        self.assertTrue((d / "final-adapter" / "ES_PROVENANCE.json").exists())
        self.assertEqual(json.loads((d / "SUMMARY.json").read_text())["stop_reason"], "budget_hard_kill")

    def test_watchdog_exits_at_max_usd_during_a_slow_generation(self):
        cfg = self.cfg(validation={"every": 0}, backend={"sleep_s": 30.0},
                       budget={"usd_per_hour": 3600.0, "max_usd": 1.0, "heartbeat_seconds": 0.1})
        cfg_path = self.dir / "slow.yaml"
        cfg_path.write_text(yaml.safe_dump(cfg))
        d = self.dir / "watchdog"
        env = dict(os.environ, PYTHONPATH=str(REPO))
        p = subprocess.run([sys.executable, "-m", "es.es_loop", "run", "--config", str(cfg_path), "--run-dir", str(d)],
                           cwd=REPO, env=env, capture_output=True, text=True, timeout=25)
        self.assertEqual(p.returncode, 3, p.stdout[-2000:] + p.stderr[-2000:])
        self.assertEqual(json.loads((d / "status.json").read_text())["state"], "killed")
        self.assertGreaterEqual(json.loads((d / "billing.json").read_text())["elapsed_s"], 1.0)
        self.assertEqual(read_jsonl(d / "journal.jsonl"), [])

    def test_model_load_is_not_counted_as_generation_time(self):
        clock = FakeClock(step=1.0)

        class SlowLoad(ESRun):
            def ensure_backend(self):
                if self.backend is None:
                    clock.t += 5000.0   # a 27B model load
                super().ensure_backend()

        run = SlowLoad(self.cfg(validation={"every": 0}, es={"generations": 2}), self.dir / "load", clock=clock,
                       log=lambda s: None)
        run.prepare()
        run.run()
        gen_seconds = [e["gen_seconds"] for e in read_jsonl(self.dir / "load" / "journal.jsonl")]
        self.assertTrue(all(x < 1000 for x in gen_seconds), gen_seconds)
        self.assertGreater(run.elapsed_total(), 5000)   # still billed

    def test_export_and_aux_commands_leave_a_live_run_alone(self):
        d = self.dir / "live"
        os.environ["ES_FAULT"] = "after_journal:2"
        with self.assertRaises(SimulatedCrash):
            self.run_to_end(d)
        os.environ.pop("ES_FAULT")
        journal = (d / "journal.jsonl").read_bytes()
        config = (d / "config.json").read_bytes()
        live = d / "es-candidates-mock"
        (live / "in-flight-adapter").mkdir(parents=True, exist_ok=True)
        self.assertEqual(main(["export", "--run-dir", str(d), "--out", str(self.dir / "exported")]), 0)
        cfg_path = self.dir / "live.yaml"
        cfg_path.write_text(yaml.safe_dump(self.cfg()))
        self.assertEqual(main(["preflight", "--config", str(cfg_path), "--run-dir", str(d)]), 0)
        self.assertEqual((d / "journal.jsonl").read_bytes(), journal)   # gen 2 line not truncated
        self.assertEqual((d / "config.json").read_bytes(), config)
        self.assertTrue((live / "in-flight-adapter").exists())
        self.assertFalse(list(d.glob("es-candidates-mock-aux-*")))
        self.assertTrue((self.dir / "exported" / "ES_PROVENANCE.json").exists())

    def test_aux_command_time_counts_against_the_budget(self):
        d = self.dir / "auxbill"
        cfg_path = self.dir / "auxbill.yaml"
        cfg_path.write_text(yaml.safe_dump(self.cfg(budget={"usd_per_hour": 4.39, "overhead_hours": 0.5})))
        for cmd in ("preflight", "parity"):
            main([cmd, "--config", str(cfg_path), "--run-dir", str(d), "--set", "gates.parity_pairs=2"])
        bill = json.loads((d / "billing.json").read_text())
        self.assertEqual(sorted(bill["aux_by_command"]), ["parity", "preflight"])
        self.assertGreater(bill["aux_s"], 0.0)
        run = ESRun(self.cfg(budget={"usd_per_hour": 4.39, "overhead_hours": 0.5}), d, clock=FakeClock(),
                    log=lambda s: None)
        run.prepare()
        self.assertAlmostEqual(run._extra_hours(), 0.5 + bill["aux_s"] / 3600.0)
        run.run()
        self.assertEqual(json.loads((d / "billing.json").read_text())["aux_s"], bill["aux_s"])  # run keeps it

    def test_second_run_process_is_refused(self):
        try:
            import fcntl  # noqa: F401
        except ImportError:
            self.skipTest("no flock on this platform")
        d = self.dir / "locked"
        d.mkdir()
        code = ("import fcntl, sys, time\n"
                "f = open(sys.argv[1], 'a+'); fcntl.flock(f.fileno(), fcntl.LOCK_EX)\n"
                "print('locked', flush=True); time.sleep(30)\n")
        holder = subprocess.Popen([sys.executable, "-c", code, str(d / "RUN.lock")], stdout=subprocess.PIPE, text=True)
        try:
            self.assertEqual(holder.stdout.readline().strip(), "locked")
            with self.assertRaises(RuntimeError):
                ESRun(self.cfg(), d, log=lambda s: None).prepare()
            self.assertFalse((d / "journal.jsonl").exists())
        finally:
            holder.kill()
            holder.wait()

    def test_coverage_finds_module_groups_the_backend_ignores(self):
        from safetensors.torch import load_file

        from es.backends.base import Backend, make_reply

        class QOnly(Backend):
            """Applies only q_proj LoRA weights, like an engine that silently skips other module kinds."""
            def generate(self, adapters, prompts):
                out = []
                for ad in adapters:
                    sd = load_file(str(Path(ad.path) / "adapter_model.safetensors"))
                    v = sum(float(t.double().sum()) for k, t in sorted(sd.items()) if "q_proj" in k)
                    out.append([make_reply(f"<think>t</think>SCORE {v:.12f}", "stop", 3, False) for _ in prompts])
                return out

        run = ESRun(self.cfg(), self.dir / "cov", backend=QOnly(), log=lambda s: None)
        run.prepare(pin_config=False)
        res = run.coverage(n_prompts=4)
        self.assertFalse(res["passed"])
        self.assertEqual(sorted(g.rsplit(".", 1)[-1] for g in res["ignored"]), ["down_proj", "v_proj"])
        self.assertEqual(len(res["groups"]), 3)
        ok = ESRun(self.cfg(), self.dir / "cov2", log=lambda s: None)
        ok.prepare(pin_config=False)
        self.assertTrue(ok.coverage(n_prompts=4)["passed"])

    def test_require_g0_blocks_launch_until_parity_and_coverage_pass(self):
        # the mock prints scores to 9 decimals, so float32 rebuild noise flips digits: parity accuracy is
        # tested with the real tiny model (test_hf_tiny); this test is about enforcement only
        cfg = self.cfg(gates={"require_g0": True, "parity_pairs": 3, "parity_min_identical": 0},
                       es={"generations": 2})
        d = self.dir / "g0"
        run = ESRun(cfg, d, clock=FakeClock(), log=lambda s: None)
        run.prepare()
        with self.assertRaises(RuntimeError) as ctx:
            run.run()
        self.assertIn("parity.json missing", str(ctx.exception))
        aux = ESRun(cfg, d, log=lambda s: None)
        aux.prepare(pin_config=False)
        self.assertTrue(aux.parity()["passed"])
        self.assertTrue(aux.coverage(n_prompts=4)["passed"])
        with self.assertRaises(RuntimeError):   # records made under other backend numerics do not count
            other = ESRun(deep_merge(cfg, {"essa": {"write_dtype": "bfloat16"}}), self.dir / "g0b", clock=FakeClock(),
                          log=lambda s: None)
            for name in ("parity.json", "coverage.json"):
                (self.dir / "g0b").mkdir(exist_ok=True)
                (self.dir / "g0b" / name).write_bytes((d / name).read_bytes())
            other.prepare()
            other.run()
        run = ESRun(cfg, d, clock=FakeClock(), log=lambda s: None)
        run.prepare()
        self.assertEqual(run.run()["stop_reason"], "generations")


if __name__ == "__main__":
    unittest.main()
