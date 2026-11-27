"""The Atlantic v9 config parses, covers every hyperparameter the loop reads, and keeps private
material out of the repository (paths only, never the terminal set)."""
import unittest
from pathlib import Path

from es.es_loop import DEFAULT_CONFIG, config_hash, load_config, plan

ES_DIR = Path(__file__).resolve().parents[1]
CONFIG = ES_DIR / "config" / "atlantic-v9.yaml"


def leaves(d, prefix=()):
    for k, v in d.items():
        if isinstance(v, dict) and v and k not in ("limit_mm_per_prompt", "extra_kwargs", "chat_template_kwargs"):
            yield from leaves(v, prefix + (k,))
        else:
            yield prefix + (k,)


class ConfigTests(unittest.TestCase):
    def test_every_default_is_set_explicitly(self):
        import yaml
        raw = yaml.safe_load(CONFIG.read_text())
        missing = []
        for path in leaves(DEFAULT_CONFIG):
            node = raw
            for p in path:
                if not isinstance(node, dict) or p not in node:
                    missing.append(".".join(path))
                    break
                node = node[p]
        self.assertEqual(missing, [])

    def test_values_and_plan(self):
        cfg = load_config(CONFIG)
        self.assertEqual(cfg["es"]["pairs"], 24)
        # values pre-registered in PROTOCOL-atlantic-v9.md
        self.assertEqual(cfg["run"]["seed"], 20261127)
        self.assertEqual(cfg["es"]["optimizer"]["kind"], "sgd")
        self.assertEqual(cfg["es"]["optimizer"]["lr_sigma2_coef"], 1.7)
        self.assertEqual(cfg["es"]["awd"], 0.005)
        self.assertEqual((cfg["fitness"]["pair_bonus"], cfg["fitness"]["retention_penalty"]), (0.5, 1.5))
        self.assertEqual((cfg["budget"]["no_start_after_usd"], cfg["budget"]["max_usd"]), (215, 220))
        self.assertEqual((cfg["gates"]["dev_gate_gen"], cfg["gates"]["pace_gen"]), (40, 40))
        self.assertEqual(cfg["validation"]["retention"], 32)
        self.assertNotEqual(cfg["validation"]["retention_path"], cfg["data"]["retention"]["path"])
        self.assertTrue(cfg["gates"]["require_g0"])
        self.assertTrue(cfg["budget"]["hard_kill"])
        self.assertGreater(cfg["budget"]["heartbeat_seconds"], 0)
        self.assertEqual(cfg["backend"]["engine"]["max_lora_rank"], 16)
        self.assertTrue(cfg["backend"]["engine"]["enable_prefix_caching"])
        self.assertEqual(cfg["backend"]["engine"]["limit_mm_per_prompt"], {"image": 0, "video": 0})
        p = plan(cfg, [12])
        self.assertEqual((p["population"], p["batch"], p["replies_per_gen"]), (49, 128, 6272))
        self.assertTrue(0.10 <= p["retention_share"] <= 0.20)

    def test_resume_mutable_sections(self):
        cfg = load_config(CONFIG)
        h = config_hash(cfg)
        self.assertEqual(h, config_hash(load_config(CONFIG, ["budget.max_usd=999", "backend.engine.max_loras=2",
                                                             "fitness.grader_workers=2"])))
        self.assertNotEqual(h, config_hash(load_config(CONFIG, ["es.sigma=0.07"])))
        # memory and scheduling may change on a new pod; numerics may not (they change the fitness function)
        self.assertEqual(h, config_hash(load_config(CONFIG, ["backend.engine.max_num_seqs=64",
                                                             "backend.engine.gpu_memory_utilization=0.85",
                                                             "backend.engine.max_cpu_loras=16"])))
        for change in ("backend.engine.fp8_base=true", "backend.engine.kv_cache_dtype=fp8",
                       "backend.engine.dtype=float16", "backend.engine.lora_dtype=float32",
                       "backend.engine.max_model_len=2048", "backend.engine.extra_kwargs={quantization: fp8}"):
            self.assertNotEqual(h, config_hash(load_config(CONFIG, [change])), change)

    def test_no_private_material_or_terminal_set(self):
        text = CONFIG.read_text()
        self.assertNotIn("terminal.jsonl", text)
        for p in ES_DIR.rglob("*.jsonl"):
            self.fail(f"item file inside the public repository: {p}")

    def test_no_em_dashes_in_es_sources(self):
        for p in list(ES_DIR.rglob("*.py")) + list(ES_DIR.rglob("*.md")) + list(ES_DIR.rglob("*.yaml")):
            self.assertNotIn(chr(0x2014), p.read_text(encoding="utf-8"), str(p))


if __name__ == "__main__":
    unittest.main()
