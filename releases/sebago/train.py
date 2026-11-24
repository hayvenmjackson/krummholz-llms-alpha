"""Reproduce Sebago, a 40,530-parameter symbolic field-record reader."""
import hashlib
import itertools
import json
import random
from pathlib import Path

import torch
import struct
from torch import nn
from torch.nn import functional as F

OUT = Path(__file__).resolve().parent
KEYS = list(range(6))
DIGITS = range(6, 14)
ANSWER, ABSENT, END, PAD = range(14, 18)
WIDTH = 48
STEPS = 1500
BATCH = 48
PREFIX = 14  # query, three 4-token records, answer delimiter


class Reader(nn.Module):
    def __init__(self):
        super().__init__()
        self.tokens = nn.Embedding(18, WIDTH)
        self.positions = nn.Embedding(PREFIX + 4, WIDTH)
        self.layers = nn.ModuleList([
            nn.TransformerEncoderLayer(WIDTH, 4, 2 * WIDTH, dropout=0.0, batch_first=True)
            for _ in range(2)
        ])
        self.output = nn.Linear(WIDTH, 18)
        self.register_buffer("mask", torch.ones(PREFIX + 4, PREFIX + 4, dtype=torch.bool).triu(1))

    def forward(self, tokens):
        length = tokens.shape[1]
        hidden = self.tokens(tokens) + self.positions(torch.arange(length))
        for layer in self.layers:
            hidden = layer(hidden, src_mask=self.mask[:length, :length], is_causal=True)
        return self.output(hidden)


def save_weights(net, path):
    header = {"__metadata__": {"name": "Sebago", "format": "pt", "vocabulary": "18 symbols"}}
    buffer = bytearray()
    for name, tensor in net.named_parameters():
        raw = tensor.detach().cpu().contiguous().numpy().astype("<f4", copy=False).tobytes()
        start = len(buffer)
        buffer.extend(raw)
        header[name] = {"dtype": "F32", "shape": list(tensor.shape), "data_offsets": [start, len(buffer)]}
    encoded = json.dumps(header, separators=(",", ":")).encode()
    encoded += b" " * (-len(encoded) % 8)
    path.write_bytes(struct.pack("<Q", len(encoded)) + encoded + buffer)


def make_prefix(query, keys, values):
    return [query] + [tok for key, value in zip(keys, values) for tok in (key, *value)] + [ANSWER]


def pair(rng, pool):
    keys = rng.sample(KEYS, 4)
    query = keys[0]
    values = rng.sample(pool, 3)
    order = rng.sample(range(3), 3)
    def inputs(present):
        fields = [query if present else keys[3], *keys[1:3]]
        return make_prefix(query, [fields[i] for i in order], [values[i] for i in order])
    return inputs(True), inputs(False), [*values[0], END], [ABSENT, END]


@torch.inference_mode()
def generate(net, prefixes):
    tokens = torch.tensor(prefixes)
    for _ in range(4):
        new = net(tokens)[:, -1].argmax(-1)
        tokens = torch.cat([tokens, new[:, None]], dim=1)
    out = []
    for row in tokens[:, PREFIX:].tolist():
        out.append(row[:row.index(END) + 1] if END in row else row)
    return out


def score(net, cases):
    records = []
    for i in range(0, len(cases), 32):
        block = cases[i:i+32]
        guesses = generate(net, [pref for p, m, *_ in block for pref in (p, m)])
        for j, (p, m, a, b) in enumerate(block):
            records.append({"pair": i+j, "present_prompt": p, "missing_prompt": m, "value": a[:-1],
                            "present_completion": guesses[2*j], "missing_completion": guesses[2*j+1],
                            "present": guesses[2*j] == a, "absent": guesses[2*j+1] == b})
    return {"pairs": len(records), "correct_present": sum(r["present"] for r in records),
            "correct_absent": sum(r["absent"] for r in records),
            "both": sum(r["present"] and r["absent"] for r in records)}, records


def main():
    torch.set_num_threads(2)
    torch.manual_seed(271124)
    rng = random.Random(271124)
    values = list(itertools.product(DIGITS, repeat=3))
    random.Random(42).shuffle(values)
    test_pool, train_pool = values[:96], values[96:]
    net = Reader()
    save_weights(net, OUT / "sebago-init.safetensors")
    opt = torch.optim.AdamW(net.parameters(), lr=0.0015)
    test_rng = random.Random(271125)
    cases = [pair(test_rng, test_pool) for _ in range(512)]
    baseline, _ = score(net.eval(), cases)
    net.train()
    for step in range(1, STEPS + 1):
        seqs, targets = [], []
        for j in range(BATCH):
            p, m, a, b = pair(rng, train_pool)
            pref, completion = (p, a) if j % 2 else (m, b)
            padded = completion + [PAD] * (4-len(completion))
            seqs.append(pref + padded[:-1])
            targets.append(padded)
        logits = net(torch.tensor(seqs))[:, PREFIX-1:]
        loss = F.cross_entropy(logits.reshape(-1, 18), torch.tensor(targets).flatten(), ignore_index=PAD)
        opt.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(net.parameters(), 1.0)
        opt.step()
        if step % 250 == 0:
            print(f"step {step}/{STEPS}, training loss {loss.item():.4f}", flush=True)
    scores, records = score(net.eval(), cases)
    model_path = OUT / "sebago-40k.safetensors"
    save_weights(net, model_path)
    result = {"weights_sha256": hashlib.sha256(model_path.read_bytes()).hexdigest(),"task": "three-key symbolic records, three-symbol values, greedy decoding over the full 18-token vocabulary",
              "model_parameters": sum(p.numel() for p in net.parameters()), "seed": 271124,
              "steps": STEPS, "examples_per_step": BATCH, "torch_version": str(torch.__version__),
              "training_value_types": len(train_pool), "held_out_value_types": len(test_pool),
              "unique_queried_test_values": len({tuple(r["value"]) for r in records}),
              "train_test_value_type_overlap": len(set(train_pool) & set(test_pool)),
              "untrained": baseline, "trained": scores, "cases": records}
    # Keep each paired prompt and its scored completions on one line for git review.
    summary = {k: v for k, v in result.items() if k != "cases"}
    rows = ",\n".join("    " + json.dumps(case) for case in records)
    report = json.dumps(summary, indent=2)[:-2] + ',\n  "cases": [\n' + rows + "\n  ]\n}\n"
    (OUT / "evaluation.json").write_text(report, encoding="utf8")
    print(json.dumps({k: v for k,v in result.items() if k != "cases"}), flush=True)


if __name__ == "__main__":
    main()
