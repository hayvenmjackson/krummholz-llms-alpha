# Safety gate (C++)

One gate for every model: Atlantic Lite and Atlantic 27B sit behind the same code. `safety_gate.hpp` (header only,
C++17) joins layer 1 (the fixed rules in `../safety_screen`) and layer 2 (a judge that reads the request and the
finished reply). A reply is shown only if both pass it; otherwise the user gets a fixed refusal written in code.
`make test` runs the fixed cases.

- `before(request)`: rules on the request. A stopped request never reaches the model.
- `after(request, reply)`: rules on the reply, then the judge. Harmful: fixed refusal. Judge missing or failing: no reply.
- The gate will not start without a rules file and a judge.

## What is and is not measured (November 14, 2026)

- Layer 1 alone stops about a third of harmful requests it has not seen (HarmBench held-out half 54 of 159); see
  `../safety_screen/README.md`. It is not a guarantee that every harmful request is refused.
- Weights alone, HarmBench (300 behaviors): Atlantic 27B v7 1 harmful, Atlantic Lite v8 23 harmful.
- Layer 2 is the part that closes the gap. The judge is not wired to a model yet: on a server or desktop it can be the
  HarmBench classifier (MIT, 7B) through llama.cpp; a phone needs a smaller one, not chosen yet.
- A gate judged by the HarmBench classifier scores zero on HarmBench by construction, so that number proves nothing.
  It has to be measured on a set the classifier was not built for, with a second, different judge.
