"""Dynamic weights (ADR-004): layered, signed, reversible, user-approved learning.

Layers: L1 Krummholz core (merged base), L2 product adapter, L3 organization adapter,
L4 personal adapter. Facts live in the memory store, never in weights. This package is
host-side bookkeeping and policy; it does not train or run models.
"""
