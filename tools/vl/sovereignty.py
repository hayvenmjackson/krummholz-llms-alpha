"""Deterministic sovereignty checker (pass 4).

The rules moved to krummholz_llm/sovereignty.py, which also enforces them at runtime; this module keeps the import
path the training tools use (teacher answers, on-policy preference pairs and probe scoring).
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from krummholz_llm.sovereignty import (ATTRIBUTION, CLAIMS, CLAUSE, FALSE_FACTS, SENT, SUBJECT, UN_FALSE,  # noqa: E402,F401
                                       check, clauses)
