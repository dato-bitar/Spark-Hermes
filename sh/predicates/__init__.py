"""Spark-Hermes v2 predicates: a closed vocabulary of facts about a workspace, evaluated in Python.

from sh.predicates import evaluate, judge, parse_all
judge([["file_exists", "out/report.txt"], ["digest_is", "out/report.txt", "<sha256>"]], "/workspace")
"""

from .evaluate import Verdict, evaluate, judge
from .schema import OPS, STRUCTURAL, Predicate, PredicateError, parse, parse_all

__all__ = ["OPS", "STRUCTURAL", "Predicate", "PredicateError", "Verdict", "evaluate", "judge", "parse", "parse_all"]
