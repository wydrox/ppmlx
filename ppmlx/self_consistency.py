# ppmlx/self_consistency.py — multi-run extraction with majority voting
"""
Self-Consistency: run slot extraction 3 times with variation, keep only
candidates that appear in ≥2 runs. Eliminates hallucinations from small models.
"""

from __future__ import annotations
from dataclasses import dataclass
from typing import Callable
import re

from ppmlx.slot_extractor import SlotExtractor, ExtractedCandidate


@dataclass
class ConsensusCandidate:
    """A candidate that survived self-consistency voting."""
    candidate: ExtractedCandidate
    num_runs: int            # 2 or 3
    cluster_size: int        # how many raw candidates clustered together
    consensus_confidence: float  # adjusted confidence based on agreement


class SelfConsistencyExtractor:
    """
    Run extraction 3 times with variation, keep majority-voted candidates.

    Variation comes from:
    - Different temperature (0.0, 0.3, 0.5)
    - Different few-shot examples (none, coding, infra — placeholder for now)

    The core insight: real facts are stable across runs; hallucinations vary.
    A hallucination in one run is very unlikely to appear in 2 others.
    """

    def __init__(
        self,
        model_name: str = "gemma-4-e2b",
        generation_fn: Callable | None = None,
        num_runs: int = 3,
        agreement_threshold: int = 2,
        # Variation parameters
        temperatures: tuple[float, ...] = (0.0, 0.3, 0.5),
        embedding_fn: Callable | None = None,  # for fuzzy matching across runs
    ):
        self.model_name = model_name
        self.generation_fn = generation_fn
        self.num_runs = num_runs
        self.agreement_threshold = agreement_threshold
        self.temperatures = temperatures[:num_runs]
        self._embed_fn = embedding_fn

    def extract(
        self,
        segment_text: str,
        fact_types: list[str],
    ) -> list[ConsensusCandidate]:
        """
        Run extraction multiple times and return consensus candidates.
        """
        all_runs: list[list[ExtractedCandidate]] = []

        for run_idx in range(self.num_runs):
            temp = self.temperatures[min(run_idx, len(self.temperatures) - 1)]

            extractor = SlotExtractor(
                model_name=self.model_name,
                generation_fn=self.generation_fn,
                temperature=temp,
            )

            candidates = extractor.extract(segment_text, fact_types)
            all_runs.append(candidates)

        if not all_runs or not any(all_runs):
            return []

        # Cluster candidates across runs
        clusters = self._cluster_candidates(all_runs)

        # Vote: keep clusters with ≥ agreement_threshold members
        consensus: list[ConsensusCandidate] = []
        for cluster in clusters:
            if len(cluster) >= self.agreement_threshold:
                # Select medoid (most central) candidate from cluster
                medoid = self._select_medoid(cluster)
                if medoid is None:
                    continue

                # Adjust confidence: base × agreement_bonus
                agreement_bonus = min(1.0, len(cluster) / self.num_runs)
                adjusted_conf = round(medoid.confidence * (0.7 + 0.3 * agreement_bonus), 4)

                consensus.append(ConsensusCandidate(
                    candidate=medoid,
                    num_runs=len(cluster),
                    cluster_size=len(cluster),
                    consensus_confidence=adjusted_conf,
                ))

        return consensus

    def _cluster_candidates(
        self,
        all_runs: list[list[ExtractedCandidate]],
    ) -> list[list[ExtractedCandidate]]:
        """
        Cluster candidates across runs by fuzzy type+subject+predicate+object match.

        Two candidates match if:
        - Same type
        - Subject tokens Jaccard ≥ 0.6
        - Predicate tokens Jaccard ≥ 0.6
        - Object tokens Jaccard ≥ 0.5 (objects can vary more in wording)
        """
        # Flatten with run metadata
        flat: list[tuple[int, ExtractedCandidate]] = []
        for run_idx, candidates in enumerate(all_runs):
            for c in candidates:
                flat.append((run_idx, c))

        if not flat:
            return []

        # Greedy clustering: pick seed, find all within threshold, remove, repeat
        remaining = list(flat)
        clusters: list[list[ExtractedCandidate]] = []

        while remaining:
            seed_run, seed = remaining.pop(0)
            cluster = [seed]

            i = 0
            while i < len(remaining):
                _, other = remaining[i]
                if self._candidates_match(seed, other):
                    cluster.append(other)
                    remaining.pop(i)
                else:
                    i += 1

            clusters.append(cluster)

        return clusters

    @staticmethod
    def _candidates_match(a: ExtractedCandidate, b: ExtractedCandidate) -> bool:
        """Return True if two candidates likely describe the same fact."""
        if a.type != b.type:
            return False

        subj_score = _token_jaccard(a.subject, b.subject)
        pred_score = _token_jaccard(a.predicate, b.predicate)
        obj_score = _token_jaccard(a.object, b.object)

        return subj_score >= 0.6 and pred_score >= 0.6 and obj_score >= 0.5

    @staticmethod
    def _select_medoid(cluster: list[ExtractedCandidate]) -> ExtractedCandidate | None:
        """
        Select the most central candidate from a cluster.
        Centrality = average pairwise Jaccard similarity to other cluster members.
        """
        if len(cluster) == 1:
            return cluster[0]
        if not cluster:
            return None

        best_idx = 0
        best_score = -1.0

        for i, ci in enumerate(cluster):
            total = 0.0
            for j, cj in enumerate(cluster):
                if i == j:
                    continue
                total += (
                    _token_jaccard(ci.subject, cj.subject) +
                    _token_jaccard(ci.predicate, cj.predicate) +
                    _token_jaccard(ci.object, cj.object)
                ) / 3.0
            avg = total / (len(cluster) - 1)
            if avg > best_score:
                best_score = avg
                best_idx = i

        return cluster[best_idx]


def _token_jaccard(a: str, b: str) -> float:
    """Jaccard similarity between token sets of two strings."""
    tokens_a = set(re.findall(r"\b\w+\b", a.lower()))
    tokens_b = set(re.findall(r"\b\w+\b", b.lower()))
    if not tokens_a or not tokens_b:
        return 1.0 if a.lower() == b.lower() else 0.0
    return len(tokens_a & tokens_b) / len(tokens_a | tokens_b)
