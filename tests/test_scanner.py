"""Tests for scanner pipeline logic."""

import pytest
from data.scanner import evaluate_candidate_rotation
from data.deduplicator import deduplicate_articles, compute_recurring_themes, article_hash


class TestDeduplicator:
    def test_deduplicate_known_article(self):
        h = article_hash("Test headline", "Reuters")
        articles = [{"headline": "Test headline", "source": "Reuters", "article_hash": h}]
        novel, new_hashes = deduplicate_articles(articles, known_hashes={h})
        assert len(novel) == 0
        assert len(new_hashes) == 0

    def test_deduplicate_novel_article(self):
        h = article_hash("New story", "Bloomberg")
        articles = [{"headline": "New story", "source": "Bloomberg", "article_hash": h}]
        novel, new_hashes = deduplicate_articles(articles, known_hashes=set())
        assert len(novel) == 1
        assert len(new_hashes) == 1

    def test_no_duplicates_within_run(self):
        h = article_hash("Same article", "Reuters")
        articles = [
            {"headline": "Same article", "source": "Reuters", "article_hash": h},
            {"headline": "Same article", "source": "Reuters", "article_hash": h},
        ]
        novel, _ = deduplicate_articles(articles, known_hashes=set())
        assert len(novel) == 1  # deduped within run

    def test_recurring_themes(self):
        articles = [
            {"headline": "Tariff fears hit markets"},
            {"headline": "Tariffs could escalate further"},
            {"headline": "Tariff impact on tech sector"},
        ]
        themes = compute_recurring_themes(articles, min_mentions=3)
        theme_words = [t["theme"] for t in themes]
        assert any("tariff" in w for w in theme_words)


class TestCandidateRotation:
    _rotation_tiers = [
        {"max_tenure_days": 3, "min_score_diff": 12},
        {"max_tenure_days": 10, "min_score_diff": 8},
        {"max_tenure_days": 30, "min_score_diff": 5},
        {"max_tenure_days": 99999, "min_score_diff": 3},
    ]

    def _make_candidate(self, symbol, score, entry_date="2026-02-01", slot=1, consec_low=0):
        return {
            "symbol": symbol, "score": score, "entry_date": entry_date,
            "slot": slot, "consecutive_low_runs": consec_low
        }

    def test_no_rotation_below_threshold(self):
        active = [
            self._make_candidate("A", 70, slot=1),
            self._make_candidate("B", 68, slot=2),
            self._make_candidate("C", 65, slot=3),
        ]
        # Challenger only 3 points above weakest (65) — below 5pt threshold for 30+ day tenure
        scanner = {"D": {"score": 68, "path": "momentum"}}
        new_active, rotations = evaluate_candidate_rotation(
            scanner_scores=scanner, active_candidates=active,
            rotation_tiers=self._rotation_tiers,
            weak_pick_score_threshold=60, weak_pick_consecutive_runs=5,
            emergency_rotation_new_score=70, run_date="2026-03-05",
        )
        assert len(rotations) == 0
        assert {c["symbol"] for c in new_active} == {"A", "B", "C"}

    def test_rotation_above_threshold(self):
        active = [
            self._make_candidate("A", 70, slot=1),
            self._make_candidate("B", 68, slot=2),
            self._make_candidate("C", 60, entry_date="2026-01-01", slot=3),  # old tenure
        ]
        # Challenger 6 points above weakest (60) — above 5pt threshold
        scanner = {"D": {"score": 66, "path": "momentum"}}
        new_active, rotations = evaluate_candidate_rotation(
            scanner_scores=scanner, active_candidates=active,
            rotation_tiers=self._rotation_tiers,
            weak_pick_score_threshold=60, weak_pick_consecutive_runs=5,
            emergency_rotation_new_score=70, run_date="2026-03-05",
        )
        assert len(rotations) == 1
        assert rotations[0]["out_ticker"] == "C"
        assert rotations[0]["in_ticker"] == "D"

    def test_max_one_rotation_per_run(self):
        active = [
            self._make_candidate("A", 50, entry_date="2026-01-01", slot=1),
            self._make_candidate("B", 48, entry_date="2026-01-01", slot=2),
            self._make_candidate("C", 46, entry_date="2026-01-01", slot=3),
        ]
        scanner = {
            "D": {"score": 75, "path": "momentum"},
            "E": {"score": 74, "path": "momentum"},
        }
        _, rotations = evaluate_candidate_rotation(
            scanner_scores=scanner, active_candidates=active,
            rotation_tiers=self._rotation_tiers,
            weak_pick_score_threshold=60, weak_pick_consecutive_runs=5,
            emergency_rotation_new_score=70, run_date="2026-03-05",
        )
        assert len(rotations) <= 1

    def test_new_entry_requires_large_gap(self):
        # Candidate entered 2 days ago — requires 12pt gap
        active = [
            self._make_candidate("A", 70, slot=1),
            self._make_candidate("B", 68, slot=2),
            self._make_candidate("C", 65, entry_date="2026-03-03", slot=3),  # new entry
        ]
        scanner = {"D": {"score": 72, "path": "momentum"}}  # only 7pt above C
        _, rotations = evaluate_candidate_rotation(
            scanner_scores=scanner, active_candidates=active,
            rotation_tiers=self._rotation_tiers,
            weak_pick_score_threshold=60, weak_pick_consecutive_runs=5,
            emergency_rotation_new_score=70, run_date="2026-03-05",
        )
        assert len(rotations) == 0  # 7pt < 12pt required for new entry

    def test_emergency_rotation_all_weak(self):
        active = [
            self._make_candidate("A", 50, entry_date="2026-03-03", slot=1),
            self._make_candidate("B", 48, entry_date="2026-03-03", slot=2),
            self._make_candidate("C", 46, entry_date="2026-03-03", slot=3),
        ]
        scanner = {"D": {"score": 75, "path": "momentum"}}  # >= emergency threshold
        _, rotations = evaluate_candidate_rotation(
            scanner_scores=scanner, active_candidates=active,
            rotation_tiers=self._rotation_tiers,
            weak_pick_score_threshold=60, weak_pick_consecutive_runs=5,
            emergency_rotation_new_score=70, run_date="2026-03-05",
        )
        assert len(rotations) == 1
        assert rotations[0]["reason"] == "emergency_rotation"
