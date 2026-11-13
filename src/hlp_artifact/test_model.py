"""Independent checks of exact timing choices and scientific input alignment."""
from fractions import Fraction

import pytest

from .model import Design, Evidence, cost, envelope, signature, winners


@pytest.fixture(scope="module")
def evidence():
    return Evidence()


def test_exact_crossover_with_tied_implementations():
    designs = [Design(8, 16, 18928, 68, 16),
               Design(6, 16, 17328, 88, 12),
               Design(7, 16, 18128, 88, 12)]
    segments, ties = envelope(designs)
    assert [(x["lo"], x["hi"]) for x in segments] == [(0, 5), (5, None)]
    assert signature(segments[0]["winners"]) == ((8, 16),)
    assert signature(segments[1]["winners"]) == ((6, 16), (7, 16))
    assert len(ties) == 1 and ties[0]["rho"] == Fraction(5)
    assert len(ties[0]["winners"]) == 3


def test_fractional_crossing_is_not_rounded():
    designs = [Design(4, 16, 100, 1, 5), Design(8, 16, 100, 2, 2)]
    segments, ties = envelope(designs)
    assert segments[0]["hi"] == Fraction(1, 3)
    assert ties[0]["rho"] == Fraction(1, 3)
    assert len(winners(designs, Fraction(1, 3))) == 2


def test_nonoptimal_pair_crossing_does_not_create_region():
    designs = [Design(4, 1, 10, 1, 1), Design(6, 1, 20, 9, 2),
               Design(8, 1, 30, 4, 5)]
    segments, ties = envelope(designs)
    assert len(segments) == 1 and segments[0]["hi"] is None
    assert not ties
    assert signature(segments[0]["winners"]) == ((4, 1),)


def test_interval_and_boundary_winners_match_all_candidates(evidence):
    for cap, (segments, ties) in evidence.hulls.items():
        full = evidence.designs[cap]
        for s in segments:
            sample = (s["lo"] + s["hi"]) / 2 if s["hi"] is not None else s["lo"] + 100
            assert winners(full, sample) == s["winners"]
        for t in ties:
            assert winners(full, t["rho"]) == t["winners"]


def test_saturation_example_and_timing_boundaries():
    assert cost(4, 16, 1600).rounds == cost(4, 32, 1600).rounds == 108
    assert cost(4, 32, 1600).qubits == 18912
    for rho in (Fraction(0), Fraction(1, 10), Fraction(5), Fraction(50)):
        a, b = cost(4, 16, 1600), cost(8, 16, 1600)
        assert (a.rounds + rho * a.layers) - (b.rounds + rho * b.layers) == 40
        a, b = cost(6, 16, 2400), cost(8, 16, 2400)
        assert (a.rounds + rho * a.layers) - (b.rounds + rho * b.layers) == 20 - 4 * rho


def test_display_range_contains_every_positive_transition(evidence):
    crossings = {t["rho"] for _, ties in evidence.hulls.values() for t in ties}
    assert crossings == {Fraction(5), Fraction(10), Fraction(17)}
    for cap in evidence.caps:
        assert signature(winners(evidence.designs[cap], Fraction(1, 1000))) == ((8, 16),)
        assert winners(evidence.designs[cap], Fraction(64)) == winners(evidence.designs[cap], Fraction(10**6))


def test_raw_evidence_is_immutable(evidence):
    evidence.assert_sources_unchanged()
