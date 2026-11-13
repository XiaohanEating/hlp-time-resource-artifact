"""Read archived results and derive exact, finite-domain timing quantities.

No quantum packages are imported. No samples or hardware parameters are fitted.
All timing comparisons use T/t_SE = R + rho*L, with equal boundary-stage times.
"""
from __future__ import annotations

import csv
from dataclasses import asdict, dataclass
from fractions import Fraction
from hashlib import sha256
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
OUTPUT = ROOT / "output"


@dataclass(frozen=True, order=True)
class Design:
    ports: int
    buses: int
    qubits: int
    rounds: int
    layers: int

    @property
    def label(self):
        return f"{self.ports}p/{self.buses}b"


def ceildiv(a: int, b: int) -> int:
    return (a + b - 1) // b


def cost(ports: int, buses: int, cap: int) -> Design:
    """Reproduce archived frontier.cost for n=16,d0=5, including full waves."""
    widths = [min(buses, 16 - i) for i in range(0, 16, buses)]
    batches = [min(ports, 16 - i) for i in range(0, 16, ports)]
    rounds = 4 * len(widths) * (12 + 5 * (len(batches) - 1))
    layers = 4 * sum(ceildiv(25 * w * k, cap) for w in widths for k in batches)
    return Design(ports, buses, 16 * 16 * 49 + buses * (50 * ports - 1), rounds, layers)


def from_record(row) -> Design:
    return Design(row["ports"], row["buses"], row["qubits"],
                  row["se_rounds"], row["cross_gate_layers"])


def nondominated(designs):
    """Retain all ties in R,L, as the archived frontier does."""
    return sorted(d for d in designs if not any(
        x.rounds <= d.rounds and x.layers <= d.layers
        and (x.rounds, x.layers) != (d.rounds, d.layers) for x in designs))


def winners(designs, rho: Fraction):
    values = [(Fraction(d.rounds) + rho * d.layers, d) for d in designs]
    best = min(v for v, _ in values)
    return tuple(sorted(d for v, d in values if v == best))


def envelope(designs):
    """Exact open intervals and positive tie points for all rho > 0.

    Fractions prevent numerical tie-breaking. Designs with identical timing
    remain explicitly tied even if their qubit use differs within the cap.
    """
    candidates = set()
    for a in designs:
        for b in designs:
            if a.layers == b.layers:
                continue
            cross = Fraction(b.rounds - a.rounds, a.layers - b.layers)
            if cross > 0:
                candidates.add(cross)
    cuts = [Fraction(0)] + sorted(candidates)
    segments = []
    for i, lo in enumerate(cuts):
        hi = cuts[i + 1] if i + 1 < len(cuts) else None
        sample = (lo + hi) / 2 if hi is not None else lo + 1
        selected = winners(designs, sample)
        if segments and segments[-1]["winners"] == selected:
            segments[-1]["hi"] = hi
        else:
            segments.append({"lo": lo, "hi": hi, "winners": selected})
    ties = []
    for rho in sorted(candidates):
        selected = winners(designs, rho)
        if len({(d.rounds, d.layers) for d in selected}) > 1:
            ties.append({"rho": rho, "winners": selected})
    return segments, ties


def load_json(path):
    return json.loads(path.read_text(encoding="utf-8-sig"))


def signature(designs):
    return tuple((d.ports, d.buses) for d in designs)


class Evidence:
    def __init__(self):
        self.paths = {
            "frontier": ROOT / "results/concurrency_validation/frontier.json",
            "allocation": ROOT / "results/capacity_validation/allocation.json",
            "schedules": ROOT / "results/capacity_validation/schedules.json",
            "resources": ROOT / "results/hardware_audit/resources.json",
        }
        self.hashes_before = {name: sha256(p.read_bytes()).hexdigest()
                              for name, p in self.paths.items()}
        self.raw = {name: load_json(p) for name, p in self.paths.items()}
        self.frontier = self.raw["frontier"]
        self.caps = self.frontier["caps"]
        self.budget = self.frontier["total_qubit_budget"]
        self.domains = [(q, b) for q in self.frontier["port_domain"]
                        for b in self.frontier["bus_domain"]
                        if cost(q, b, 25).qubits <= self.budget]
        self.designs = {c: [cost(q, b, c) for q, b in self.domains] for c in self.caps}
        self.pareto = {row["cap"]: sorted(from_record(x) for x in row["pareto"])
                       for row in self.frontier["optimal_uniform"]}
        self.hulls = {c: envelope(self.pareto[c]) for c in self.caps}
        self.resources = {r["ports"]: r for r in self.raw["resources"]["records"]}
        self.checks = self.validate()

    def validate(self):
        assert len(self.domains) == 160 and len(self.caps) == 256
        for c in self.caps:
            assert nondominated(self.designs[c]) == self.pareto[c], c
        for p in self.frontier["points"]:
            d = cost(p["ports"], 16, p["cap"])
            assert (d.rounds, d.layers) == (p["se_rounds"], p["cross_gate_layers"])
        certificates = []
        for row in self.frontier["summary"]:
            q, c = row["ports"], row["cap"]
            path = ROOT / f"results/concurrency_validation/n16_q{q}_cap{c}_certificate.json"
            proof = load_json(path)
            total = 0
            for batch in proof["batches"]:
                sizes = [len(s) * 25 for s in batch["sublayers"]]
                assert max(sizes) <= c
                assert len(sizes) == batch["lower_bound"] == ceildiv(sum(sizes), c)
                total += len(sizes)
            assert total == cost(q, 16, c).layers
            certificates.append({"path": str(path.relative_to(ROOT)),
                                 "sha256": sha256(path.read_bytes()).hexdigest()})
        for record in self.raw["schedules"]["records"]:
            if record["n"] == 16:
                assert cost(record["ports"], record["buses"], 1600).rounds == record["conditional_rounds"]
        anchors = next(r for r in self.raw["allocation"]["results"] if r["n"] == 16)["anchors"]
        for anchor in anchors:
            candidates = [d.rounds for d in self.designs[1600] if d.qubits <= anchor["total_qubit_budget"]]
            assert min(candidates) == anchor["uniform_cycle_rounds"]
        assert cost(4, 16, 1600).layers == cost(8, 16, 1600).layers == 16
        assert cost(6, 16, 2400).layers == 12
        assert signature(winners(self.pareto[2400], Fraction(5))) == ((6, 16), (8, 16))
        for q, r in self.resources.items():
            assert cost(q, 16, 1600).qubits == r["qubits"]
        return {"candidate_cap_pairs_recomputed": len(self.domains) * len(self.caps),
                "pareto_sets_matched": len(self.caps), "saved_points_matched": len(self.frontier["points"]),
                "certificates_checked": certificates, "allocation_anchors_matched": len(anchors),
                "new_quantum_samples": 0, "decoder_runs": 0}

    def export(self):
        dest = OUTPUT / "data"
        dest.mkdir(parents=True, exist_ok=True)
        with (dest / "all_uniform_candidates.csv").open("w", encoding="utf-8", newline="") as f:
            out = csv.DictWriter(f, fieldnames=["cap", "ports", "buses", "qubits", "rounds", "layers"])
            out.writeheader()
            for c, designs in self.designs.items():
                out.writerows({"cap": c, **asdict(d)} for d in designs)
        envelopes = []
        for c, (segments, ties) in self.hulls.items():
            envelopes.append({"cap": c,
                "segments": [{"rho_lo": str(s["lo"]), "rho_hi": str(s["hi"]) if s["hi"] is not None else "infinity",
                              "winners": [asdict(d) for d in s["winners"]]} for s in segments],
                "tie_points": [{"rho": str(t["rho"]), "winners": [asdict(d) for d in t["winners"]]} for t in ties]})
        write_json(dest / "exact_timing_envelopes.json", {"equation": "T/t_SE = R + rho L",
                  "scope": "rho>0; equal nominal phase durations; no hardware calibration",
                  "envelopes": envelopes})
        write_json(dest / "resources.json", self.raw["resources"])

    def assert_sources_unchanged(self):
        assert self.hashes_before == {name: sha256(p.read_bytes()).hexdigest()
                                      for name, p in self.paths.items()}


def write_json(path: Path, data):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
