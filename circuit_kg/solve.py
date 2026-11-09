"""Node solves by local relaxation: each node solves its own law from its contacts.

A Network is the parameterised form of a subgraph of the circuit KG:
  species: name -> {"fixed": mu or ln c, or None}, plus optional exchange
           {"G_ex": g, "mu_res": m} (linear mode only)
  junctions: name -> {"nu": {species: coef}, "G": g}            (linear mode)
                   or {"nu": {...}, "k": k, "K": K}             (mass action)

Linear mode   J_r = G_r A_r,  A_r = -sum_s nu_sr mu_s
Mass action   J_r = k_r prod_reac c^|nu| - (k_r/K_r) prod_prod c^nu,  mu = RT ln c

The relaxation is Gauss-Seidel over nodes; a node update uses only the
junctions it touches and the states of their other participants.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field

from .core import RT


@dataclass
class Network:
    species: dict
    junctions: dict
    incident: dict = field(default_factory=dict)

    def __post_init__(self):
        self.incident = {s: [] for s in self.species}
        for r, j in self.junctions.items():
            for s in j["nu"]:
                self.incident[s].append(r)

    def free(self):
        return [s for s, d in self.species.items() if d.get("fixed") is None]


# ------------------------------------------------------------------ linear

def linear_flux(net: Network, mu: dict, r: str) -> float:
    j = net.junctions[r]
    return j["G"] * -sum(nu * mu[s] for s, nu in j["nu"].items())


def relax_linear(net: Network, mu0: dict | None = None, tol=1e-13, max_sweeps=100000):
    mu = {s: (d["fixed"] if d.get("fixed") is not None else 0.0) for s, d in net.species.items()}
    if mu0:
        mu.update(mu0)
    for sweep in range(max_sweeps):
        delta = 0.0
        for s in net.free():
            d = net.species[s]
            num = d.get("G_ex", 0.0) * d.get("mu_res", 0.0)
            den = d.get("G_ex", 0.0)
            for r in net.incident[s]:
                j = net.junctions[r]
                nu_s = j["nu"][s]
                rest = sum(nu * mu[t] for t, nu in j["nu"].items() if t != s)
                num += nu_s * j["G"] * -rest
                den += nu_s * nu_s * j["G"]
            new = num / den
            delta = max(delta, abs(new - mu[s]))
            mu[s] = new
        if delta < tol:
            return mu, sweep + 1
    raise RuntimeError("relaxation did not converge")


# ------------------------------------------------------------------ mass action

def mass_action_flux(net: Network, x: dict, r: str) -> float:
    j = net.junctions[r]
    fwd = sum(-nu * x[s] for s, nu in j["nu"].items() if nu < 0)
    bwd = sum(nu * x[s] for s, nu in j["nu"].items() if nu > 0)
    return j["k"] * (math.exp(fwd) - math.exp(bwd) / j["K"])


def _node_residual(net, x, s):
    return sum(net.junctions[r]["nu"][s] * mass_action_flux(net, x, r) for r in net.incident[s])


def relax_mass_action(net: Network, x0: dict | None = None, tol=1e-12, max_sweeps=20000):
    """x = ln c.  Each free node solves its residual = 0 by bracketed bisection on x_s."""
    x = {s: (d["fixed"] if d.get("fixed") is not None else 0.0) for s, d in net.species.items()}
    if x0:
        x.update(x0)
    for sweep in range(max_sweeps):
        delta = 0.0
        for s in net.free():
            lo, hi = x[s] - 1.0, x[s] + 1.0
            f = lambda v: _node_residual(net, {**x, s: v}, s)
            # residual decreases in x_s (consumption grows, production falls)
            while f(lo) < 0:
                lo -= 2 * (hi - lo)
            while f(hi) > 0:
                hi += 2 * (hi - lo)
            for _ in range(200):
                mid = 0.5 * (lo + hi)
                if f(mid) > 0:
                    lo = mid
                else:
                    hi = mid
                if hi - lo < tol:
                    break
            new = 0.5 * (lo + hi)
            delta = max(delta, abs(new - x[s]))
            x[s] = new
        if delta < tol * 10:
            return x, sweep + 1
    raise RuntimeError("relaxation did not converge")
