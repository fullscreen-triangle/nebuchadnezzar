"""Ladders on the circuit: rung power as the fraction of the remaining drop closed.

For a series chain reservoir -> r_1 -> s_1 -> ... -> r_n -> s_n -> medium, in the
linear tangent J = G A, with resistances R_k = 1/G_k and the medium's exchange
resistance R_M:

  drop at rung k        Delta_k = J R_k
  rung power            pi_k = R_k / (R_k + ... + R_n + R_M)
  composite power       1 - prod (1 - pi_k) = 1 - R_M / R_total   (telescopes)
  floor                 R_M / R_total  -- the part no rung closes
  share of rung k       R_k / R_total = d ln J / d ln G_k  (flux control coefficient)

So the ladder's rung powers are derived from the circuit, not declared, once
conductances exist; without them they are ABSENT.
"""
from __future__ import annotations

import math

from .solve import Network, linear_flux, relax_linear


def rung_powers(R: list[float], R_M: float) -> list[float]:
    out, tail = [], R_M + sum(R)
    for r in R:
        out.append(r / tail)
        tail -= r
    return out


def composite(pis: list[float]) -> float:
    return 1.0 - math.prod(1.0 - p for p in pis)


def shares(R: list[float], R_M: float) -> list[float]:
    tot = sum(R) + R_M
    return [r / tot for r in R]


def chain_network(G: list[float], G_M: float, mu_in: float, mu_out: float) -> Network:
    species = {"s0": {"fixed": mu_in}}
    junctions = {}
    for k, g in enumerate(G, start=1):
        species[f"s{k}"] = {"fixed": None}
        junctions[f"r{k}"] = {"nu": {f"s{k-1}": -1.0, f"s{k}": 1.0}, "G": g}
    species[f"s{len(G)}"].update({"G_ex": G_M, "mu_res": mu_out})
    return Network(species, junctions)


def chain_flux(G, G_M, mu_in=10.0, mu_out=0.0) -> float:
    net = chain_network(G, G_M, mu_in, mu_out)
    mu, _ = relax_linear(net)
    return linear_flux(net, mu, "r1")
