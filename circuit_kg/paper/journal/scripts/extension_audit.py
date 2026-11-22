"""
Measurements of dcat-ap-plus-homo-sapien reported in the manuscript (writes ``data/extension.json``).

Run inside the extension's environment, after ``full_inputs.py``::

    cd dcat-ap-plus-homo-sapien
    uv run --no-sync python ../circuit_kg/paper/journal/scripts/extension_audit.py --inputs <dir>

Every number in the manuscript's extension section is read from the file this script writes.
The sections are:

* ``kernel``       the extension's own run over the whole corpus: structure, verdicts per question,
                   gate letters, bases, the Hess fit and the per-reaction direction details;
* ``surface``      Delta_r G'm over pH x pMg for the medium-blocked reaction whose sign is most
                   evenly split, at three ionic strengths;
* ``catalogue``    the emitted named graphs: triples per graph, serialisation size, SHACL result
                   and timings;
* ``comparison``   a second run under a narrower medium, joined to the first on verdict IRIs with
                   SPARQL, and the stage digests of both runs;
* ``controls``     the shapes against planted defects in the bundled example's catalogue, each
                   paired with the unaltered catalogue;
* ``scaling``      emission and validation time against corpus size;
* ``schema``       classes and slots of the merged schema by contributing schema, and the
                   ancestry of each class of the extension;
* ``scidats``      the verdict table written to Parquet with ``CircuitDataset`` metadata and read back.
"""

from __future__ import annotations

import argparse
import json
import math
import socket
import tempfile
import time
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import numpy as np
import yaml
from rdflib import Dataset, Graph, Literal, Namespace, URIRef
from rdflib.namespace import RDF, SH

from dcat_ap_plus_homo_sapien import __version__
from dcat_ap_plus_homo_sapien.corpus import Corpus
from dcat_ap_plus_homo_sapien.dcat_ap_plus_homo_sapien_impl import CircuitKernel
from dcat_ap_plus_homo_sapien.emit import Answer, emit, shapes, union
from dcat_ap_plus_homo_sapien.examples import load_example
from dcat_ap_plus_homo_sapien.medium import WATER, Legendre, Medium
from dcat_ap_plus_homo_sapien.thermo import MeasurementTable

HS = Namespace("https://w3id.org/nfdi-de/dcat-ap-plus/homo-sapien/")
OUT = Path(__file__).resolve().parents[1] / "data" / "extension.json"


def clock() -> float:
    """Wall-clock seconds."""
    return time.perf_counter()


def shacl(g: Graph) -> tuple[bool, Graph, float]:
    """Validate a graph against the extension's shapes; return conformance, report and seconds."""
    from pyshacl import validate

    t = clock()
    conforms, report, _ = validate(g, shacl_graph=shapes(), inference="none")
    return bool(conforms), report, clock() - t


def results_of(report: Graph) -> list[dict[str, str]]:
    """The validation results of a pySHACL report: focus node, path and constraint component."""
    out = []
    for r in report.subjects(RDF.type, SH.ValidationResult):
        out.append(
            {
                "focus": str(report.value(r, SH.focusNode)),
                "path": str(report.value(r, SH.resultPath) or ""),
                "component": str(report.value(r, SH.sourceConstraintComponent)).rsplit("#", 1)[-1],
            }
        )
    return out


# --------------------------------------------------------------------------------- kernel run
def kernel_section(k: CircuitKernel) -> tuple[dict[str, Any], list[Answer]]:
    """Structure, verdicts and direction details of one run."""
    t = clock()
    structure = k.structure().to_dict()
    t_structure = clock() - t
    t = clock()
    answers = k.answers()
    t_answers = clock() - t
    by_q: dict[str, Counter[str]] = defaultdict(Counter)
    gates: dict[str, Counter[str]] = defaultdict(Counter)
    bases: dict[str, Counter[str]] = defaultdict(Counter)
    blockers: dict[str, Counter[str]] = defaultdict(Counter)
    conventions: Counter[str] = Counter()
    for a in answers:
        v = a.verdict
        by_q[a.question][v.label.value] += 1
        if v.gate is not None:
            gates[a.question][v.gate.value] += 1
        if v.basis is not None:
            bases[a.question][v.basis.value] += 1
        if v.blocker:
            blockers[a.question][v.blocker] += 1
        conventions.update(v.conventions)

    rt = k.medium.rt
    directions = []
    for rid, v in k.directions().items():
        value = k.chemical_values.get(rid)
        if value is None:
            continue
        st = {c: n for c, n in value.stoichiometry.items() if c != WATER and n}
        at_1mm = rt * math.log(1e-3) * sum(st.values())
        d_lo, d_hi = v.detail.get("dG_prime_range", [None, None])
        directions.append(
            {
                "reaction": rid,
                "name": k.corpus.reactions[rid].name,
                "source": value.source,
                "g0": value.g0,
                "sigma": value.sigma,
                "label": v.label.value,
                "payload": v.payload,
                "blocker": v.blocker,
                "dG_prime_range": [d_lo, d_hi],
                "dG_prime_m_range": [d_lo + at_1mm, d_hi + at_1mm] if d_lo is not None else None,
                "RT_lnQ_range": v.detail.get("RT_lnQ_range"),
                "sign_fixed_at_1mM": v.detail.get("dG_prime_m_sign_fixed"),
            }
        )
    section = {
        "structure": structure,
        "verdicts": {q: dict(c) for q, c in by_q.items()},
        "gates": {q: dict(c) for q, c in gates.items()},
        "bases": {q: dict(c) for q, c in bases.items()},
        "blockers": {q: dict(c.most_common(12)) for q, c in blockers.items()},
        "conventions_used": dict(conventions),
        "hess_fit": k.hess.fit(),
        "directions": directions,
        "answers": len(answers),
        "timing_s": {"structure": t_structure, "answers": t_answers},
    }
    return section, answers


def surface_section(k: CircuitKernel, directions: list[dict[str, Any]]) -> dict[str, Any]:
    """Delta_r G'm over pH x pMg for the medium-blocked reaction whose sign is most evenly split."""
    blocked = [d for d in directions if d["label"] == "undecidable" and d["blocker"] == "medium"]
    m = k.medium
    rt = m.rt
    ph = np.linspace(*m.ph, 29)
    pmg = np.linspace(*m.pmg, 29)

    def grid(rid: str, ionic: float) -> np.ndarray:
        value = k.chemical_values[rid]
        st = {c: n for c, n in value.stoichiometry.items() if c != WATER and n}
        at_1mm = rt * math.log(1e-3) * sum(st.values())
        z = np.empty((len(pmg), len(ph)))
        for i, y in enumerate(pmg):
            for j, x in enumerate(ph):
                t, _ = k.legendre.reaction(value.stoichiometry, float(x), ionic, m.temperature, float(y))
                z[i, j] = value.g0 + float(t) + at_1mm
        return z

    best, score = None, -1.0
    for d in blocked:
        z = grid(d["reaction"], float(np.mean(m.ionic_strength)))
        if z.min() < 0 < z.max():
            s = min(-z.min(), z.max()) / (z.max() - z.min())
            if s > score:
                best, score = d, s
    if best is None:
        return {}
    return {
        "reaction": best["reaction"],
        "name": best["name"],
        "source": best["source"],
        "sigma": best["sigma"],
        "ph": ph.tolist(),
        "pmg": pmg.tolist(),
        "ionic_strength": {str(i): grid(best["reaction"], i).tolist() for i in (m.ionic_strength[0], m.ionic_strength[1])},
        "balance": score,
    }


# ---------------------------------------------------------------------------------- catalogue
def catalogue_section(k: CircuitKernel, answers: list[Answer], workdir: Path) -> tuple[dict[str, Any], Dataset]:
    """Emit, serialise and validate the full catalogue."""
    t = clock()
    ds = emit(k.corpus, k.medium, answers, run="full-a")
    t_emit = clock() - t
    counts = {str(g.identifier).rsplit("/", 1)[-1]: len(g) for g in ds.graphs() if len(g)}
    t = clock()
    path = workdir / "catalogue.trig"
    path.write_text(ds.serialize(format="trig"), encoding="utf-8")
    t_ser = clock() - t
    g = union(ds)
    conforms, report, t_val = shacl(g)
    focus_types = Counter(str(o).rsplit("/", 1)[-1] for s, o in g.subject_objects(RDF.type) if str(o).startswith(str(HS)))
    return {
        "triples_per_graph": counts,
        "triples_total": len(g),
        "trig_bytes": path.stat().st_size,
        "conforms": conforms,
        "violations": len(results_of(report)),
        "instances_of_extension_classes": dict(focus_types),
        "timing_s": {"emit": t_emit, "serialise": t_ser, "validate": t_val},
    }, ds


def comparison_section(k: CircuitKernel, ds_a: Dataset, answers_a: list[Answer]) -> dict[str, Any]:
    """A second run under a narrower medium, joined to the first on verdict IRIs."""
    narrow = Medium(ph=(7.1, 7.3), ionic_strength=(0.15, 0.2), pmg=(3.0, 3.2))
    kb = CircuitKernel(k.corpus, k.measurements, k.legendre, narrow, k.q)
    # the stages that do not read the medium windows are shared; directions are recomputed
    for name in ("graph", "_matches", "hess", "_derived", "chemical_values"):
        kb.__dict__[name] = k.__dict__[name]
    answers_b = kb.answers()
    ds_b = emit(kb.corpus, kb.medium, answers_b, run="full-b")
    derived_a = next(g for g in ds_a.graphs() if str(g.identifier).endswith("full-a/graph/derived"))
    derived_b = next(g for g in ds_b.graphs() if str(g.identifier).endswith("full-b/graph/derived"))
    both = Dataset()
    ga, gb = both.graph(URIRef("urn:compare:a")), both.graph(URIRef("urn:compare:b"))
    for t in derived_a.triples((None, HS.label, None)):
        ga.add(t)
    for t in derived_a.triples((None, HS.question, None)):
        ga.add(t)
    for t in derived_b.triples((None, HS.label, None)):
        gb.add(t)
    t = clock()
    rows = both.query(
        """
        PREFIX hs: <https://w3id.org/nfdi-de/dcat-ap-plus/homo-sapien/>
        SELECT ?q ?la ?lb (COUNT(?v) AS ?n) WHERE {
          GRAPH <urn:compare:a> { ?v hs:label ?la ; hs:question ?q }
          GRAPH <urn:compare:b> { ?v hs:label ?lb }
        } GROUP BY ?q ?la ?lb
        """
    )
    t_join = clock() - t
    joined: list[dict[str, Any]] = []
    for q, la, lb, n in rows:
        joined.append(
            {
                "question": str(q).rsplit("/", 1)[-1],
                "a": str(la).rsplit("/", 1)[-1],
                "b": str(lb).rsplit("/", 1)[-1],
                "n": int(n),
            }
        )
    ids_a = {str(s) for s in derived_a.subjects(HS.label, None)}
    ids_b = {str(s) for s in derived_b.subjects(HS.label, None)}
    changed = [r for r in joined if r["a"] != r["b"]]
    flips = [
        {"reaction": a.about, "name": k.corpus.reactions[a.about].name, "a": a.verdict.label.value,
         "b": b.verdict.label.value, "payload_b": b.verdict.payload, "blocker_a": a.verdict.blocker}
        for a, b in zip(answers_a, answers_b, strict=True)
        if a.question == "direction" and a.verdict.label != b.verdict.label
    ]
    return {
        "medium_b": narrow.to_dict(),
        "verdicts_a": len(ids_a),
        "verdicts_b": len(ids_b),
        "shared_iris": len(ids_a & ids_b),
        "joined": joined,
        "changed": changed,
        "changed_total": sum(r["n"] for r in changed),
        "direction_changes": flips,
        "conventions_digest_differs": json.dumps(k.medium.conventions(), sort_keys=True)
        != json.dumps(narrow.conventions(), sort_keys=True),
        "timing_s": {"sparql_join": t_join},
    }


def controls_section() -> dict[str, Any]:
    """Each shape against a planted defect, and the unaltered catalogue as the control."""
    ex = load_example("central_carbon")
    kernel = CircuitKernel.from_example(ex)
    base = union(kernel.catalogue(run="control"))
    conforms, report, t = shacl(base)
    out: dict[str, Any] = {"unaltered": {"conforms": conforms, "results": len(results_of(report)), "seconds": t,
                                         "triples": len(base)}}

    def first(label: URIRef, has: URIRef | None = None) -> URIRef:
        for v in sorted(base.subjects(HS.label, label)):
            if has is None or base.value(v, has) is not None:
                return v  # type: ignore[return-value]
        raise LookupError(label)

    def supplied_with_convention() -> URIRef:
        for v in sorted(base.subjects(HS.basis, HS.Supplied)):
            if base.value(v, HS.usesConvention) is not None:
                return v  # type: ignore[return-value]
        raise LookupError("supplied")

    run = next(base.subjects(RDF.type, HS.CircuitAnalysis))
    defects = {
        "V1: payload on a non-answered verdict": lambda g: g.add((first(HS.Absent), HS.payload, Literal(0.0))),
        "V2: blocker removed from an absent verdict": lambda g: g.remove((first(HS.Absent), HS.blocker, None)),
        "gate letter on an answered verdict": lambda g: g.add((first(HS.Answered), HS.gate, HS.GateWitnessed)),
        "supplied basis without a convention": lambda g: g.remove((supplied_with_convention(), HS.usesConvention, None)),
        "label outside the vocabulary": lambda g: g.set((first(HS.Absent), HS.label, HS.Maybe)),
        "stage digest not a SHA-256": lambda g: g.set((run, HS.stageDigest, Literal("not-a-digest"))),
    }
    out["defects"] = {}
    for name, plant in defects.items():
        g = Graph()
        for t3 in base:
            g.add(t3)
        plant(g)
        conforms, report, t = shacl(g)
        res = results_of(report)
        out["defects"][name] = {
            "conforms": conforms,
            "results": len(res),
            "components": dict(Counter(r["component"] for r in res)),
            "seconds": t,
        }
    return out


def scaling_section(k: CircuitKernel, answers: list[Answer]) -> list[dict[str, Any]]:
    """Emission and validation time against the number of reactions."""
    order = sorted(k.corpus.reactions)
    out = []
    for n in (125, 250, 500, 1000, 2000):
        ids = set(order[:n])
        sub = k.corpus.subset(ids, name=f"first-{n}")
        keep = [
            a for a in answers
            if (a.about_kind == "reaction" and a.about in ids) or (a.about_kind == "species" and a.about in sub.species)
        ]
        t = clock()
        ds = emit(sub, k.medium, keep, run=f"scale-{n}")
        t_emit = clock() - t
        g = union(ds)
        conforms, _, t_val = shacl(g)
        out.append({"reactions": n, "verdicts": len(keep), "triples": len(g), "emit_s": t_emit,
                    "validate_s": t_val, "conforms": conforms})
    return out


def schema_section() -> dict[str, Any]:
    """Classes and slots of the merged schema by contributing schema, and the ancestry of the extension's classes."""
    from dcat_ap_plus_homo_sapien.datamodel import MERGED_SCHEMA_PATH
    from dcat_ap_plus_homo_sapien.datamodel import dcat_ap_plus_homo_sapien_pydantic as models
    from pydantic import BaseModel

    merged = yaml.safe_load(Path(MERGED_SCHEMA_PATH).read_text(encoding="utf-8"))
    classes, slots, enums = merged["classes"], merged.get("slots", {}), merged.get("enums", {})

    def ancestry(name: str) -> list[str]:
        chain = [name]
        while (parent := classes[chain[-1]].get("is_a")) is not None:
            chain.append(parent)
        return chain

    own = [c for c, d in classes.items() if str(d.get("from_schema", "")).endswith("/homo-sapien/")]
    pyd = [n for n, o in vars(models).items() if isinstance(o, type) and issubclass(o, BaseModel) and o.__module__ == models.__name__]
    return {
        "classes_by_schema": dict(Counter(d.get("from_schema") for d in classes.values())),
        "slots_by_schema": dict(Counter(d.get("from_schema") for d in slots.values())),
        "enums_by_schema": dict(Counter(d.get("from_schema") for d in enums.values())),
        "classes": len(classes),
        "slots": len(slots),
        "enums": len(enums),
        "extension_classes": {c: {"ancestry": ancestry(c), "class_uri": classes[c].get("class_uri"),
                                  "slots": len(classes[c].get("slots", []))} for c in own},
        "pydantic_models": len(pyd),
    }


def scidats_section(answers: list[Answer], workdir: Path) -> dict[str, Any]:
    """The verdict table in Parquet with CircuitDataset metadata, written and read back offline."""
    import pandas as pd
    from scidats import SciDatS

    from dcat_ap_plus_homo_sapien.datamodel import dcat_ap_plus_homo_sapien_pydantic as m

    attempts: list[object] = []
    connect = socket.socket.connect

    def refuse(_sock: socket.socket, address: object) -> None:
        attempts.append(address)
        raise OSError(f"network blocked: {address}")

    socket.socket.connect = refuse  # type: ignore[method-assign]
    try:
        frame = pd.DataFrame(
            {
                "question": [a.question for a in answers],
                "about": [a.about for a in answers],
                "label": [a.verdict.label.value for a in answers],
                "basis": [a.verdict.basis.value if a.verdict.basis else None for a in answers],
                "gate": [a.verdict.gate.value if a.verdict.gate else None for a in answers],
                "blocker": [a.verdict.blocker for a in answers],
                "payload": [json.dumps(a.verdict.payload, default=str) if a.verdict.is_answered else None for a in answers],
            }
        )
        hs = "https://w3id.org/nfdi-de/dcat-ap-plus/homo-sapien/"
        meta = m.CircuitDataset(
            id=f"{hs}cytosol/derived",
            title=["Verdicts on the cytosolic reactions of Homo sapiens"],
            description=["One verdict per question over 2,914 cytosolic Reactome reactions."],
            was_generated_by=[m.CircuitAnalysis(id=f"{hs}cytosol/run", title=["circuit analysis"])],
        )
        t = clock()
        sd = SciDatS(dataframe=frame)
        sd.metadata = meta
        profile = sd.metadata_profile.name
        path = workdir / "verdicts.parquet"
        sd.write_parquet(str(path))
        t_write = clock() - t
        t = clock()
        back = SciDatS(input_filename=str(path))
        t_read = clock() - t
        same = back.dataframe.equals(frame)
        kind = type(back.metadata).__name__
    finally:
        socket.socket.connect = connect  # type: ignore[method-assign]
    return {
        "rows": len(frame),
        "bytes": path.stat().st_size,
        "profile": profile,
        "round_trip_equal": bool(same),
        "metadata_class_read": kind,
        "metadata_triples": len(back.graph),
        "network_attempts": len(attempts),
        "timing_s": {"write": t_write, "read": t_read},
    }


def main() -> None:
    """Write data/extension.json."""
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[1])
    ap.add_argument("--inputs", type=Path, required=True, help="directory written by full_inputs.py")
    a = ap.parse_args()
    out: dict[str, Any] = {"software_version": __version__}
    t = clock()
    k = CircuitKernel(
        Corpus.load(a.inputs / "corpus.json"),
        MeasurementTable.from_csv(a.inputs / "measurements.csv"),
        Legendre.load(a.inputs / "microspecies.json"),
        Medium(),
    )
    out["kernel"], answers = kernel_section(k)
    print("kernel", round(clock() - t, 1), flush=True)
    out["surface"] = surface_section(k, out["kernel"]["directions"])
    print("surface", flush=True)
    with tempfile.TemporaryDirectory() as tmp:
        work = Path(tmp)
        out["catalogue"], ds = catalogue_section(k, answers, work)
        print("catalogue", out["catalogue"]["timing_s"], flush=True)
        out["comparison"] = comparison_section(k, ds, answers)
        print("comparison", out["comparison"]["changed_total"], flush=True)
        out["controls"] = controls_section()
        print("controls", flush=True)
        out["scaling"] = scaling_section(k, answers)
        print("scaling", flush=True)
        out["schema"] = schema_section()
        out["scidats"] = scidats_section(answers, work)
        print("scidats", flush=True)
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(out, indent=1, default=str) + "\n", encoding="utf-8")
    print("wrote", OUT)


if __name__ == "__main__":
    main()
