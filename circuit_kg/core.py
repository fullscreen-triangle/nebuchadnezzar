"""Shared types: provenance tags and verdicts.

Every element the kernel produces carries a provenance tag, and every
computation returns a verdict.  A verdict other than ANSWERED never carries
a payload (Masamune, Thm. one-graph; HFQ non-degeneracy).
"""
from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any

PKG = Path(__file__).resolve().parent
REPO = PKG.parent
RAW = PKG / "data" / "raw"
DERIVED = PKG / "data" / "derived"
RESULTS = PKG / "results"
MEDIA = PKG / "media"
REACTOME_TGZ = REPO / "demo" / "public" / "homo_sapiens.3.1.sbml.tgz"

RT = 2.577  # kJ/mol at 310 K
UNIT_ROUNDOFF = 2.0 ** -53


class Tag(str, Enum):
    STATED = "stated"        # the source record asserts it
    SUPPLIED = "supplied"    # a named convention produced it
    DERIVED = "derived"      # the kernel computed it from other values
    ABSENT = "absent"        # neither source nor convention determines it


_BASIS_ORDER = {Tag.STATED: 0, Tag.SUPPLIED: 1, Tag.DERIVED: 2}


def combine(*tags: Tag) -> Tag:
    """Basis is never laundered (dcat-ap-plus-p450 R2): tags compose as max under
    stated < supplied < derived, and anything combined with an absent value is absent."""
    tags = [Tag(t) for t in tags]
    if not tags or Tag.ABSENT in tags:
        return Tag.ABSENT
    return max(tags, key=_BASIS_ORDER.__getitem__)


@dataclass(frozen=True)
class Tagged:
    value: Any
    tag: Tag
    convention: str | None = None  # required when tag is SUPPLIED

    def __post_init__(self):
        if self.tag is Tag.SUPPLIED and not self.convention:
            raise ValueError("a supplied value must name its convention")
        if self.tag is Tag.ABSENT and self.value is not None:
            raise ValueError("an absent value carries no value")

    def as_json(self):
        d = {"value": self.value, "tag": self.tag.value}
        if self.convention:
            d["convention"] = self.convention
        return d


ABSENT = Tagged(None, Tag.ABSENT)


class Label(str, Enum):
    ANSWERED = "answered"          # payload present
    EMPTY = "certified-empty"      # well posed, answer is empty
    ABSENT = "absent"              # a required datum is absent
    UNDERDETERMINED = "underdetermined"  # several readings, no rule selects
    UNDECIDABLE = "undecidable"    # data cannot resolve it (within tolerance)
    REFUSED = "refused"            # the question is outside what the graph can state
    SUBFLOOR = "subfloor"          # a composite power below its declared target


# Empty-answer letters (dcat-ap-plus-p450 section 7.2), orthogonal to the label
GATES = {"+": "witnessed absent", "-": "not recorded", "a": "both", "open":
         "a name the source never uses", "?": "undecided"}


@dataclass
class Verdict:
    label: Label
    payload: Any = None
    blocker: str | None = None
    remedy: str | None = None
    detail: dict = field(default_factory=dict)
    gate: str | None = None

    def __post_init__(self):
        if self.label is not Label.ANSWERED and self.payload is not None:
            raise ValueError(f"verdict {self.label.value} may not carry a payload")
        if self.label not in (Label.ANSWERED, Label.EMPTY) and not self.blocker:
            raise ValueError(f"verdict {self.label.value} must name its blocker")
        if self.gate is not None:
            if self.gate not in GATES:
                raise ValueError(f"unknown gate letter {self.gate!r}")
            if self.label not in (Label.EMPTY, Label.ABSENT):
                raise ValueError("a gate letter belongs to an empty or absent answer")

    def as_json(self):
        d = {"label": self.label.value}
        for k in ("payload", "blocker", "remedy", "gate"):
            v = getattr(self, k)
            if v is not None:
                d[k] = v
        if self.detail:
            d["detail"] = self.detail
        return d


def answered(payload, **detail) -> Verdict:
    return Verdict(Label.ANSWERED, payload, detail=detail)


def absent(blocker: str, remedy: str | None = None, **detail) -> Verdict:
    return Verdict(Label.ABSENT, None, blocker, remedy, detail)
