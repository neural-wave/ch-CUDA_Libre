"""Loads the privacy taxonomy (configs/privacy_taxonomy.yaml): categories, actions and replacement tags."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import yaml

DEFAULT_TAXONOMY = Path(__file__).resolve().parents[2] / "configs" / "privacy_taxonomy.yaml"


@dataclass(frozen=True)
class Category:
    id: str
    action: str  # "tag" or "keep"
    tag: str
    identifier: str
    role_tags: dict[str, str] = field(default_factory=dict)
    label: str = ""  # for explanations: "a person's name"
    sources: tuple[str, ...] = ()  # literature behind the category (HIPAA, I2B2, ...)
    why_kept: str = ""  # for categories kept by policy


class Taxonomy:
    def __init__(self, categories: dict[str, Category]):
        self.categories = categories

    @classmethod
    def load(cls, path: str | Path = DEFAULT_TAXONOMY) -> Taxonomy:
        raw = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
        categories = {
            cid: Category(
                id=cid,
                action=spec["action"],
                tag=spec["tag"],
                identifier=spec["identifier"],
                role_tags=spec.get("role_tags", {}) or {},
                label=spec.get("label", cid.lower()),
                sources=tuple(spec.get("sources", [])),
                why_kept=spec.get("why_kept", ""),
            )
            for cid, spec in raw["categories"].items()
        }
        return cls(categories)

    def action(self, category: str) -> str:
        return self.categories[category].action

    def replacement(self, category: str, role: str | None = None) -> str:
        cat = self.categories[category]
        return cat.role_tags.get(role, cat.tag) if role else cat.tag
