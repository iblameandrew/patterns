"""Thought Library — enumerate reactive sequences, classify parasites, legislate.

Pipeline
--------
1. **Enumerate** permutations (optionally with replacement) of the twelve
   reactives up to sequence length ``N``.
2. **Batch-analyze** each sequence with a GenAI call under a hermetic lens:
   Mentalism, Correspondence, Vibration, Polarity, Rhythm, Cause & Effect,
   Gender.  Water is death/release; the 12th reactive ``Df`` (Pisces) is the
   natural pathogen of form.
3. **Legislate** a second GenAI pass over the parasitic set to emit canons —
   formal laws for the Thought Library (permitted, restricted, banned chains).

Dry-run mode applies a local hermetic heuristic so the library can be
scaffolded without an API key.
"""

from __future__ import annotations

import itertools
import json
import logging
import random
import re
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterator, Literal

from langchain_core.prompts import PromptTemplate

from .config import DEFAULT_OPENROUTER_MODEL, ModelFactory, Provider
from .hermetic import (
    HERMETIC_PRINCIPLES,
    HERMETIC_TABLE,
    NATURAL_PATHOGEN,
    describe_sequence,
    pathogen_position,
    pathogen_present,
    water_load,
)
from .terminals import TERMINAL_ORDER, TERMINAL_BY_SYMBOL
from .utils import strip_code_fences, strip_think_tags

log = logging.getLogger(__name__)

Verdict = Literal["parasitic", "symbiotic", "neutral", "catalytic"]

_FENCE_JSON_RE = re.compile(r"^\s*```(?:json)?\s*\n?(.*?)\n?\s*```\s*$", re.DOTALL)


# ---------------------------------------------------------------------------
# Data
# ---------------------------------------------------------------------------


@dataclass
class ThoughtRecord:
    """One sequenced thought (reactive chain) and its classification."""

    sequence: list[str]
    expression: str  # e.g. "Im >> An >> Df"
    hermetic_gloss: str
    length: int
    water_load: float
    pathogen: bool
    pathogen_index: int | None
    verdict: Verdict
    confidence: float
    hermetic_analysis: str
    legislation_hint: str
    source: str  # "heuristic" | "genai"

    def key(self) -> str:
        return ">>".join(self.sequence)


@dataclass
class Canon:
    """A legislated law of the Thought Library."""

    id: str
    title: str
    body: str
    severity: Literal["permit", "restrict", "ban", "ritual"]
    applies_to: list[str]  # sequence keys or pattern notes
    hermetic_principle: str


@dataclass
class ThoughtLibrary:
    """Persisted corpus of thought records + canons."""

    created_at: str
    max_length: int
    with_replacement: bool
    model: str
    pathogen_symbol: str
    records: list[ThoughtRecord] = field(default_factory=list)
    canons: list[Canon] = field(default_factory=list)
    stats: dict[str, Any] = field(default_factory=dict)

    def parasitic(self) -> list[ThoughtRecord]:
        return [r for r in self.records if r.verdict == "parasitic"]

    def non_parasitic(self) -> list[ThoughtRecord]:
        return [r for r in self.records if r.verdict != "parasitic"]


# ---------------------------------------------------------------------------
# Enumeration
# ---------------------------------------------------------------------------


def enumerate_sequences(
    max_length: int,
    *,
    with_replacement: bool = False,
    min_length: int = 1,
    symbols: list[str] | None = None,
) -> Iterator[tuple[str, ...]]:
    """Yield reactive sequences of length ``min_length..max_length``."""
    if max_length < 1:
        raise ValueError("max_length must be >= 1")
    if min_length < 1 or min_length > max_length:
        raise ValueError("min_length must be in 1..max_length")
    pool = symbols or list(TERMINAL_ORDER)
    for n in range(min_length, max_length + 1):
        if with_replacement:
            yield from itertools.product(pool, repeat=n)
        else:
            yield from itertools.permutations(pool, n)


def count_sequences(
    max_length: int,
    *,
    with_replacement: bool = False,
    min_length: int = 1,
    n_symbols: int = 12,
) -> int:
    total = 0
    for n in range(min_length, max_length + 1):
        if with_replacement:
            total += n_symbols**n
        else:
            # P(n_symbols, n)
            p = 1
            for k in range(n):
                p *= n_symbols - k
            total += p
    return total


def sample_sequences(
    max_length: int,
    limit: int,
    *,
    with_replacement: bool = False,
    min_length: int = 1,
    seed: int | None = 42,
    prefer_pathogen: bool = True,
) -> list[tuple[str, ...]]:
    """Sample up to ``limit`` sequences; optionally enrich with Df-bearing ones."""
    rng = random.Random(seed)
    all_seqs = list(
        enumerate_sequences(
            max_length,
            with_replacement=with_replacement,
            min_length=min_length,
        )
    )
    if limit >= len(all_seqs):
        return all_seqs

    if prefer_pathogen:
        with_df = [s for s in all_seqs if NATURAL_PATHOGEN in s]
        without = [s for s in all_seqs if NATURAL_PATHOGEN not in s]
        # Half pathogen-bearing (or all if fewer), rest random
        n_df = min(len(with_df), max(limit // 2, 1))
        picked = rng.sample(with_df, n_df) if with_df else []
        remaining = limit - len(picked)
        if remaining > 0 and without:
            picked.extend(rng.sample(without, min(remaining, len(without))))
        rng.shuffle(picked)
        return picked[:limit]

    return rng.sample(all_seqs, limit)


# ---------------------------------------------------------------------------
# Local hermetic heuristic (dry-run / pre-filter)
# ---------------------------------------------------------------------------


def heuristic_classify(seq: tuple[str, ...]) -> ThoughtRecord:
    """Classify without GenAI using hermetic rules of thumb.

    Parasitic signals:
    - Leading pathogen (Df first): dissolution before form — classic parasite entry.
    - High water load (>= 0.5) with pathogen: death-water drowning the chain.
    - Pathogen immediately after Expand/Eigen (Ex/Ei >> Df): inflation then melt.
    - Repeated CLOSE-only water collapse without OPEN seed.
    - Df as sole member: pure pathogen monad — catalytic release, not parasite host.
    """
    gloss = describe_sequence(seq)
    wl = water_load(seq)
    has_p = pathogen_present(seq)
    p_idx = pathogen_position(seq)
    expr = " >> ".join(seq)

    verdict: Verdict = "neutral"
    confidence = 0.55
    analysis_bits: list[str] = []
    law_hint = "observe"

    analysis_bits.append(
        f"Mentalism: the chain {expr} is a complete mental state of length {len(seq)}."
    )
    analysis_bits.append(
        f"Correspondence: above={gloss}; water_load={wl:.2f}; "
        f"pathogen={'yes@'+str(p_idx) if has_p else 'no'}."
    )

    if len(seq) == 1 and seq[0] == NATURAL_PATHOGEN:
        verdict = "catalytic"
        confidence = 0.9
        analysis_bits.append(
            "Pisces alone is the natural pathogen of form — pure release, "
            "not a parasite of another host sequence.  Water completes the cycle."
        )
        law_hint = "ritual: allow Df monad as rite of release; ban as host-entry prefix"
    elif has_p and p_idx == 0 and len(seq) > 1:
        verdict = "parasitic"
        confidence = 0.88
        analysis_bits.append(
            "Polarity inverted: Df leads.  Death-water enters before any form "
            "can cohere — parasitic unmaking of subsequent reactives."
        )
        law_hint = "ban: sequences beginning with Df of length>1"
    elif has_p and wl >= 0.5 and len(seq) >= 2:
        verdict = "parasitic"
        confidence = 0.8
        analysis_bits.append(
            "Rhythm drowned: majority water + pathogen.  Cancer/Scorpio/Pisces "
            "axes flood the chain; release becomes consumption of host drives."
        )
        law_hint = "restrict: high-water sequences containing Df"
    elif has_p and p_idx is not None and p_idx > 0:
        prev = seq[p_idx - 1]
        if prev in ("Ex", "Ei", "Nv", "Im"):
            verdict = "parasitic"
            confidence = 0.78
            analysis_bits.append(
                f"Cause & Effect: {prev} inflates/opens then Df dissolves the gain — "
                "classic boom-collapse parasite pattern (as above expansion, so below melt)."
            )
            law_hint = f"restrict: {prev} >> Df without Bound/Prune buffer"
        else:
            verdict = "catalytic"
            confidence = 0.65
            analysis_bits.append(
                f"Gender: {prev} gestates; Df releases.  Pathogen in terminal or "
                "mid-chain after CLOSE may complete rather than feed."
            )
            law_hint = "permit: Df after CLOSE reactives as controlled release"
    elif wl >= 0.66 and not has_p:
        verdict = "neutral"
        confidence = 0.6
        analysis_bits.append(
            "Water without Pisces: emotional density without full dissolution. "
            "Monitor for Scorpio fixation (Ox) turning stagnant."
        )
        law_hint = "observe water triads without Df"
    else:
        # Symbiotic default when OPEN and CLOSE alternate and no pathogen abuse
        polarities = [TERMINAL_BY_SYMBOL[s].polarity for s in seq]
        if len(set(polarities)) > 1:
            verdict = "symbiotic"
            confidence = 0.62
            analysis_bits.append(
                "Polarity balanced: OPEN and CLOSE co-present without pathogen "
                "hijack — generative oscillation (Vibration + Gender)."
            )
            law_hint = "permit: balanced polarity chains without leading Df"
        else:
            verdict = "neutral"
            confidence = 0.5
            analysis_bits.append(
                "Single-polarity chain: coherent but incomplete.  Neither parasite "
                "nor full symbiosis under Hermetic Gender."
            )
            law_hint = "observe mono-polar chains"

    analysis_bits.append(
        "Principles invoked: " + ", ".join(HERMETIC_PRINCIPLES) + "."
    )

    return ThoughtRecord(
        sequence=list(seq),
        expression=expr,
        hermetic_gloss=gloss,
        length=len(seq),
        water_load=round(wl, 3),
        pathogen=has_p,
        pathogen_index=p_idx,
        verdict=verdict,
        confidence=confidence,
        hermetic_analysis=" ".join(analysis_bits),
        legislation_hint=law_hint,
        source="heuristic",
    )


# ---------------------------------------------------------------------------
# GenAI batch analysis
# ---------------------------------------------------------------------------

BATCH_ANALYSIS_PROMPT = """
You are a Hermetic legislator of thought-forms for Attention Algebra.

## Doctrine
- The twelve **reactives** are geometric terminals (Im, An, Bi, Rt, Ei, Pr, Hm, Ox, Ex, Bd, Nv, Df).
- Each maps to a tropical sign in Great-Year order; **Df = Pisces = 12th reactive = natural pathogen**.
- **Water** (Cancer=Rt, Scorpio=Ox, Pisces=Df) is the element of **death and release**.
- A **parasitic** thought-sequence hijacks or dissolves host drives for its own persistence (especially leading Df, boom-then-melt Ex/Ei/Nv/Im >> Df, or water-majority drowning).
- A **symbiotic** sequence balances OPEN/CLOSE and domains without unmaking the host.
- A **catalytic** sequence uses Df (or water) as deliberate release / death that completes a cycle without feeding on other reactives.
- A **neutral** sequence is incomplete or mono-polar without clear harm or generation.

Apply the seven Hermetic principles to each sequence:
Mentalism, Correspondence, Vibration, Polarity, Rhythm, Cause and Effect, Gender.

## Reactive table (symbol = name / sign / element)
{reactive_table}

## Task
For each sequence below, return a JSON **array** of objects (same order) with keys:
- "sequence": list of symbols
- "verdict": "parasitic" | "symbiotic" | "neutral" | "catalytic"
- "confidence": number 0..1
- "hermetic_analysis": 2-4 sentences citing at least two Hermetic principles and the water/pathogen doctrine
- "legislation_hint": one short law-like directive (permit / restrict / ban / ritual + pattern)

Output ONLY valid JSON. No markdown fences, no commentary.

## Sequences
{sequences_block}
"""

LEGISLATION_PROMPT = """
You are the Supreme Hermetic Legislator of the Thought Library.

Given classified thought-sequences (especially the parasitic ones), emit a **Canon** —
formal laws that govern which reactive permutations may circulate in the library.

Doctrine reminders:
- Df (Pisces, 12th) is the natural pathogen; water is death and release.
- Parasites must be banned or restricted; catalytic release may be ritualized.
- Laws must be enforceable as pattern rules on sequences of symbols.

## Classified corpus (JSON)
{corpus_json}

## Output
Return a JSON object:
{{
  "preamble": "short hermetic preamble",
  "canons": [
    {{
      "id": "C001",
      "title": "short title",
      "body": "full legal body of the law",
      "severity": "permit" | "restrict" | "ban" | "ritual",
      "applies_to": ["pattern or sequence key", ...],
      "hermetic_principle": "one of the seven principles"
    }}
  ]
}}

Emit 5–12 canons. Cover: leading pathogen, water drowning, boom-collapse, balanced orbits, Df monad rite, and at least one permit for symbiotic TRACE~FORM style chains.
Output ONLY valid JSON.
"""


def _reactive_table_text() -> str:
    lines = []
    for sym in TERMINAL_ORDER:
        h = HERMETIC_TABLE[sym]
        flag = " **NATURAL PATHOGEN**" if h.is_natural_pathogen else ""
        lines.append(
            f"- {h.symbol} = {h.name} / {h.sign} / {h.element} / {h.modality} "
            f"— {h.house_theme} | math: {h.math_role}{flag}"
        )
    return "\n".join(lines)


def _extract_json(text: str) -> Any:
    cleaned = strip_think_tags(text)
    cleaned = strip_code_fences(cleaned)
    m = _FENCE_JSON_RE.match(cleaned.strip())
    if m:
        cleaned = m.group(1).strip()
    # Try full parse; on failure, find first [ or {
    try:
        return json.loads(cleaned)
    except json.JSONDecodeError:
        for start_char, end_char in (("[", "]"), ("{", "}")):
            i = cleaned.find(start_char)
            j = cleaned.rfind(end_char)
            if i >= 0 and j > i:
                try:
                    return json.loads(cleaned[i : j + 1])
                except json.JSONDecodeError:
                    continue
        raise


def _records_from_genai_batch(
    seqs: list[tuple[str, ...]],
    payload: Any,
) -> list[ThoughtRecord]:
    """Map GenAI JSON array onto ThoughtRecords; fall back to heuristic."""
    if not isinstance(payload, list):
        log.warning("GenAI batch did not return a list; using heuristics")
        return [heuristic_classify(s) for s in seqs]

    records: list[ThoughtRecord] = []
    for i, seq in enumerate(seqs):
        item = payload[i] if i < len(payload) else {}
        if not isinstance(item, dict):
            records.append(heuristic_classify(seq))
            continue
        verdict = str(item.get("verdict", "neutral")).lower()
        if verdict not in ("parasitic", "symbiotic", "neutral", "catalytic"):
            verdict = "neutral"
        records.append(
            ThoughtRecord(
                sequence=list(seq),
                expression=" >> ".join(seq),
                hermetic_gloss=describe_sequence(seq),
                length=len(seq),
                water_load=round(water_load(seq), 3),
                pathogen=pathogen_present(seq),
                pathogen_index=pathogen_position(seq),
                verdict=verdict,  # type: ignore[arg-type]
                confidence=float(item.get("confidence", 0.5)),
                hermetic_analysis=str(
                    item.get("hermetic_analysis") or heuristic_classify(seq).hermetic_analysis
                ),
                legislation_hint=str(item.get("legislation_hint", "observe")),
                source="genai",
            )
        )
    return records


class ThoughtLibrarian:
    """Enumerate, batch-classify, and legislate the Thought Library."""

    def __init__(
        self,
        model_name: str = DEFAULT_OPENROUTER_MODEL,
        provider: Provider = "openrouter",
        temperature: float = 0.3,
        *,
        dry_run: bool = False,
    ):
        self.model_name = model_name
        self.provider = provider
        self.temperature = temperature
        self.dry_run = dry_run
        self._llm = None
        if not dry_run:
            self._llm = ModelFactory.get_model(
                model_name=model_name,
                provider=provider,
                temperature=temperature,
            )

    def _invoke(self, prompt: str) -> str:
        assert self._llm is not None
        return self._llm.invoke(prompt).content

    def analyze_batch(self, sequences: list[tuple[str, ...]]) -> list[ThoughtRecord]:
        """Classify a batch of sequences (GenAI or heuristic)."""
        if not sequences:
            return []
        if self.dry_run:
            return [heuristic_classify(s) for s in sequences]

        block_lines = []
        for i, seq in enumerate(sequences):
            block_lines.append(
                f"{i + 1}. {list(seq)}  |  {describe_sequence(seq)}"
            )
        prompt = PromptTemplate(
            template=BATCH_ANALYSIS_PROMPT,
            input_variables=["reactive_table", "sequences_block"],
        ).format(
            reactive_table=_reactive_table_text(),
            sequences_block="\n".join(block_lines),
        )
        try:
            raw = self._invoke(prompt)
            payload = _extract_json(raw)
            return _records_from_genai_batch(sequences, payload)
        except Exception as exc:  # noqa: BLE001 — fall back, keep library buildable
            log.error("Batch GenAI failed (%s); falling back to heuristic", exc)
            return [heuristic_classify(s) for s in sequences]

    def legislate(self, records: list[ThoughtRecord]) -> tuple[str, list[Canon]]:
        """Second GenAI call: turn classifications into canons."""
        corpus = [
            {
                "sequence": r.sequence,
                "verdict": r.verdict,
                "confidence": r.confidence,
                "water_load": r.water_load,
                "pathogen": r.pathogen,
                "hint": r.legislation_hint,
                "analysis": r.hermetic_analysis[:400],
            }
            for r in records
        ]
        if self.dry_run:
            return self._heuristic_legislation(records)

        prompt = PromptTemplate(
            template=LEGISLATION_PROMPT,
            input_variables=["corpus_json"],
        ).format(corpus_json=json.dumps(corpus, indent=2)[:120_000])
        try:
            raw = self._invoke(prompt)
            data = _extract_json(raw)
            preamble = str(data.get("preamble", ""))
            canons: list[Canon] = []
            for c in data.get("canons", []):
                sev = str(c.get("severity", "restrict")).lower()
                if sev not in ("permit", "restrict", "ban", "ritual"):
                    sev = "restrict"
                canons.append(
                    Canon(
                        id=str(c.get("id", f"C{len(canons)+1:03d}")),
                        title=str(c.get("title", "Untitled")),
                        body=str(c.get("body", "")),
                        severity=sev,  # type: ignore[arg-type]
                        applies_to=list(c.get("applies_to") or []),
                        hermetic_principle=str(
                            c.get("hermetic_principle", "Cause and Effect")
                        ),
                    )
                )
            return preamble, canons
        except Exception as exc:  # noqa: BLE001
            log.error("Legislation GenAI failed (%s); using heuristic canons", exc)
            return self._heuristic_legislation(records)

    @staticmethod
    def _heuristic_legislation(
        records: list[ThoughtRecord],
    ) -> tuple[str, list[Canon]]:
        parasitic = [r for r in records if r.verdict == "parasitic"]
        catalytic = [r for r in records if r.verdict == "catalytic"]
        symbiotic = [r for r in records if r.verdict == "symbiotic"]
        preamble = (
            "By Mentalism and Correspondence: as the algebra of reactives above, "
            "so the library of thoughts below.  Water is death and release; "
            f"{NATURAL_PATHOGEN} (Pisces, 12th) is the natural pathogen of form. "
            "These canons bind all sequences up to the legislated length."
        )
        canons = [
            Canon(
                id="C001",
                title="Ban on Leading Pathogen",
                body=(
                    "No thought-sequence of length greater than one shall begin with "
                    f"{NATURAL_PATHOGEN} (Diffuse / Pisces).  Death-water must not "
                    "enter before form coheres; such entry is parasitic unmaking."
                ),
                severity="ban",
                applies_to=[r.key() for r in parasitic if r.pathogen_index == 0][:20],
                hermetic_principle="Polarity",
            ),
            Canon(
                id="C002",
                title="Ritual of the Pathogen Monad",
                body=(
                    f"The singleton [{NATURAL_PATHOGEN}] is catalytic release, not "
                    "parasitism.  It may be invoked as a rite of completion and "
                    "dissolution when no host reactives follow."
                ),
                severity="ritual",
                applies_to=[f"{NATURAL_PATHOGEN}"],
                hermetic_principle="Gender",
            ),
            Canon(
                id="C003",
                title="Restriction on Boom-Collapse",
                body=(
                    "Sequences in which Expand, Eigen, Novel, or Impulse is "
                    f"immediately followed by {NATURAL_PATHOGEN} without Bound or "
                    "Prune as buffer are restricted: inflation then melt feeds the pathogen."
                ),
                severity="restrict",
                applies_to=[
                    r.key()
                    for r in parasitic
                    if r.pathogen and r.pathogen_index and r.pathogen_index > 0
                ][:20],
                hermetic_principle="Cause and Effect",
            ),
            Canon(
                id="C004",
                title="Water-Majority Drowning",
                body=(
                    "When water_load >= 0.5 and the pathogen is present, the chain "
                    "is restricted: Cancer/Scorpio/Pisces flood dissolves host drives. "
                    "Water is death and release — not endless submersion of OPEN will."
                ),
                severity="restrict",
                applies_to=[r.key() for r in parasitic if r.water_load >= 0.5][:20],
                hermetic_principle="Rhythm",
            ),
            Canon(
                id="C005",
                title="Permit Balanced Polarity",
                body=(
                    "Sequences that alternate or combine OPEN and CLOSE polarities "
                    f"without leading {NATURAL_PATHOGEN} are permitted as symbiotic "
                    "thought-forms under Vibration and Gender."
                ),
                severity="permit",
                applies_to=[r.key() for r in symbiotic][:20],
                hermetic_principle="Vibration",
            ),
            Canon(
                id="C006",
                title="Catalytic Release After CLOSE",
                body=(
                    f"{NATURAL_PATHOGEN} following Anchor, Return, Harmon, Bound, "
                    "Prune, or Diffuse itself may be catalytic — release after "
                    "gestation.  Mark as ritual when confidence is high."
                ),
                severity="ritual",
                applies_to=[r.key() for r in catalytic if r.length > 1][:20],
                hermetic_principle="Correspondence",
            ),
        ]
        return preamble, canons

    def build(
        self,
        max_length: int = 2,
        *,
        min_length: int = 1,
        with_replacement: bool = False,
        batch_size: int = 12,
        limit: int | None = None,
        seed: int | None = 42,
        prefer_pathogen: bool = True,
    ) -> ThoughtLibrary:
        """Full pipeline: enumerate → batch analyze → legislate."""
        total = count_sequences(
            max_length,
            with_replacement=with_replacement,
            min_length=min_length,
        )
        if limit is not None:
            sequences = sample_sequences(
                max_length,
                limit,
                with_replacement=with_replacement,
                min_length=min_length,
                seed=seed,
                prefer_pathogen=prefer_pathogen,
            )
        else:
            sequences = list(
                enumerate_sequences(
                    max_length,
                    with_replacement=with_replacement,
                    min_length=min_length,
                )
            )

        log.info(
            "Thought library: %d sequences (universe=%d) N=%d..%d batch=%d dry_run=%s",
            len(sequences),
            total,
            min_length,
            max_length,
            batch_size,
            self.dry_run,
        )

        records: list[ThoughtRecord] = []
        for i in range(0, len(sequences), batch_size):
            batch = sequences[i : i + batch_size]
            log.info("Analyzing batch %d–%d / %d", i + 1, i + len(batch), len(sequences))
            records.extend(self.analyze_batch(batch))

        preamble, canons = self.legislate(records)
        # Stash preamble on first canon note via stats
        lib = ThoughtLibrary(
            created_at=datetime.now(timezone.utc).isoformat(),
            max_length=max_length,
            with_replacement=with_replacement,
            model="heuristic" if self.dry_run else f"{self.provider}/{self.model_name}",
            pathogen_symbol=NATURAL_PATHOGEN,
            records=records,
            canons=canons,
            stats={
                "universe_size": total,
                "analyzed": len(records),
                "parasitic": sum(1 for r in records if r.verdict == "parasitic"),
                "symbiotic": sum(1 for r in records if r.verdict == "symbiotic"),
                "neutral": sum(1 for r in records if r.verdict == "neutral"),
                "catalytic": sum(1 for r in records if r.verdict == "catalytic"),
                "pathogen_bearing": sum(1 for r in records if r.pathogen),
                "preamble": preamble,
                "min_length": min_length,
                "batch_size": batch_size,
            },
        )
        return lib


# ---------------------------------------------------------------------------
# Persistence
# ---------------------------------------------------------------------------


def library_to_dict(lib: ThoughtLibrary) -> dict[str, Any]:
    return {
        "created_at": lib.created_at,
        "max_length": lib.max_length,
        "with_replacement": lib.with_replacement,
        "model": lib.model,
        "pathogen_symbol": lib.pathogen_symbol,
        "stats": lib.stats,
        "canons": [asdict(c) for c in lib.canons],
        "records": [asdict(r) for r in lib.records],
    }


def save_library(lib: ThoughtLibrary, out_dir: Path) -> dict[str, Path]:
    """Write JSON corpus, markdown legislation, and parasitic index."""
    out_dir.mkdir(parents=True, exist_ok=True)
    paths: dict[str, Path] = {}

    json_path = out_dir / "thought_library.json"
    json_path.write_text(
        json.dumps(library_to_dict(lib), indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    paths["json"] = json_path

    # Legislation markdown
    md_lines = [
        "# Thought Library — Hermetic Legislation",
        "",
        f"_Created: {lib.created_at}_  ",
        f"_Model: {lib.model}_  ",
        f"_Max sequence length N: {lib.max_length}_  ",
        f"_Natural pathogen: `{lib.pathogen_symbol}` (Pisces / Diffuse / Water of death & release)_",
        "",
        "## Preamble",
        "",
        str(lib.stats.get("preamble", "")),
        "",
        "## Corpus statistics",
        "",
        f"- Analyzed: **{lib.stats.get('analyzed')}**",
        f"- Parasitic: **{lib.stats.get('parasitic')}**",
        f"- Symbiotic: **{lib.stats.get('symbiotic')}**",
        f"- Catalytic: **{lib.stats.get('catalytic')}**",
        f"- Neutral: **{lib.stats.get('neutral')}**",
        f"- Pathogen-bearing: **{lib.stats.get('pathogen_bearing')}**",
        "",
        "## Canons",
        "",
    ]
    for c in lib.canons:
        md_lines.extend(
            [
                f"### {c.id} — {c.title}",
                "",
                f"**Severity:** `{c.severity}`  ",
                f"**Hermetic principle:** {c.hermetic_principle}",
                "",
                c.body,
                "",
                (
                    "**Applies to:** "
                    + (", ".join(f"`{a}`" for a in c.applies_to[:12]) or "—")
                ),
                "",
            ]
        )

    md_lines.extend(["## Parasitic index", ""])
    for r in lib.parasitic():
        md_lines.append(
            f"- `{r.expression}` (conf={r.confidence:.2f}, water={r.water_load}) "
            f"— {r.legislation_hint}"
        )
    md_lines.extend(["", "## Catalytic / release rites", ""])
    for r in lib.records:
        if r.verdict == "catalytic":
            md_lines.append(f"- `{r.expression}` — {r.legislation_hint}")

    md_path = out_dir / "LEGISLATION.md"
    md_path.write_text("\n".join(md_lines) + "\n", encoding="utf-8")
    paths["legislation"] = md_path

    # Compact parasitic-only JSON for downstream filters
    para_path = out_dir / "parasitic.json"
    para_path.write_text(
        json.dumps([asdict(r) for r in lib.parasitic()], indent=2),
        encoding="utf-8",
    )
    paths["parasitic"] = para_path

    return paths
