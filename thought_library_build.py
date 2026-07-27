#!/usr/bin/env python3
"""CLI: build the Hermetic Thought Library of reactive permutations.

Enumerates sequences of the twelve reactives up to length N, batch-classifies
each chain as parasitic / symbiotic / neutral / catalytic via GenAI (or a
local hermetic heuristic in --dry-run), then legislates canons.

Doctrine:
  - Df (Diffuse) = Pisces = 12th reactive = natural pathogen
  - Water = death and release
  - Analysis invokes the seven Hermetic principles

Examples
--------
  # Scaffold without API (heuristic legislation)
  python thought_library_build.py --dry-run -N 2 --limit 48

  # Full GenAI pass on length-1..2 universe (132 + 12 = 144 sequences)
  python thought_library_build.py -N 2 --batch-size 16

  # Sample length-3 space with pathogen enrichment
  python thought_library_build.py -N 3 --limit 96 --batch-size 12
"""

from __future__ import annotations

import argparse
import logging
import os
import sys
from pathlib import Path

from dotenv import load_dotenv

from attention_algebra.config import DEFAULT_OPENROUTER_MODEL, OPENROUTER_MODELS
from attention_algebra.hermetic import HERMETIC_TABLE, NATURAL_PATHOGEN
from attention_algebra.thought_library import (
    ThoughtLibrarian,
    count_sequences,
    save_library,
)
from attention_algebra.terminals import TERMINAL_ORDER


def _configure_logging(verbose: bool) -> None:
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(
        level=level,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        datefmt="%H:%M:%S",
    )


def main(argv: list[str] | None = None) -> int:
    load_dotenv()
    parser = argparse.ArgumentParser(
        description=(
            "Build a Thought Library: permute reactives up to length N, "
            "classify parasites under hermetic law, legislate canons. "
            f"Natural pathogen = {NATURAL_PATHOGEN} (Pisces / water of death)."
        )
    )
    parser.add_argument(
        "-N",
        "--max-length",
        type=int,
        default=2,
        help="Maximum sequence length (default: 2)",
    )
    parser.add_argument(
        "--min-length",
        type=int,
        default=1,
        help="Minimum sequence length (default: 1)",
    )
    parser.add_argument(
        "--with-replacement",
        action="store_true",
        help="Allow repeated reactives in a sequence (product vs permutations)",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Cap number of sequences (random sample; enriches pathogen-bearing)",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=12,
        help="Sequences per GenAI classification call (default: 12)",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="RNG seed for sampling (default: 42)",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="No API calls — hermetic heuristic classification + legislation",
    )
    parser.add_argument(
        "--provider",
        choices=("openrouter", "llama.cpp"),
        default="openrouter",
    )
    parser.add_argument(
        "--model",
        default=DEFAULT_OPENROUTER_MODEL,
        help=f"Model id (default: {DEFAULT_OPENROUTER_MODEL}). "
        f"OpenRouter presets: {', '.join(OPENROUTER_MODELS[:3])}…",
    )
    parser.add_argument(
        "-o",
        "--out-dir",
        type=Path,
        default=Path("thought_library_out"),
        help="Output directory (default: thought_library_out/)",
    )
    parser.add_argument(
        "-v",
        "--verbose",
        action="store_true",
    )
    parser.add_argument(
        "--print-table",
        action="store_true",
        help="Print hermetic reactive table and exit",
    )
    args = parser.parse_args(argv)
    _configure_logging(args.verbose)

    if args.print_table:
        print(f"{'Sym':<4} {'Name':<10} {'Sign':<12} {'Element':<6} Functional expansion")
        print("-" * 88)
        for sym in TERMINAL_ORDER:
            h = HERMETIC_TABLE[sym]
            flag = "  *PATHOGEN*" if h.is_natural_pathogen else ""
            print(
                f"{h.symbol:<4} {h.name:<10} {h.sign:<12} {h.element:<6} "
                f"{h.functional_display}{flag}"
            )
        return 0

    universe = count_sequences(
        args.max_length,
        with_replacement=args.with_replacement,
        min_length=args.min_length,
    )
    print(
        f"Universe size for N={args.min_length}..{args.max_length} "
        f"(replacement={args.with_replacement}): {universe}"
    )
    if args.limit:
        print(f"Sampling limit: {args.limit}")
    if universe > 5000 and args.limit is None and not args.dry_run:
        print(
            "WARNING: large universe without --limit will cost many API calls. "
            "Consider --limit 128 or --dry-run first.",
            file=sys.stderr,
        )

    if not args.dry_run and args.provider == "openrouter":
        if not os.getenv("OPENROUTER_API_KEY"):
            print(
                "OPENROUTER_API_KEY missing. Use --dry-run or set the key.",
                file=sys.stderr,
            )
            return 2

    librarian = ThoughtLibrarian(
        model_name=args.model,
        provider=args.provider,
        dry_run=args.dry_run,
    )
    lib = librarian.build(
        max_length=args.max_length,
        min_length=args.min_length,
        with_replacement=args.with_replacement,
        batch_size=args.batch_size,
        limit=args.limit,
        seed=args.seed,
    )
    paths = save_library(lib, args.out_dir)

    print()
    print("=== Thought Library complete ===")
    print(f"  Pathogen:     {lib.pathogen_symbol} (Pisces / water of death & release)")
    print(f"  Analyzed:     {lib.stats.get('analyzed')}")
    print(f"  Parasitic:    {lib.stats.get('parasitic')}")
    print(f"  Symbiotic:    {lib.stats.get('symbiotic')}")
    print(f"  Catalytic:    {lib.stats.get('catalytic')}")
    print(f"  Neutral:      {lib.stats.get('neutral')}")
    print(f"  JSON:         {paths['json']}")
    print(f"  Legislation:  {paths['legislation']}")
    print(f"  Parasitic:    {paths['parasitic']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
