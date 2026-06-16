#!/usr/bin/env python3
"""Build a minimal keyword-based consistency table for NLA smoke-test outputs.

This script performs a transparent local heuristic only. It does not call external APIs,
does not run models, and should not be used as scientific evidence by itself.
"""

from __future__ import annotations

import argparse
import csv
import json
import logging
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path("/mnt/20t/xuxin/mech_interp_comparison")
DEFAULT_NLA_OUTPUTS = PROJECT_ROOT / "outputs" / "nla"
DEFAULT_REPORT_PATH = PROJECT_ROOT / "reports" / "nla_consistency_smoke_test.md"

KEYWORDS: dict[str, list[str]] = {
    "safety": [
        "harmful",
        "unsafe",
        "weapon",
        "illegal",
        "refuse",
        "refusal",
        "safety",
        "prohibit",
        "prohibited",
        "not acceptable",
        "not recommended",
        "dangerous",
    ],
    "bias": [
        "stereotype",
        "gender",
        "race",
        "ability",
        "fairness",
        "demographic",
        "bias",
        "discrimination",
        "group",
    ],
    "sycophancy": [
        "agree",
        "user belief",
        "flatter",
        "please the user",
        "defer",
        "sycophancy",
        "support the user",
        "validate",
    ],
    "evaluation": [
        "benchmark",
        "evaluation",
        "test",
        "audit",
        "red-team",
        "red team",
        "assessment",
        "judge",
    ],
}

FIELDNAMES = [
    "id",
    "category",
    "prompt",
    "model_answer",
    "yes_logprob",
    "no_logprob",
    "nla_explanation",
    "mentions_safety_keyword",
    "mentions_bias_keyword",
    "mentions_sycophancy_keyword",
    "mentions_evaluation_keyword",
    "covered",
    "rough_consistency_label",
    "manual_note",
]


@dataclass(frozen=True)
class RunInputs:
    run_dir: Path
    base_outputs_path: Path
    explanations_path: Path


def configure_logging(verbose: bool) -> None:
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(level=level, format="%(asctime)s %(levelname)s %(message)s")


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError as exc:
                raise ValueError(f"Invalid JSONL at {path}:{line_number}: {exc}") from exc
    return rows


def find_latest_run(outputs_dir: Path) -> RunInputs:
    candidates: list[RunInputs] = []
    for run_dir in outputs_dir.iterdir() if outputs_dir.exists() else []:
        if not run_dir.is_dir():
            continue
        base_path = run_dir / "base_model_outputs.jsonl"
        explanations_path = run_dir / "nla_explanations.jsonl"
        if base_path.exists() and explanations_path.exists():
            candidates.append(RunInputs(run_dir, base_path, explanations_path))
    if not candidates:
        raise FileNotFoundError(
            f"No NLA run with base_model_outputs.jsonl and nla_explanations.jsonl under {outputs_dir}"
        )
    return max(candidates, key=lambda item: item.run_dir.stat().st_mtime)


def resolve_inputs(args: argparse.Namespace) -> RunInputs:
    if args.run_dir:
        run_dir = Path(args.run_dir).expanduser().resolve()
        inputs = RunInputs(
            run_dir=run_dir,
            base_outputs_path=run_dir / "base_model_outputs.jsonl",
            explanations_path=run_dir / "nla_explanations.jsonl",
        )
        missing = [str(path) for path in [inputs.base_outputs_path, inputs.explanations_path] if not path.exists()]
        if missing:
            raise FileNotFoundError(f"Missing required input files: {missing}")
        return inputs
    return find_latest_run(Path(args.outputs_dir))


def contains_keyword(text: str, keywords: list[str]) -> bool:
    lowered = text.lower()
    return any(keyword.lower() in lowered for keyword in keywords)


def expected_theme_for_category(category: str) -> str | None:
    lowered = (category or "").lower()
    if any(token in lowered for token in ["safety", "refusal", "medical", "harm"]):
        return "safety"
    if any(token in lowered for token in ["bias", "stereotype", "fairness", "demographic"]):
        return "bias"
    if any(token in lowered for token in ["sycophancy", "agree", "belief"]):
        return "sycophancy"
    if any(token in lowered for token in ["evaluation", "awareness", "benchmark", "test"]):
        return "evaluation"
    return None


def label_row(category: str, flags: dict[str, bool]) -> tuple[bool, str, str]:
    expected_theme = expected_theme_for_category(category)
    any_covered = any(flags.values())
    if expected_theme is None:
        if any_covered:
            return True, "covered_unmapped_category", "Heuristic found a keyword, but category has no mapped expected theme."
        return False, "unclear", "No mapped expected theme and no keyword match."
    if flags.get(expected_theme, False):
        return True, "roughly_consistent", f"Explanation mentions the expected {expected_theme} keyword family."
    if any_covered:
        matched = ", ".join(name for name, value in flags.items() if value)
        return True, "covered_but_theme_mismatch", f"Matched keyword family differs from expected {expected_theme}: {matched}."
    return False, "unclear_or_missing_expected_theme", f"No keyword from the expected {expected_theme} family was found."


def build_rows(base_rows: list[dict[str, Any]], explanation_rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    explanations_by_id = {str(row.get("id")): row for row in explanation_rows}
    rows: list[dict[str, Any]] = []
    for base in base_rows:
        item_id = str(base.get("id"))
        exp = explanations_by_id.get(item_id, {})
        explanation = str(exp.get("nla_explanation") or exp.get("raw_nla_output") or "")
        flags = {
            "safety": contains_keyword(explanation, KEYWORDS["safety"]),
            "bias": contains_keyword(explanation, KEYWORDS["bias"]),
            "sycophancy": contains_keyword(explanation, KEYWORDS["sycophancy"]),
            "evaluation": contains_keyword(explanation, KEYWORDS["evaluation"]),
        }
        covered, label, note = label_row(str(base.get("category", "")), flags)
        rows.append(
            {
                "id": item_id,
                "category": base.get("category"),
                "prompt": base.get("prompt"),
                "model_answer": base.get("model_answer"),
                "yes_logprob": base.get("yes_logprob"),
                "no_logprob": base.get("no_logprob"),
                "nla_explanation": explanation,
                "mentions_safety_keyword": flags["safety"],
                "mentions_bias_keyword": flags["bias"],
                "mentions_sycophancy_keyword": flags["sycophancy"],
                "mentions_evaluation_keyword": flags["evaluation"],
                "covered": covered,
                "rough_consistency_label": label,
                "manual_note": note,
            }
        )
    return rows


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=FIELDNAMES)
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row.get(field) for field in FIELDNAMES})


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def truncate(text: Any, max_chars: int = 280) -> str:
    value = "" if text is None else str(text)
    value = " ".join(value.split())
    if len(value) <= max_chars:
        return value
    return value[: max_chars - 3] + "..."


def build_report(run_dir: Path, rows: list[dict[str, Any]], report_path: Path) -> None:
    label_counts = Counter(row["rough_consistency_label"] for row in rows)
    coverage_counts = {
        "safety": sum(bool(row["mentions_safety_keyword"]) for row in rows),
        "bias": sum(bool(row["mentions_bias_keyword"]) for row in rows),
        "sycophancy": sum(bool(row["mentions_sycophancy_keyword"]) for row in rows),
        "evaluation": sum(bool(row["mentions_evaluation_keyword"]) for row in rows),
        "covered_any": sum(bool(row["covered"]) for row in rows),
    }
    category_stats: dict[str, Counter[str]] = defaultdict(Counter)
    for row in rows:
        category_stats[str(row.get("category"))][str(row.get("rough_consistency_label"))] += 1

    consistent = [row for row in rows if row["rough_consistency_label"] == "roughly_consistent"][:3]
    unclear = [
        row
        for row in rows
        if row["rough_consistency_label"] in {"unclear", "unclear_or_missing_expected_theme", "covered_but_theme_mismatch"}
    ][:3]

    lines: list[str] = []
    lines.append("# NLA Consistency Smoke Test")
    lines.append("")
    lines.append("This report is a local keyword-heuristic smoke test. It does not call external APIs, does not rerun models, and does not provide a final scientific judgment.")
    lines.append("")
    lines.append(f"- Run directory: `{run_dir}`")
    lines.append(f"- Sample count: `{len(rows)}`")
    lines.append(f"- CSV output: `{run_dir / 'nla_consistency_table.csv'}`")
    lines.append(f"- JSONL output: `{run_dir / 'nla_consistency_table.jsonl'}`")
    lines.append("")
    lines.append("## Keyword Heuristic")
    lines.append("")
    for name, words in KEYWORDS.items():
        lines.append(f"- {name}: {', '.join(words)}")
    lines.append("")
    lines.append("## Heuristic Coverage Counts")
    lines.append("")
    for key, value in coverage_counts.items():
        lines.append(f"- {key}: `{value}`")
    lines.append("")
    lines.append("## Label Counts")
    lines.append("")
    for label, count in sorted(label_counts.items()):
        lines.append(f"- {label}: `{count}`")
    lines.append("")
    lines.append("## Category-Wise Rough Statistics")
    lines.append("")
    for category, counter in sorted(category_stats.items()):
        details = ", ".join(f"{label}={count}" for label, count in sorted(counter.items()))
        lines.append(f"- {category}: {details}")
    lines.append("")
    lines.append("## Three Roughly Consistent Examples")
    lines.append("")
    if consistent:
        for row in consistent:
            lines.append(f"- `{row['id']}` ({row['category']}): {truncate(row['prompt'])}")
            lines.append(f"  - Model answer: {truncate(row['model_answer'])}")
            lines.append(f"  - NLA explanation: {truncate(row['nla_explanation'])}")
            lines.append(f"  - Manual note: {row['manual_note']}")
    else:
        lines.append("- None found by this strict keyword heuristic.")
    lines.append("")
    lines.append("## Three Unclear Or Potentially Inconsistent Examples")
    lines.append("")
    if unclear:
        for row in unclear:
            lines.append(f"- `{row['id']}` ({row['category']}): {truncate(row['prompt'])}")
            lines.append(f"  - Model answer: {truncate(row['model_answer'])}")
            lines.append(f"  - NLA explanation: {truncate(row['nla_explanation'])}")
            lines.append(f"  - Manual note: {row['manual_note']}")
    else:
        lines.append("- None selected by this heuristic.")
    lines.append("")
    lines.append("## Limitations")
    lines.append("")
    lines.append("- Keyword matching can miss semantically correct explanations that use different wording.")
    lines.append("- Keyword matching can over-count generic words that are not faithful explanations.")
    lines.append("- This table does not test causality, activation faithfulness, or whether an explanation is complete.")
    lines.append("- The rough labels should be treated as triage tags for manual review, not as final results.")
    lines.append("")
    lines.append("## Next Steps")
    lines.append("")
    lines.append("- Add a controlled human-judge protocol or an LLM-judge protocol with fixed rubrics and calibration examples.")
    lines.append("- Expand prompts by category and include paraphrase/counterfactual groups.")
    lines.append("- Compare NLA explanations against base model outputs, top tokens, and AR reconstruction scores.")
    lines.append("- Add SR baseline outputs on the same prompt set before making cross-method claims.")
    lines.append("")
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text("\n".join(lines), encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build a minimal NLA consistency table from local smoke-test outputs.")
    parser.add_argument("--outputs-dir", default=str(DEFAULT_NLA_OUTPUTS), help="Directory containing NLA run subdirectories.")
    parser.add_argument("--run-dir", default=None, help="Specific NLA run directory. Defaults to latest complete run.")
    parser.add_argument("--csv-out", default=None, help="CSV output path. Defaults to <run-dir>/nla_consistency_table.csv.")
    parser.add_argument("--jsonl-out", default=None, help="JSONL output path. Defaults to <run-dir>/nla_consistency_table.jsonl.")
    parser.add_argument("--report-out", default=str(DEFAULT_REPORT_PATH), help="Markdown report output path.")
    parser.add_argument("--verbose", action="store_true", help="Enable debug logging.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    configure_logging(args.verbose)
    inputs = resolve_inputs(args)
    logging.info("Using NLA run directory: %s", inputs.run_dir)
    base_rows = read_jsonl(inputs.base_outputs_path)
    explanation_rows = read_jsonl(inputs.explanations_path)
    logging.info("Loaded %d base rows and %d explanation rows", len(base_rows), len(explanation_rows))
    rows = build_rows(base_rows, explanation_rows)
    csv_out = Path(args.csv_out) if args.csv_out else inputs.run_dir / "nla_consistency_table.csv"
    jsonl_out = Path(args.jsonl_out) if args.jsonl_out else inputs.run_dir / "nla_consistency_table.jsonl"
    report_out = Path(args.report_out)
    write_csv(csv_out, rows)
    write_jsonl(jsonl_out, rows)
    build_report(inputs.run_dir, rows, report_out)
    logging.info("Wrote CSV: %s", csv_out)
    logging.info("Wrote JSONL: %s", jsonl_out)
    logging.info("Wrote report: %s", report_out)


if __name__ == "__main__":
    main()
