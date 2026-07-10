"""Generate Chinchilla scaling-law report from sweep results."""

from __future__ import annotations

import json
from pathlib import Path

from chinchilla.fit import(
    compute_optimal_allotment,
    fit_scaling_law,
    tokens_for_target_acc
)
from chinchilla.sweep import RESULTS_PATH


REPORT_PATH = Path("reports/chinchilla_addition.md")


def generate_report(
        results_path: Path = RESULTS_PATH,
        output_path: Path = REPORT_PATH,
) -> str:
    results: list[dict] = json.loads(results_path.read_text())
    dense = [r for r in results if r["architecture"] == "dense"]
    moe = [r for r in results if r["architecture"] == "moe"]

    lines = [
        "# Chinchilla Scaling Laws - Addition Transformer",
        "",
        "Empirical scaling study for charactor-level addition (`a + b = c`, operands 0-999).",
        "",
        "## Setup",
        "",
        "- **Loss:** answer-only cross-entropy (tokens after `= `)",
        "- **Law:** `:(N, D) = E + A/N^a + B/D^b with a = 0.34, b = 0.28 (Chinchilla defaults)",
        "- **Compute:** `FLOPs ~ 6 x N_active x D` (training)",
        "- **MoE:** 4 experts, top-1 routing; `N_activr` counts one expert FFN per layer"
        "",
        f"- **Runs:** {len(results)} total ({len(dense)} dense, {len(moe)} MoE)",
        "",
        "## Results",
        "",
        "| arch | preset | N | N_active | D (tokens) | loss | acc | FLOPs |",
        "|------|--------|---|----------|------------|------|-----|-------|",
    ]

    for r in sorted(results, key=lambda r: (r["architecture"], r["preset"], r["tokens"])):
        lines.append(
            f"| {r['architecture']:,} | {r['preset']:}, | {r['tokens']:,} | "
            f"{r['n_active_params']:,} | {r['tokens']:,}, | {r['final_loss']:.4f} | "
            f"{r['addition_acc']:.1%} | {r['flops']:.2e}  |"
        )

    lines.extend(["", "## Fitted scaling laws", ""])

    for name, subset in [("Dense", dense), ("MoE (active params)", moe)]:
        if len(subset) < 3:
            lines.append(f"*{name}: insufficient runs ({len(subset)})*\n")
            continue
        fit = fit_scaling_law(subset)
        lines.extend([
            f"### {name}",
            "",
            f"- `E = {fit.E:.4f:.4f}`, `A = {fit.A:.2e}`, `B = {fit.B:.2e}`",
            f"- RMSE = {fit.rmse:.4f}",
            "",
        ])

        target_loss = 0.05
        n99 = tokens_for_target_acc(subset, 0.99)
        if n99:
            lines.extend([
                f"**99% accuracy (best run):** preset `{n99['preset']}`, "
                f"N={n99['n_params']:,}, D={n99['tokens']:,}, "
                f"FLOPs={n99['flops']:.2e}",
                "",
            ])

        for flops in (1e12, 1e13, 1e14):
            n_opt, d_opt, l_opt = compute_optimal_allotment(fit, flops)
            lines.append(
                f"- Compute-optimal @ {flops:.0e} FLOPs: " 
                f"N={n_opt:,.0f}, D={d_opt:,.0f}, predicted loss={l_opt:.4f}"
            )
        lines.append("")

    lines.extend([
        "## Dense vs MoE",
        "",
        "MoE has **heigher total N** but similar **active N** per token. Under the same",
        "active-parameter budget, MoE should match dense FLOPs/token;; differences in",
        "fitted `A` reflect routing overhead and under-trained inactive experts.",
        "",
        "## How to reproduce",
        "",
        "```bash",
        "uv run -m chinchilla sweep             # full grid (slow on CPU)",
        "uv run -m chinchilla sweep --quick     # 18-run subset",
        "uv run -m chinchilla report",
        "```",
    ])

    text = "\n".join(lines) + "\n"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(text)
    print(f"Wrote {output_path.resolve()}")
    return text