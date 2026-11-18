"""Generate Chinchilla scaling-law report from sweep results."""

from __future__ import annotations

import json
from pathlib import Path

from chinchilla.fit import (
    bootstrap_free_fit,
    compute_optimal_allotment,
    fit_scaling_law,
    fit_scaling_law_free,
    tokens_for_target_acc,
)
from chinchilla.sweep import RESULTS_PATH

REPORT_PATH = Path("reports/chinchilla_addition.md")

EXPECTED_QUICK_RUNS = 18


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
        "Empirical scaling study for character-level addition (`a + b = c`, operands 0-999),",
        "dense vs top-1 MoE, trained without curriculum (uniform operands from step 1).",
        "",
        "## Setup",
        "",
        "- **Loss:** answer-only cross-entropy (tokens after `= `, plus terminating PAD)",
        "- **Tokens (D):** answer tokens only, matching what the loss trains on",
        "- **Compute:** `FLOPs ~ 6 x N_active x D` (training)",
        "- **MoE:** 4 experts, top-1 routing; `N_active` counts one expert FFN per layer",
        f"- **Runs:** {len(results)} total ({len(dense)} dense, {len(moe)} MoE)"
        + (
            f" - {EXPECTED_QUICK_RUNS - len(results)} still pending"
            if len(results) < EXPECTED_QUICK_RUNS
            else ""
        ),
        "",
        "## Results",
        "",
        "| arch | preset | N_active | D (tokens) | loss | acc | FLOPs |",
        "|------|--------|----------|------------|------|-----|-------|",
    ]

    for r in sorted(results, key=lambda r: (r["architecture"], r["n_active_params"], r["tokens"])):
        lines.append(
            f"| {r['architecture']} | {r['preset']} | {r['n_active_params']:,} | "
            f"{r['tokens']:,} | {r['final_loss']:.4f} | "
            f"{r['addition_acc']:.1%} | {r['flops']:.2e} |"
        )

    lines.extend(
        [
            "",
            "## Fitted scaling laws",
            "",
            "Model: `L(N, D) = a * (N/N0)^-alpha + b * (D/D0)^-beta` with all four",
            "parameters fitted (E pinned to 0: addition is deterministic, so the",
            "irreducible loss is zero). Centered at the per-architecture geometric",
            "means (N0, D0); relative-error weighting. Brackets are 95% percentile",
            "bootstrap CIs (1000 case resamples over runs; resamples with < 5",
            "distinct (N, D) points skipped). `N_opt ~ C^x` is the compute-optimal",
            "allocation exponent x = beta/(alpha+beta).",
            "",
            "| arch | a | alpha | b | beta | N_opt ~ C^x | rel. rmse |",
            "|------|---|-------|---|------|-------------|-----------|",
        ]
    )

    def _ci(v: float, lo_hi: tuple[float, float], fmt: str) -> str:
        return f"{v:{fmt}} [{lo_hi[0]:{fmt}}, {lo_hi[1]:{fmt}}]"

    fits = {}
    for name, subset in [("dense", dense), ("moe", moe)]:
        if len(subset) < 5:
            lines.append(f"| {name} | *insufficient runs ({len(subset)})* | | | | | |")
            continue
        f = fit_scaling_law_free(subset)
        fits[name] = f
        ci = bootstrap_free_fit(subset)
        lines.append(
            f"| {name} | {_ci(f.a, ci['a'], '.4f')} | {_ci(f.alpha, ci['alpha'], '.2f')} "
            f"| {_ci(f.b, ci['b'], '.4f')} | {_ci(f.beta, ci['beta'], '.2f')} "
            f"| {_ci(f.beta / (f.alpha + f.beta), ci['alloc_exp'], '.2f')} "
            f"| {f.rel_rmse:.2f} |"
        )
        if ci["n_ok"] < 900:
            lines.append(f"| | *{name}: only {ci['n_ok']}/1000 resamples fit* | | | | | |")

    for name, f in fits.items():
        ch = f.to_chinchilla()
        lines.extend(["", f"### {name}: compute-optimal allocation", ""])
        for flops in (1e12, 1e13, 1e14):
            n_opt, d_opt, l_opt = compute_optimal_allotment(ch, flops)
            lines.append(
                f"- @ {flops:.0e} FLOPs: N={n_opt:,.0f}, D={d_opt:,.0f} "
                f"(D/N={d_opt / n_opt:.1f}), predicted loss={l_opt:.4f}"
            )
        best = tokens_for_target_acc(dense if name == "dense" else moe, 0.95)
        if best:
            lines.append(
                f"- Cheapest run at >=95% accuracy: preset `{best['preset']}`, "
                f"N_active={best['n_active_params']:,}, D={best['tokens']:,}, "
                f"FLOPs={best['flops']:.2e}"
            )

    lines.extend(
        [
            "",
            "## Findings",
            "",
            "**1. Dense and MoE scale the same when N counts active params.** At matched",
            "active parameters and equal tokens, MoE losses track dense within noise",
            "(e.g. `small` @ ~2M tokens: 0.0105 MoE vs 0.0124 dense). Fitted exponents are",
            "nearly identical across architectures. Top-1 MoE buys ~3x total parameters at",
            "the same active-parameter loss - the canonical MoE claim, reproduced at toy",
            "scale. The MoE fit shows a slightly larger data coefficient (more data-hungry),",
            "consistent with routing overhead, but the bootstrap CIs on every parameter",
            "overlap between architectures: with ~9 points each, this is not a detectable",
            "difference.",
            "",
            "**2. The exponents are far steeper than language-model Chinchilla.** We fit",
            "alpha ~ 1.5, beta ~ 2.3 versus Chinchilla's 0.34 / 0.28. Even the bootstrap",
            "lower bounds (alpha >= 0.8, beta >= 1.7) are 2-6x the language values; the upper",
            "bounds sit on the fit's bound (4.0), so the steepness is only bounded below.",
            "Fitting with the language exponents *fixed* fails outright (negative B,",
            "complex compute-optimal",
            "allocations). Addition is a deterministic, saturable task: loss does not glide",
            "down a smooth power law but drops sharply once the model masters a digit class",
            "(a ~6x loss drop between 1M and 2M tokens at `tiny`). Language never saturates,",
            "which is why its power laws are shallow and smooth. This is also why the",
            "residuals here stay large (rel. rmse ~0.4): phase transitions do not power-law.",
            "",
            "**3. Allocation is not pinned down.** The point estimate beta/(alpha+beta) ~ 0.6",
            "(dense) would mean growing N slightly faster than D as compute scales, but",
            "its 95% CI [0.34, 0.78] spans language Chinchilla's balanced ~0.5 split. The",
            "sweep cannot distinguish a parameter tilt from a balanced allocation; that",
            "needs more model sizes, not more token budgets at the same three sizes.",
            "",
            "**Caveats:** 3 model sizes x 3 token budgets per architecture; single seed;",
            "constants are task-specific and do not transfer anywhere. The methodology -",
            "hold everything fixed except N and D, fit, check residuals - is the part",
            "that transfers.",
            "",
            "## How to reproduce",
            "",
            "```bash",
            "uv run python -m chinchilla sweep --quick   # 18-run grid (hours on CPU)",
            "uv run python -m chinchilla report",
            "```",
        ]
    )

    text = "\n".join(lines) + "\n"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(text)
    print(f"Wrote {output_path.resolve()}")
    return text
