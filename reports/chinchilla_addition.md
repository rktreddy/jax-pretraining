# Chinchilla Scaling Laws - Addition Transformer

Empirical scaling study for character-level addition (`a + b = c`, operands 0-999),
dense vs top-1 MoE, trained without curriculum (uniform operands from step 1).

## Setup

- **Loss:** answer-only cross-entropy (tokens after `= `, plus terminating PAD)
- **Tokens (D):** answer tokens only, matching what the loss trains on
- **Compute:** `FLOPs ~ 6 x N_active x D` (training)
- **MoE:** 4 experts, top-1 routing; `N_active` counts one expert FFN per layer
- **Runs:** 17 total (9 dense, 8 MoE) - 1 still pending

## Results

| arch | preset | N_active | D (tokens) | loss | acc | FLOPs |
|------|--------|----------|------------|------|-----|-------|
| dense | tiny | 102,656 | 495,683 | 1.1806 | 0.0% | 3.05e+11 |
| dense | tiny | 102,656 | 998,502 | 0.8667 | 4.7% | 6.15e+11 |
| dense | tiny | 102,656 | 1,992,269 | 0.1352 | 89.1% | 1.23e+12 |
| dense | small | 797,440 | 495,683 | 0.1855 | 87.5% | 2.37e+12 |
| dense | small | 797,440 | 998,502 | 0.0464 | 96.9% | 4.78e+12 |
| dense | small | 797,440 | 1,992,269 | 0.0124 | 98.4% | 9.53e+12 |
| dense | medium | 3,635,968 | 495,683 | 0.1116 | 90.6% | 1.08e+13 |
| dense | medium | 3,635,968 | 998,502 | 0.0298 | 95.3% | 2.18e+13 |
| dense | medium | 3,635,968 | 1,992,269 | 0.0058 | 98.4% | 4.35e+13 |
| moe | tiny | 105,096 | 495,683 | 1.1681 | 0.0% | 3.13e+11 |
| moe | tiny | 105,096 | 998,502 | 0.4471 | 34.4% | 6.30e+11 |
| moe | tiny | 105,096 | 1,992,269 | 0.1019 | 84.4% | 1.26e+12 |
| moe | small | 807,184 | 495,683 | 0.1810 | 87.5% | 2.40e+12 |
| moe | small | 807,184 | 998,502 | 0.0409 | 93.8% | 4.84e+12 |
| moe | small | 807,184 | 1,992,269 | 0.0105 | 98.4% | 9.65e+12 |
| moe | medium | 3,661,528 | 495,683 | 0.1449 | 90.6% | 1.09e+13 |
| moe | medium | 3,661,528 | 998,502 | 0.0316 | 95.3% | 2.19e+13 |

## Fitted scaling laws

Model: `L(N, D) = a * (N/N0)^-alpha + b * (D/D0)^-beta` with all four
parameters fitted (E pinned to 0: addition is deterministic, so the
irreducible loss is zero). Centered at the per-architecture geometric
means (N0, D0); relative-error weighting.

| arch | a | alpha | b | beta | rel. rmse |
|------|---|-------|---|------|-----------|
| dense | 0.0097 | 1.49 | 0.0279 | 2.33 | 0.39 |
| moe | 0.0076 | 1.66 | 0.0409 | 2.31 | 0.37 |

### dense: compute-optimal allocation

- @ 1e+12 FLOPs: N=194,105, D=858,643 (D/N=4.4), predicted loss=0.1007
- @ 1e+13 FLOPs: N=789,712, D=2,110,475 (D/N=2.7), predicted loss=0.0124
- @ 1e+14 FLOPs: N=3,212,929, D=5,187,374 (D/N=1.6), predicted loss=0.0015
- Cheapest run at >=95% accuracy: preset `small`, N_active=797,440, D=998,502, FLOPs=4.78e+12

### moe: compute-optimal allocation

- @ 1e+12 FLOPs: N=173,902, D=958,394 (D/N=5.5), predicted loss=0.0874
- @ 1e+13 FLOPs: N=664,316, D=2,508,846 (D/N=3.8), predicted loss=0.0095
- @ 1e+14 FLOPs: N=2,537,725, D=6,567,563 (D/N=2.6), predicted loss=0.0010
- Cheapest run at >=95% accuracy: preset `medium`, N_active=3,661,528, D=998,502, FLOPs=2.19e+13

## Findings

**1. Dense and MoE scale the same when N counts active params.** At matched
active parameters and equal tokens, MoE losses track dense within noise
(e.g. `small` @ ~2M tokens: 0.0105 MoE vs 0.0124 dense). Fitted exponents are
nearly identical across architectures. Top-1 MoE buys ~3x total parameters at
the same active-parameter loss - the canonical MoE claim, reproduced at toy
scale. The MoE fit shows a slightly larger data coefficient (more data-hungry),
consistent with routing overhead, but with ~9 points per architecture this is
suggestive rather than conclusive.

**2. The exponents are 5-8x steeper than language-model Chinchilla.** We fit
alpha ~ 1.5, beta ~ 2.3 versus Chinchilla's 0.34 / 0.28. Fitting with the
language exponents *fixed* fails outright (negative B, complex compute-optimal
allocations). Addition is a deterministic, saturable task: loss does not glide
down a smooth power law but drops sharply once the model masters a digit class
(a ~6x loss drop between 1M and 2M tokens at `tiny`). Language never saturates,
which is why its power laws are shallow and smooth. This is also why the
residuals here stay large (rel. rmse ~0.4): phase transitions do not power-law.

**3. Allocation tilts toward parameters.** beta/(alpha+beta) ~ 0.6, i.e. grow N
slightly faster than D as compute scales - unlike language Chinchilla's
balanced ~0.5 split (D/N ~ 20 at LLM scale). Small models are the bottleneck
for this task; data is cheap (synthetic and infinite).

**Caveats:** 3 model sizes x 3 token budgets per architecture; single seed;
constants are task-specific and do not transfer anywhere. The methodology -
hold everything fixed except N and D, fit, check residuals - is the part
that transfers.

## How to reproduce

```bash
uv run python -m chinchilla sweep --quick   # 18-run grid (hours on CPU)
uv run python -m chinchilla report
```
