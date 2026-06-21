# Threshold test — ASR_rel dose-response (Phase 2b, unmatched)

One-sample test of ASR_rel vs 0 at each poison rate; demonstrates inert (≤10%) → install (≥50%) rather than a single linear slope.

| poison rate | n | mean ASR_rel | 95% CI | p (vs 0) | significant? |
|---|---|---|---|---|---|
| 0 | 3 | +0.000 | [+0.000, +0.000] | — | no |
| 0.005 | 3 | +0.046 | [-0.056, +0.149] | 0.193 | no |
| 0.01 | 3 | +0.031 | [-0.016, +0.079] | 0.106 | no |
| 0.02 | 3 | +0.000 | [-0.093, +0.093] | 0.998 | no |
| 0.05 | 3 | +0.075 | [-0.032, +0.181] | 0.095 | no |
| 0.1 | 3 | +0.072 | [+0.039, +0.104] | 0.011 | **yes** |
| 0.5 | 1 | +0.098 | [+nan, +nan] | — | no |
| 0.75 | 3 | +0.333 | [+0.176, +0.490] | 0.012 | **yes** |
| 0.9 | 3 | +0.527 | [+0.342, +0.711] | 0.007 | **yes** |
| 1 | 3 | +0.498 | [+0.325, +0.670] | 0.006 | **yes** |

**Convexity (vs Kulkarni's linear ν):** quadratic term = +0.375, F(1,25) = 4.83, p = 0.0375 (n=28 points). A positive quadratic term with small p means the dose-response is **accelerating/convex**, not the single linear slope ν.

**Operational install point:** lowest poison rate whose mean ASR_rel clears the install gate (0.2) = **0.75**. Low rates (≤0.1) stay below this gate (mean ASR_rel ≤ 0.075) even where a small statistical effect is detectable at 0.1.

_Honest interpretation: the dose-response is convex/accelerating and does not reach operational installation (ASR_rel ≥ 0.2) until poison > 0.5 — a different curve SHAPE than Kulkarni et al.'s linear vulnerability slope. We frame the 'threshold' OPERATIONALLY (install gate), not as a hard statistical floor: a small, sub-operational effect is already detectable at 10% poison (mean 0.072, p=0.011), so we do not claim the attack is statistically inert below 10%._
