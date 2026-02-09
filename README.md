# ECON-5200-Assignment-2
# Audit 02: Deconstructing Statistical Lies

## Overview
This audit investigates three common ways data can mislead decision-makers: **Latency Skew**, **False Positives**, and **Survivorship Bias**. Each section uses simulated data and manual calculations to expose how summary statistics and naive accuracy claims can hide critical truths.

---

## Phase 1 — Latency Skew (Robust Statistics)

**Scenario:** A cloud provider "NebulaCloud" advertises a mean latency of 35ms. We simulated 1,000 requests — 980 normal (20–50ms) and 20 spike requests (1,000–5,000ms) — to test this claim.

**Key Findings:**
- The **Mean** was inflated to ~90ms due to 20 extreme outliers, making it an unreliable summary.
- The **Median** remained stable around ~35ms, reflecting the typical user experience.
- **Standard Deviation** exploded because it squares deviations, amplifying outlier influence.
- **MAD (Median Absolute Deviation)**, computed manually with NumPy, stayed low and stable — proving its robustness against skewed tails.

**Takeaway:** In Pareto-distributed systems, always prefer median and MAD over mean and SD.

---

## Phase 2 — The False Positive Paradox (Bayesian Reasoning)

**Scenario:** "IntegrityAI" claims a 98% accurate plagiarism detector (Sensitivity = 98%, Specificity = 98%). We applied Bayes' Theorem across three base-rate environments.

**Key Findings:**

| Scenario | Base Rate | P(Cheater \| Flagged) |
|---|---|---|
| A — Bootcamp | 50% | 98.00% |
| B — Econ Class | 5% | 72.06% |
| C — Honors Seminar | 0.1% | **4.67%** |

- At a 0.1% base rate, over **95% of flagged students are innocent**.
- High accuracy is meaningless without considering the prevalence of the event.

**Takeaway:** Never evaluate a classifier without accounting for the base rate. A "98% accurate" system can still be wrong most of the time in low-prevalence settings.

---

## Phase 3 — Sample Ratio Mismatch (Chi-Square Audit)

**Scenario:** "FinFlash" ran a 50/50 A/B test with 100,000 users but observed 50,250 vs 49,750.

**Key Findings:**
- Manual Chi-Square statistic = **5.00**, exceeding the 3.84 threshold (p < 0.05).
- The 500-user gap is statistically significant — not random noise.
- Likely cause: an engineering bug (e.g., app crash) causing attrition in the treatment group.

**Takeaway:** Always run an SRM check before trusting A/B test results.

---

## Phase 4 — Survivorship Bias (Crypto Market Simulation)

**Scenario:** Simulated 10,000 crypto token launches using a Pareto distribution where 99% of tokens have near-zero market cap.

**Key Findings (alpha=1.0, scale=$100,000):**

| Metric | All Tokens (Graveyard) | Survivors (Top 1%) |
|---|---|---|
| Count | 10,000 | 100 |
| Mean | $796,054.40 | $35,955,574.98 |
| Median | $197,055.45 | $18,849,568.20 |

- **Survivorship Bias Multiplier: 45.2x** — the Top 1% mean is 45 times the overall mean.
- The vast majority of tokens cluster near the minimum, while a tiny fraction explode in value.
- Using alpha=1.0 (the most extreme Pareto distribution) closely mirrors real crypto markets like Pump.fun, where 98.6% of tokens fail.

**Takeaway:** Crypto success stories represent extreme outliers. Evaluating the market based only on visible winners is textbook survivorship bias. The 45.2x bias multiplier shows that "Listed Coins" paint a wildly misleading picture of the actual market.

---

## Tools & Methods
- **Python**, **NumPy**, **Pandas**, **Matplotlib**
- Manual implementations (no scipy.stats for MAD; manual Chi-Square and Bayes)
- Pareto / Power-Law simulation for realistic skewed data

## Author
Financial Data Science Audit — Topic 4, 5, 6
