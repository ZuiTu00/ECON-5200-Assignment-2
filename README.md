# ECON-5200-Assignment-2
"""
Audit 02: Deconstructing Statistical Lies
==========================================
Topics: Latency Skew, False Positives, Sample Ratio Mismatch, Survivorship Bias
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

np.random.seed(42)

# ============================================================
# Phase 1: The Robustness Audit — Latency Skew
# ============================================================

# Step 1.1 - Simulate skewed latency data
normal_traffic = np.random.randint(20, 50, 980)
spike_traffic = np.random.randint(1000, 5000, 20)
latency_logs = np.concatenate([normal_traffic, spike_traffic])

# Step 1.2 - Manual MAD vs SD
def calculate_mad(data):
    median = np.median(data)
    abs_devs = np.abs(data - median)
    return np.median(abs_devs)

mad = calculate_mad(latency_logs)
sd = np.std(latency_logs)
mean = np.mean(latency_logs)
median = np.median(latency_logs)

print("=" * 60)
print("PHASE 1: Latency Skew — Robust Statistics")
print("=" * 60)
print(f"  Mean:   {mean:.2f} ms")
print(f"  Median: {median:.2f} ms")
print(f"  SD:     {sd:.2f} ms")
print(f"  MAD:    {mad:.2f} ms")
print(f"\n  --> SD is {sd/mad:.1f}x larger than MAD due to outliers.\n")


# ============================================================
# Phase 2: The Probability Audit — False Positive Paradox
# ============================================================

def bayesian_audit(prior, sensitivity, specificity):
    p_flagged = sensitivity * prior + (1 - specificity) * (1 - prior)
    posterior = (sensitivity * prior) / p_flagged
    return posterior

scenarios = {
    "A (Bootcamp, 50%)":  0.50,
    "B (Econ Class, 5%)": 0.05,
    "C (Honors, 0.1%)":   0.001,
}

print("=" * 60)
print("PHASE 2: False Positive Paradox — Bayesian Audit")
print("=" * 60)
for name, base_rate in scenarios.items():
    p = bayesian_audit(prior=base_rate, sensitivity=0.98, specificity=0.98)
    print(f"  Scenario {name}: P(Cheater|Flagged) = {p*100:.2f}%")
print()


# ============================================================
# Phase 3: The Bias Audit — Sample Ratio Mismatch (SRM)
# ============================================================

observed = np.array([50250, 49750])
expected = np.array([50000, 50000])
chi_sq = np.sum((observed - expected)**2 / expected)

print("=" * 60)
print("PHASE 3: Sample Ratio Mismatch — Chi-Square Test")
print("=" * 60)
print(f"  Observed: Control={observed[0]}, Treatment={observed[1]}")
print(f"  Expected: {expected[0]} each")
print(f"  Chi-Square statistic: {chi_sq:.2f}")
print(f"  Threshold (df=1, p<0.05): 3.84")
if chi_sq > 3.84:
    print("  --> INVALID: Sample Ratio Mismatch detected!\n")
else:
    print("  --> No significant mismatch.\n")


# ============================================================
# Phase 4: Survivorship Bias — Crypto Market Simulation
# ============================================================

n = 10000
peak_mcap = (np.random.pareto(a=1.5, size=n) + 1) * 1000

df_all = pd.DataFrame({"Peak_Market_Cap": peak_mcap})
threshold = np.percentile(peak_mcap, 99)
df_survivors = df_all[df_all["Peak_Market_Cap"] >= threshold].copy()

print("=" * 60)
print("PHASE 4: Survivorship Bias — Crypto Token Simulation")
print("=" * 60)
print(f"  All Tokens (Graveyard) — Count: {len(df_all)}, Mean: ${df_all['Peak_Market_Cap'].mean():,.2f}")
print(f"  Survivors (Top 1%)    — Count: {len(df_survivors)}, Mean: ${df_survivors['Peak_Market_Cap'].mean():,.2f}")
print(f"  Bias Multiplier: {df_survivors['Peak_Market_Cap'].mean() / df_all['Peak_Market_Cap'].mean():.1f}x\n")


# ============================================================
# Visualization
# ============================================================

fig, axes = plt.subplots(2, 2, figsize=(14, 10))
fig.suptitle("Audit 02: Deconstructing Statistical Lies", fontsize=16, fontweight="bold")

# 1 — Latency Distribution
axes[0, 0].hist(latency_logs, bins=80, color="steelblue", edgecolor="black")
axes[0, 0].axvline(mean, color="red", linestyle="--", label=f"Mean={mean:.0f}")
axes[0, 0].axvline(median, color="lime", linestyle="--", label=f"Median={median:.0f}")
axes[0, 0].set_title("Phase 1: Latency Distribution (Mean vs Median)")
axes[0, 0].set_xlabel("Latency (ms)")
axes[0, 0].legend()

# 2 — Bayesian Posterior by Base Rate
rates = [0.001, 0.01, 0.05, 0.1, 0.2, 0.5]
posteriors = [bayesian_audit(r, 0.98, 0.98) * 100 for r in rates]
axes[0, 1].bar([f"{r*100}%" for r in rates], posteriors, color="coral", edgecolor="black")
axes[0, 1].set_title("Phase 2: P(Cheater|Flagged) by Base Rate")
axes[0, 1].set_ylabel("Posterior Probability (%)")
axes[0, 1].set_xlabel("Base Rate")

# 3 — SRM: Observed vs Expected
x = ["Control", "Treatment"]
axes[1, 0].bar(x, observed, color=["#4CAF50", "#F44336"], edgecolor="black")
axes[1, 0].axhline(50000, color="black", linestyle="--", label="Expected (50,000)")
axes[1, 0].set_title(f"Phase 3: SRM Check (Chi-Sq={chi_sq:.2f})")
axes[1, 0].set_ylabel("User Count")
axes[1, 0].legend()

# 4 — Survivorship Bias
axes[1, 1].hist(df_all["Peak_Market_Cap"], bins=100, color="crimson", edgecolor="black", alpha=0.7, label="All Tokens")
axes[1, 1].hist(df_survivors["Peak_Market_Cap"], bins=30, color="gold", edgecolor="black", alpha=0.9, label="Top 1% Survivors")
axes[1, 1].set_title("Phase 4: Survivorship Bias (Crypto)")
axes[1, 1].set_xlabel("Peak Market Cap ($)")
axes[1, 1].set_xlim(0, np.percentile(peak_mcap, 99.9))
axes[1, 1].legend()

plt.tight_layout()
plt.savefig("audit_02_results.png", dpi=150)
plt.show()

print("Chart saved as audit_02_results.png")
