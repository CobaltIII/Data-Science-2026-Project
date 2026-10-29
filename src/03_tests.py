"""Step 3: Run the four hypothesis tests on the confirmation set.

Uses the statistical tests covered in CSE558 Lectures 5-8.
The same population filter as the EDA is used here.

Output: results/tests.json
Run: python src/03_tests.py
"""

import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

ROOT = Path(__file__).resolve().parents[1]
ALPHA = 0.05
N_TESTS = 4
MIN_PER_GROUP = 5  #to ensure sufficient sample sizes (at least 5 Black and White applicants) within matched analysis cells

df = pd.read_parquet(ROOT / "data" / "processed" / "hmda_clean.parquet")
split_data = pd.read_parquet(ROOT / "data" / "processed" / "split.parquet")
df = df.merge(split_data, on="row_id")

eda = json.loads((ROOT / "results" / "eda_summary.json").read_text())
q = eda["income_quintile_cuts_000s"]  # Income cutoffs from the exploration set


def population(d):
    """Keeps eligible home loan applications for the analysis."""
    return d[
        (d["decision"] == 1)
        & (d["owner_occupancy_name"] == "Owner-occupied as a principal dwelling")
        & (d["property_type_name"].str.startswith("One-to-four"))
        & (d["lien_status_name"] == "Secured by a first lien")
        & (d["race_group"] != "Not applicable (non-natural person)")
    ]

cf = population(df[df["split"] == "confirmation"]).copy()

cf["inc_band"] = (
    pd.cut(
        cf["applicant_income_000s"],
        [-np.inf] + q + [np.inf],
        labels=["Q1", "Q2", "Q3", "Q4", "Q5"]
    )
    .astype(object)
    .fillna("NR")
)

R = {
    "n_population": int(len(cf)),
    "denial_rate": float(cf["denied"].mean()),
    "n_by_group": cf["race_group"].value_counts().to_dict()
}


def two_prop_z(x1, n1, x2, n2):
    """Calculates the two-proportion z statistic and 95% confidence interval."""
    p1 = x1 / n1
    p2 = x2 / n2

    p = (x1 + x2) / (n1 + n2)
    z = (p1 - p2) / np.sqrt(p * (1 - p) * (1 / n1 + 1 / n2))

    se = np.sqrt(
        p1 * (1 - p1) / n1
        + p2 * (1 - p2) / n2
    )

    ci = (
        p1 - p2 - 1.96 * se,
        p1 - p2 + 1.96 * se
    )

    return p1, p2, z, ci

# Test 1: Paired t-test for Black and White denial rates
bw = cf[cf["race_group"].isin(["Black", "White"])]
cells = []

for key, group in bw.groupby(
    ["lender", "loan_purpose_name", "loan_type_name", "inc_band"]
):
    black = group[group["race_group"] == "Black"]
    white = group[group["race_group"] == "White"]

    if len(black) >= MIN_PER_GROUP and len(white) >= MIN_PER_GROUP:
        cells.append({
            "rate_black": black["denied"].mean(),
            "rate_white": white["denied"].mean(),
            "n_black": len(black),
            "n_white": len(white)
        })

cells = pd.DataFrame(cells)

diff = cells["rate_black"] - cells["rate_white"]
n = len(diff)

t_stat = diff.mean() / (
    diff.std(ddof=1) / np.sqrt(n)
)
t_crit = stats.t.ppf(0.975, n - 1)

T1 = {
    "cells": n,
    "black_in_cells": int(cells["n_black"].sum()),
    "white_in_cells": int(cells["n_white"].sum()),
    "black_total": int((bw["race_group"] == "Black").sum()),
    "white_total": int((bw["race_group"] == "White").sum()),
    "mean_rate_black": float(cells["rate_black"].mean()),
    "mean_rate_white": float(cells["rate_white"].mean()),
    "estimate": float(diff.mean()),
    "sd_d": float(diff.std(ddof=1)),
    "ci95": [
        float(diff.mean() - t_crit * diff.std(ddof=1) / np.sqrt(n)),
        float(diff.mean() + t_crit * diff.std(ddof=1) / np.sqrt(n))
    ],
    "t": float(t_stat),
    "df": n - 1,
    "t_crit_one_sided": float(stats.t.ppf(1 - ALPHA, n - 1)),
    "p": float(stats.t.sf(t_stat, n - 1)),
    "share_cells_black_higher": float((diff > 0).mean())
}

#compare with a z-test that does not match applicants by cell
black_rows = bw[bw["race_group"] == "Black"]
white_rows = bw[bw["race_group"] == "White"]

xb = int(black_rows["denied"].sum())
nb = len(black_rows)
xw = int(white_rows["denied"].sum())
nw = len(white_rows)

p1, p2, z, ci = two_prop_z(xb, nb, xw, nw)

T1["unmatched_z"] = float(z)
T1["unmatched_diff"] = float(p1 - p2)
T1["unmatched_ci"] = [float(value) for value in ci]

#compare with an unpaired welch t-test on the cell rates
welch = stats.ttest_ind(
    cells["rate_black"],
    cells["rate_white"],
    equal_var=False,
    alternative="greater"
)

T1["unpaired_welch_t"] = float(welch.statistic)
T1["unpaired_welch_p"] = float(welch.pvalue)

#compare variances of the two sets of cell rates
var_ratio = (
    cells["rate_black"].var(ddof=1)
    / cells["rate_white"].var(ddof=1)
)

T1["F_var"] = float(var_ratio)
T1["F_p_two_sided"] = float(
    2 * min(
        stats.f.sf(var_ratio, n - 1, n - 1),
        stats.f.cdf(var_ratio, n - 1, n - 1)
    )
)

#correlation b/w black and white denial rates within cells
T1["corr_black_white_rates"] = float(
    np.corrcoef(cells["rate_black"], cells["rate_white"])[0, 1]
)

#check w diff min cell sizes
sensitivity = {}

for min_size in (3, 10):
    cell_diff = []

    for _, group in bw.groupby(
        ["lender", "loan_purpose_name", "loan_type_name", "inc_band"]
    ):
        black = group[group["race_group"] == "Black"]
        white = group[group["race_group"] == "White"]

        if len(black) >= min_size and len(white) >= min_size:
            difference = black["denied"].mean() - white["denied"].mean()
            cell_diff.append(difference)

    cell_diff = np.array(cell_diff)

    sensitivity[min_size] = {
        "cells": len(cell_diff),
        "mean": float(cell_diff.mean()),
        "t": float(
            cell_diff.mean()
            / (cell_diff.std(ddof=1) / np.sqrt(len(cell_diff)))
        )
    }

T1["sensitivity_min_cell"] = sensitivity
R["test1"] = T1

# Test 2: Chi-square test for racial group and denial outcome
# Using conventional home-purchase loans in income group Q3

groups = ["White", "Black", "Hispanic", "Asian", "Not provided"]

selected = cf[
    (cf["loan_type_name"] == "Conventional")
    & (cf["loan_purpose_name"] == "Home purchase")
    & (cf["inc_band"] == "Q3")
    & cf["race_group"].isin(groups)
]

table = pd.crosstab(selected["race_group"], selected["denied"])
table = table.loc[groups]

chi2, p_value, dof, expected = stats.chi2_contingency(
    table,
    correction=False
)

denial_rates = table[1.0] / table.sum(axis=1)

T2 = {
    "n": int(table.values.sum()),
    "table": {
        group: [int(value) for value in table.loc[group]]
        for group in groups
    },
    "denial_rate": denial_rates.round(4).to_dict(),
    "chi2": float(chi2),
    "df": int(dof),
    "p": float(p_value),
    "chi2_crit": float(stats.chi2.ppf(1 - ALPHA, dof)),
    "min_expected": float(expected.min()),
    "max_gap_vs_white_pp": float(
        (denial_rates.drop("White") - denial_rates["White"]).max() * 100
    )
}

#compare black nd white using 2 proportion z-test
black = selected[selected["race_group"] == "Black"]["denied"]
white = selected[selected["race_group"] == "White"]["denied"]

p1, p2, z, ci = two_prop_z(
    black.sum(), len(black),
    white.sum(), len(white)
)

T2["bw_z"] = float(z)
T2["bw_diff"] = float(p1 - p2)
T2["bw_ci"] = [float(value) for value in ci]

R["test2"] = T2

# Test 3: Chi-square test for denial reason and racial group

# Keep denied applications from the four groups, excluding missing or irrelevant reasons
denied_apps = cf[
    (cf["denied"] == 1)
    & cf["race_group"].isin(["White", "Black", "Hispanic", "Asian"])
    & ~cf["denial_reason_name_1"].isin(["Not reported", "Not denied"])
].copy()

# Combine the three rare reasons into one category
rare_reasons = [
    "Employment history",
    "Mortgage insurance denied",
    "Unverifiable information"
]

denied_apps["reason"] = denied_apps["denial_reason_name_1"].replace(
    {reason: "Other" for reason in rare_reasons}
)

# Create a table of denial reasons by racial group
table = pd.crosstab(
    denied_apps["reason"],
    denied_apps["race_group"]
)[["White", "Black", "Hispanic", "Asian"]]

chi2, p_value, dof, expected = stats.chi2_contingency(
    table,
    correction=False
)

# Calculate each reason's percentage within each racial group
reason_share = table / table.sum() * 100

T3 = {
    "n": int(table.values.sum()),
    "table": table.to_dict(),
    "chi2": float(chi2),
    "df": int(dof),
    "p": float(p_value),
    "chi2_crit": float(stats.chi2.ppf(1 - ALPHA, dof)),
    "min_expected": float(expected.min()),
    "share_pct": reason_share.round(2).to_dict(),
    "contribution": ((table - expected) ** 2 / expected).round(2).to_dict(),
    "max_gap_vs_white_pp": float(
        reason_share.drop(columns="White")
        .sub(reason_share["White"], axis=0)
        .abs()
        .max()
        .max()
    )
}

# Compare the credit history reason for Black and White applicants
black_count = int(table.loc["Credit history", "Black"])
black_total = int(table["Black"].sum())

white_count = int(table.loc["Credit history", "White"])
white_total = int(table["White"].sum())

p1, p2, z, ci = two_prop_z(
    black_count, black_total,
    white_count, white_total
)

T3["credit_history_bw"] = {
    "black": float(p1),
    "white": float(p2),
    "z": float(z),
    "ci": [float(value) for value in ci]
}

R["test3"] = T3

# Test 4: Compare higher-priced FHA loans for Black and White borrowers
fha_loans = cf[
    (cf["action_taken_name"] == "Loan originated")
    & (cf["loan_type_name"] == "FHA-insured")
    & cf["race_group"].isin(["Black", "White"])
]

# Count higher-priced loans and total loans for each group
black_loans = fha_loans[fha_loans["race_group"] == "Black"]
white_loans = fha_loans[fha_loans["race_group"] == "White"]

xb4 = int(black_loans["higher_priced"].sum())
nb4 = int(len(black_loans))

xw4 = int(white_loans["higher_priced"].sum())
nw4 = int(len(white_loans))

p1, p2, z, ci = two_prop_z(xb4, nb4, xw4, nw4)

T4 = {
    "n_black": nb4,
    "n_white": nw4,
    "hp_black": float(p1),
    "hp_white": float(p2),
    "estimate": float(p1 - p2),
    "ci95": [float(value) for value in ci],
    "z": float(z),
    "z_crit": float(stats.norm.ppf(1 - ALPHA)),
    "p": float(stats.norm.sf(z))
}

# Alternative test: chi-square test on the same 2x2 table
chi_result = stats.chi2_contingency(
    [
        [xb4, nb4 - xb4],
        [xw4, nw4 - xw4]
    ],
    correction=False
)

T4["chi2_2x2"] = float(chi_result[0])
T4["chi2_p"] = float(chi_result[1])

# Welch's t-test using the 0/1 higher-priced loan outcome
welch_result = stats.ttest_ind(
    black_loans["higher_priced"],
    white_loans["higher_priced"],
    equal_var=False,
    alternative="greater"
)

T4["welch_t"] = float(welch_result.statistic)
T4["welch_p"] = float(welch_result.pvalue)

# Check the difference separately within each income band
T4["by_band"] = {}

for band, group in fha_loans.groupby("inc_band"):
    black = group[group["race_group"] == "Black"]["higher_priced"]
    white = group[group["race_group"] == "White"]["higher_priced"]

    if len(black) >= 30:
        black_rate, white_rate, z_stat, _ = two_prop_z(
            black.sum(), len(black),
            white.sum(), len(white)
        )

        T4["by_band"][band] = {
            "n_black": int(len(black)),
            "black": float(black_rate),
            "white": float(white_rate),
            "z": float(z_stat)
        }

R["test4"] = T4

(ROOT / "results" / "tests.json").write_text(json.dumps(R, indent=1, default=float))
print(json.dumps({k: v for k, v in R.items()}, indent=1, default=float))