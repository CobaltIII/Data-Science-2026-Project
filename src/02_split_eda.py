"""Step 2: split into exploration / confirmation sets, then do all EDA on exploration only.

Input : data/processed/hmda_clean.parquet
Output: data/processed/split.parquet        row_id -> "exploration" | "confirmation"
        figures/*.pdf, figures/*.png        captioned in the report
        results/eda_summary.json            numbers quoted in captions and the text
Run   : python src/02_split_eda.py
"""
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

SEED = 42
TT_SPLIT = 0.30
ROOT = Path(__file__).resolve().parents[1]
FIG = ROOT / "figures"
RES = ROOT / "results"
FIG.mkdir(exist_ok=True)

# categorical slots 1-4 of the validated default palette, in fixed order
C = {
    "White": "#3478c5",
    "Black": "#e4774e",
    "Hispanic": "#239b82",
    "Asian": "#dba832"
}

INK, MUTED, GRID = "#292d33", "#68717c", "#e0e4e8"

plt.rcParams.update({
    "font.size": 8.5,
    "axes.edgecolor": MUTED,
    "axes.labelcolor": INK,
    "xtick.color": MUTED,
    "ytick.color": MUTED,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.grid": True,
    "grid.color": GRID,
    "grid.linewidth": 0.6,
    "axes.axisbelow": True,
    "legend.frameon": False,
    "figure.dpi": 150,
    "savefig.bbox": "tight",
})

#loading clean data
df = pd.read_parquet(ROOT / "data" / "processed" / "hmda_clean.parquet")

#splitting ibto exploration and confirmation sets w stratif
action_counts = df["action_taken_name"].map(
    df["action_taken_name"].value_counts()
)

stratify_series = df["action_taken_name"].where(
    action_counts > 50, "rare"
)
expl, conf = train_test_split(
    df.index, 
    test_size=TT_SPLIT, 
    random_state=SEED,
    stratify=stratify_series
)

split = pd.Series("exploration", index=df.index)
split[conf] = "confirmation"

split_df = pd.DataFrame({"row_id": df["row_id"], "split": split})
split_output_path = ROOT / "data" / "processed" / "split.parquet"
split_df.to_parquet(split_output_path, index=False)

def test_population(d):
    """Filters the dataset to include only home loan applications
for owner-occupied properties with 1-4 units and a first lien,
excluding non-human applicants."""
    mask = (
            (d["decision"] == 1)
            & (d["owner_occupancy_name"] == "Owner-occupied as a principal dwelling")
            & (d["property_type_name"].str.startswith("One-to-four"))
            & (d["lien_status_name"] == "Secured by a first lien")
            & (d["race_group"] != "Not applicable (non-natural person)")
        )
    return d[mask]

ex = test_population(df.loc[expl]).copy()
S = {
    "seed": SEED, 
    "test_size": TT_SPLIT, 
    "n_exploration": int(len(expl)),
    "n_confirmation": int(len(conf)), 
    "n_expl_population": int(len(ex)),
    "denial_rate_expl": float(ex["denied"].mean())
}
# income bands: quintiles of reported income, cut on the exploration set and reused for tests
q = ex["applicant_income_000s"].quantile([0.2, 0.4, 0.6, 0.8]).tolist()
S["income_quintile_cuts_000s"] = q

income_bins = [-np.inf] + q + [np.inf]
income_labels = ["Q1", "Q2", "Q3", "Q4", "Q5"]
ex["inc_band"] = pd.cut(ex["applicant_income_000s"], bins=income_bins, labels=income_labels).astype(object).fillna("Not reported")

def save(fig, name):
    fig.savefig(FIG / f"{name}.pdf")
    fig.savefig(FIG / f"{name}.png", dpi=200)
    plt.close(fig)

main = ["White", "Black", "Hispanic", "Asian"]

#fig1 raw denial rate by group
race_order = ["White", "Asian", "Not provided", "Hispanic", "Black", "Other race"]
race_data = ex.groupby("race_group")["denied"].agg(["mean", "size"])
race_data = race_data.loc[race_order]
fig, ax = plt.subplots(figsize=(3.4, 1.8))

denial_rates=race_data["mean"] * 100
ax.barh(range(len(race_data)), denial_rates, color="#3478c5", height=0.6)

race_labels=[]
for race, count in zip(race_data.index, race_data["size"]):
    race_labels.append(f"{race} (n={count:,})")

ax.set_yticks(range(len(race_data)), race_labels)
ax.invert_yaxis()
ax.set_xlabel("Denial rate (%)")
for i, v in enumerate(denial_rates):
    ax.text(v + 0.3, i, f"{v:.1f}", va="center", fontsize=7, color=INK)
save(fig, "fig1_denial_by_group")
S["denial_by_group"] = race_data["mean"].round(4).to_dict()
S["n_by_group"] = race_data["size"].to_dict()

#fig2 denial rates across income groups
income_bands = ["Q1", "Q2", "Q3", "Q4", "Q5"]

income_data = ex[ex["race_group"].isin(main)]
income_data = income_data.groupby(["inc_band", "race_group"])["denied"].mean().unstack() * 100

fig, ax = plt.subplots(figsize=(3.4, 2.1))

for race in main:
    ax.plot(
        range(5),
        income_data.loc[income_bands, race],
        marker="o",
        ms=4,
        lw=2,
        color=C[race],
        label=race
    )

ax.set_xticks(range(5), ["Q1\nlowest", "Q2", "Q3", "Q4", "Q5\nhighest"])
ax.set_xlabel("Applicant income quintile")
ax.set_ylabel("Denial rate (%)")

ax.legend(
    ncol=4,
    loc="upper center",
    bbox_to_anchor=(0.5, 1.16),
    handlelength=1.2,
    columnspacing=0.8
)

save(fig, "fig2_denial_by_income_group")

S["denial_by_band_group"] = income_data.round(2).to_dict()
S["black_white_ratio_by_band"] = (
    income_data["Black"] / income_data["White"]
).round(2).to_dict()


# fig3 income distribution before and after log transformation
fig, axes = plt.subplots(1, 2, figsize=(3.4, 1.6))

income = ex["applicant_income_000s"].dropna()
income_capped = income.clip(upper=income.quantile(0.999))

axes[0].hist(income_capped, bins=60, color="#3478c5")
axes[0].set_xlabel("Income, USD 000s (99.9% cap)")
axes[0].set_ylabel("Applications")

log_income = np.log10(income_capped)
axes[1].hist(log_income, bins=60, color="#3478c5")
axes[1].set_xlabel("log10 income, USD 000s")

save(fig, "fig3_income_log")

S["income_skew_raw_expl"] = float(income.skew())
S["income_skew_log_expl"] = float(np.log10(income).skew())

# fig4: most common denial reasons for each group
denied_apps = ex[(ex["denied"] == 1) & ex["race_group"].isin(main)]

reason_counts = denied_apps["denial_reason_name_1"].value_counts()
reasons = [reason for reason in reason_counts.index if reason != "Not reported"][:5]

denied_apps = denied_apps[denied_apps["denial_reason_name_1"] != "Not reported"]

reason_share = pd.crosstab(
    denied_apps["denial_reason_name_1"],
    denied_apps["race_group"],
    normalize="columns"
).loc[reasons, main] * 100

fig, ax = plt.subplots(figsize=(3.4, 2.3))

y = np.arange(len(reasons))
h = 0.2

for i, race in enumerate(main):
    ax.barh(
        y + (i - 1.5) * h,
        reason_share[race],
        height=h * 0.9,
        color=C[race],
        label=race
    )

short_names = {
    "Debt-to-income ratio": "Debt-to-income",
    "Credit history": "Credit history",
    "Collateral": "Collateral",
    "Credit application incomplete": "Incomplete application",
    "Other": "Other",
    "Insufficient cash (downpayment, closing costs)": "Insufficient cash",
    "Unverifiable information": "Unverifiable info"
}

ax.set_yticks(y, [short_names.get(reason, reason) for reason in reasons])
ax.invert_yaxis()
ax.set_xlabel("Share of the group's denials with a reason (%)")

ax.legend(
    ncol=4,
    loc="upper center",
    bbox_to_anchor=(0.45, 1.14),
    handlelength=1.2,
    columnspacing=0.8
)

save(fig, "fig4_denial_reasons")

S["denial_reason_share"] = reason_share.round(2).to_dict()

S["denial_reason_unreported_share"] = float(
    (ex.loc[ex["denied"] == 1, "denial_reason_name_1"] == "Not reported").mean()
)

S["unreported_by_agency"] = (
    ex[ex["denied"] == 1]
    .groupby("agency_abbr")["denial_reason_name_1"]
    .apply(lambda reasons: (reasons == "Not reported").mean())
    .round(3)
    .to_dict()
)

# fig5: higher priced loans by loan type and racial group
originated = ex[
    (ex["action_taken_name"] == "Loan originated")
    & ex["race_group"].isin(main)
]

loan_data = (
    originated.groupby(["loan_type_name", "race_group"])["higher_priced"]
    .mean()
    .unstack()[main] * 100
)

loan_types = ["Conventional", "FHA-insured", "VA-guaranteed"]

fig, ax = plt.subplots(figsize=(3.4, 1.9))
x = np.arange(len(loan_types))

for i, race in enumerate(main):
    ax.bar(
        x + (i - 1.5) * h,
        loan_data.loc[loan_types, race],
        width=h * 0.9,
        color=C[race],
        label=race
    )

ax.set_xticks(x, loan_types)
ax.set_ylabel("Higher-priced loans (%)")

ax.legend(
    ncol=4,
    loc="upper center",
    bbox_to_anchor=(0.5, 1.16),
    handlelength=1.2,
    columnspacing=0.8
)

save(fig, "fig5_higher_priced")

S["higher_priced_by_type_group"] = loan_data.round(2).to_dict()

S["loan_type_mix_by_group"] = (
    pd.crosstab(
        originated["race_group"],
        originated["loan_type_name"],
        normalize="index"
    )
    .round(3)
    .to_dict()
)
#more numbers for the analysis
S["denial_by_loan_type"] = ex.groupby("loan_type_name")["denied"].mean().round(4).to_dict()
S["denial_by_purpose"] = ex.groupby("loan_purpose_name")["denied"].mean().round(4).to_dict()
S["denial_by_band"] = ex.groupby("inc_band")["denied"].mean().round(4).to_dict()
S["denial_by_sex"] = ex.groupby("applicant_sex_name")["denied"].mean().round(4).to_dict()
S["denial_by_coapp"] = ex.groupby("has_coapplicant")["denied"].mean().round(4).to_dict()

S["median_income_by_group"] = ex.groupby("race_group")["applicant_income_000s"].median().to_dict()
S["median_lti_by_group"] = ex.groupby("race_group")["loan_to_income"].median().round(2).to_dict()

S["n_lenders_expl"] = int(ex["lender"].nunique())

lender_counts = ex["lender"].value_counts()
S["share_apps_top20_lenders"] = float(
    lender_counts.iloc[:20].sum() / lender_counts.sum()
)

lender_stats = ex.groupby("lender")["denied"].agg(["mean", "size"])
large_lenders = lender_stats[lender_stats["size"] >= 100]

S["lender_denial_rate_iqr_(n>=100)"] = (
    large_lenders["mean"]
    .quantile([0.25, 0.5, 0.75])
    .round(3)
    .tolist()
)

black_lenders = ex[ex["race_group"] == "Black"]["lender"].value_counts(normalize=True)
white_lenders = ex[ex["race_group"] == "White"]["lender"].value_counts(normalize=True)

S["lender_mix_dissimilarity_black_white"] = float(
    0.5 * black_lenders.sub(white_lenders, fill_value=0).abs().sum()
)

lender_data = ex.groupby("lender").agg(
    denial_rate=("denied", "mean"),
    black_share=("race_group", lambda s: (s == "Black").mean()),
    count=("denied", "size")
)

lender_data = lender_data[lender_data["count"] >= 100]

S["corr_denied_lender_rate_black_share"] = float(
    lender_data[["denial_rate", "black_share"]].corr().iloc[0, 1]
)

income_groups = pd.qcut(
    ex["minority_pct"],
    4,
    labels=["Q1", "Q2", "Q3", "Q4"]
)

S["denial_by_minority_pct_quartile"] = (
    ex.groupby(income_groups, observed=True)["denied"]
    .mean()
    .round(4)
    .to_dict()
)

(RES / "eda_summary.json").write_text(json.dumps(S, indent=1, default=str))
print(json.dumps(S, indent=1, default=str))