"""Create a data dictionary for the raw HMDA dataset."""

import re
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]

raw = pd.read_csv(
    ROOT / "data" / "raw" / "hmda_2017_ny_all-records_labels.csv",
    dtype=str
)

# Information about numeric columns: type, unit and meaning
NUMERIC = {
    "loan_amount_000s": ("continuous", "USD thousands", "amount applied for"),
    "applicant_income_000s": ("continuous", "USD thousands / year", "gross annual income relied on by the lender"),
    "rate_spread": ("continuous", "percentage points", "APR minus average prime offer rate; reported only above 1.5 pp (first lien)"),
    "population": ("discrete", "persons", "census-tract population"),
    "minority_population": ("continuous", "% of tract", "minority share of tract population"),
    "hud_median_family_income": ("continuous", "USD / year", "HUD median family income of the MSA"),
    "tract_to_msamd_income": ("continuous", "% of MSA median", "tract median family income relative to the MSA"),
    "number_of_owner_occupied_units": ("discrete", "housing units", "owner-occupied units in tract"),
    "number_of_1_to_4_family_units": ("discrete", "housing units", "1-4 family units in tract"),
    "as_of_year": ("discrete", "year", "reporting year"),
    "sequence_number": ("discrete", "-", "record sequence (blank in public file)"),
}

# Meanings for columns that are not numeric measurements
MEANING = {
    "respondent_id": "lender identifier",
    "agency_name": "lender's federal regulator",
    "loan_type_name": "conventional / FHA / VA / FSA-RHS",
    "property_type_name": "1-4 family, manufactured, multifamily",
    "loan_purpose_name": "purchase / refinance / home improvement",
    "owner_occupancy_name": "principal dwelling or not",
    "preapproval_name": "whether a preapproval was requested",
    "action_taken_name": "outcome of the application",
    "msamd_name": "metropolitan area",
    "county_name": "county of the property",
    "census_tract_number": "census tract",
    "applicant_ethnicity_name": "applicant ethnicity (self-reported)",
    "applicant_race_name_1": "applicant first race",
    "applicant_sex_name": "applicant sex",
    "co_applicant_sex_name": "co-applicant sex or 'No co-applicant'",
    "purchaser_type_name": "who bought the loan in the year",
    "denial_reason_name_1": "first denial reason (optional for some lenders)",
    "hoepa_status_name": "high-cost HOEPA loan or not",
    "lien_status_name": "first / subordinate / unsecured",
    "state_name": "state of the property",
    "state_abbr": "state abbreviation",
    "co_applicant_ethnicity_name": "co-applicant ethnicity",
    "edit_status_name": "edit status (blank in public file)",
    "application_date_indicator": "application date flag (blank in public file)",
}

# Add descriptions for the remaining race and denial-reason columns
for i in range(1, 6):
    MEANING[f"applicant_race_name_{i}"] = f"applicant race, choice {i}"
    MEANING[f"co_applicant_race_name_{i}"] = f"co-applicant race, choice {i}"

for i in range(1, 4):
    MEANING[f"denial_reason_name_{i}"] = f"denial reason {i} (optional for some lenders)"

rows = []

for column in raw.columns:
    values = raw[column]

    if column in NUMERIC:
        column_type, unit, meaning = NUMERIC[column]
        numbers = pd.to_numeric(values, errors="coerce")

        if numbers.notna().any():
            value_range = f"{numbers.min():g} to {numbers.max():g}"
        else:
            value_range = "all blank"

    else:
        column_type = "nominal"

        # Check whether a matching label column exists
        possible_labels = [
            column + "_name",
            column.replace("_code", "_name"),
            re.sub(r"_(\d)$", r"_name_\1", column)
        ]

        label = None
        for name in possible_labels:
            if name != column and name in raw.columns:
                label = name
                break

        if column == "agency_abbr":
            unit = "-"
            meaning = "regulator abbreviation"
        elif label:
            unit = "code"
            meaning = f"numeric code for `{label}`"
        else:
            unit = "-"
            meaning = MEANING.get(
                column,
                "label of the matching code column"
            )

        if values.notna().any():
            value_range = f"{values.nunique()} levels"
        else:
            value_range = "all blank"

    rows.append({
        "attribute": column,
        "type": column_type,
        "unit": unit,
        "range": value_range,
        "missing_n": int(values.isna().sum()),
        "missing_pct": round(100 * values.isna().mean(), 2),
        "meaning": meaning
    })

dd = pd.DataFrame(rows)
dd.to_csv(ROOT / "results" / "data_dictionary.csv", index=False)
print("Data dictionary saved to results/data_dictionary.csv")