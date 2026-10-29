## **CSE558 Data Science, Monsoon 2026, midterm project. Group: Prophet and Loss**

This repository holds the code behind our midterm report. Starting from the public HMDA 2017 New York
mortgage file, it cleans and encodes the data and splits it 70/30 into exploration and confirmation
sets (seed 42). It then draws the report's figures from the exploration set and runs four hypothesis
tests on the confirmation set:

| Test | Question | Method |
|---|---|---|
| 1 | At the same lender, for the same product and income band, are Black applicants denied more often than White applicants? | paired *t*-test |
| 2 | In one like-for-like slice of applications, does the denial rate depend on group? | χ² test of independence |
| 3 | Do lenders give different denial reasons to different groups? | χ² test of independence |
| 4 | Among approved FHA borrowers, are Black borrowers more often given a higher-priced loan? | two-proportion *z*-test |

## Repository layout

```
.
├── README.md
└── src/
    ├── 01_prepare.py          clean + encode the raw file, size check
    ├── 02_split_eda.py        70/30 split (seed 42) + the five EDA figures
    ├── 03_tests.py            the four hypothesis tests
    └── 04_data_dictionary.py  type, range and missingness of all 78 raw columns
```

Running the scripts creates these folders next to `src/`. None of them needs to be committed:

```
data/raw/          the HMDA CSV you download (step 1 below)
data/processed/    hmda_clean.parquet, hmda_encoded.parquet, split.parquet
figures/           fig1–fig5, as .pdf (used in the report) and .png
results/           prep_summary.json, eda_summary.json, tests.json, data_dictionary.csv
```

Every number quoted in the report comes from one of the three JSON files in `results/`.

## How to run

All commands below are run from this folder, the one that contains `README.md`.

### 1. Get the data 

The raw file is 357 MB, which is too large for GitHub, so it is not in the repository.

1. Go to <https://www.consumerfinance.gov/data-research/hmda/historic-data/>.
2. Choose **2017**, then **New York**, then **All records**, then **Plain language labels and HMDA codes**.
3. Unzip the download and put the CSV here:

```
data/raw/hmda_2017_ny_all-records_labels.csv
```

The CSV should have 446,902 rows and 78 columns.

### 2. Install the libraries

```bash
python -m pip install pandas numpy scipy statsmodels scikit-learn matplotlib pyarrow
```

### 3. Run everything

```bash
python src/01_prepare.py && python src/02_split_eda.py && python src/03_tests.py && python src/04_data_dictionary.py
```

The scripts must run in this order:

| Script | Reads | Writes |
|---|---|---|
| `01_prepare.py` | `data/raw/…csv` | `data/processed/hmda_clean.parquet`, `hmda_encoded.parquet`, `results/prep_summary.json` |
| `02_split_eda.py` | `hmda_clean.parquet` | `data/processed/split.parquet`, `figures/fig1–fig5`, `results/eda_summary.json` |
| `03_tests.py` | `hmda_clean.parquet`, `split.parquet`, `eda_summary.json` | `results/tests.json` |
| `04_data_dictionary.py` | `data/raw/…csv` | `results/data_dictionary.csv` |

## Reproducibility

The only random step is the exploration/confirmation split in `02_split_eda.py`
(`SEED = 42`, stratified on `action_taken`). Everything else is deterministic, so running the
pipeline twice gives byte-identical files in `results/`. To use a different seed, change `SEED`
in `src/02_split_eda.py` and rerun. That changes every EDA and test number, so the report would need
to be updated to match.


Tested with Python 3.13, pandas 3.0, numpy 2.5, scipy 1.18, statsmodels 0.15, scikit-learn 1.9,
matplotlib 3.11 and pyarrow 26.

## Data source

Consumer Financial Protection Bureau, *HMDA Loan Application Register, 2017, New York, all
records*. Public data, published under the Home Mortgage Disclosure Act. The code definitions are in
the `lar_record_codes.pdf` file that comes with the download.
