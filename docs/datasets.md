# Datasets

## Inclusion criteria

A dataset is included in PaMIR if it meets all five conditions:

1. **Flat table** — a single row-per-loan table, or a source the recipe reduces
   to one by a documented join/aggregation (e.g. `bondora`).
2. **Binary default target** — a column indicating default (1) or non-default (0).
3. **Publicly downloadable** — available without NDA or institutional access.
4. **At least 1,000 rows.**
5. **At least 3% default rate** — enough signal for streaming evaluation.

## Summary statistics

- **19 datasets**, 1,237,550 total rows
- **281,766 defaults** (overall DR 22.8%)
- **9 named countries** (Brazil, Estonia, Finland, Germany, India, Poland, Spain, Taiwan, USA), plus the non-geographic labels Synthetic / educational, Unknown, and Unspecified
- **7 product types**: Consumer, Consumer revolving + installment, Corporate, Credit card, P2P consumer, Small business, Vehicle finance
- Default rates from 3.2% to 40.9%

## Key references

Where a dataset was introduced or is canonically cited, use these (also in each
catalog entry's `citation` field). Several are part of the credit-scoring
benchmark of Lessmann, Baesens, Seow & Thomas (2015), *EJOR*, 247(1), 124–136.
Datasets without an academic reference are cited via their source and
`attribution` in [Licenses & attribution](licenses.md).

- **bankruptcy** — Liang, D., Lu, C.-C., Tsai, C.-F., & Shih, G.-A. (2016). Financial ratios and corporate governance indicators in bankruptcy prediction: A comprehensive study. European Journal of Operational Research, 252(2), 561–572.
- **gmsc** — Kaggle “Give Me Some Credit” competition (2011), Credit Fusion & Will Cukierski.
- **pakdd** — PAKDD 2010 Data Mining Competition (NeuroTech Ltd. / UFPE). Also benchmarked in Lessmann, Baesens, Seow & Thomas (2015), EJOR 247(1), 124–136.
- **poland_1yr, poland_3yr, poland_5yr** — Zięba, M., Tomczak, S. K., & Tomczak, J. M. (2016). Ensemble boosted trees with synthetic features generation in application to bankruptcy prediction. Expert Systems with Applications, 58, 93–101.
- **sba** — Li, M., Mickel, A., & Taylor, S. (2018). “Should This Loan be Approved or Denied?”: A Large Dataset with Class Assignment Guidelines. Journal of Statistics Education, 26(1), 55–66.
- **south_german** — Grömping, U. (2019). South German Credit Data: Correcting a Widely Used Data Set. Report 4/2019, Beuth University of Applied Sciences Berlin.
- **taiwan** — Yeh, I.-C., & Lien, C.-H. (2009). The comparisons of data mining techniques for the predictive accuracy of probability of default of credit card clients. Expert Systems with Applications, 36(2), 2473–2480.

## Full catalog

```python
from pamir import load_catalog
print(load_catalog().to_string())
```

---

### bankruptcy — Taiwanese Bankruptcy Prediction

| Property | Value |
|---|---|
| **Rows** | 6,819 |
| **Features** | 94 |
| **Defaults** | 220 (3.2%) |
| **Geography** | Taiwan |
| **Product** | Corporate |
| **Target** | Company bankruptcy per Taiwan Stock Exchange rules (Taiwan Economic Journal, 1999–2009) |
| **Source** | [UCI ML Repository (Taiwanese Bankruptcy Prediction, Liang et al. 2016)](https://archive.ics.uci.edu/dataset/572) |
| **Fetched from** | Kaggle dataset fedesoriano/company-bankruptcy-prediction |
| **License** | Data files © Original Authors (Kaggle) |

### bondora — Bondora P2P Lending

| Property | Value |
|---|---|
| **Rows** | 266,482 |
| **Features** | 46 |
| **Defaults** | 109,124 (40.9%) |
| **Geography** | Estonia, Finland, Spain |
| **Product** | P2P consumer |
| **Target** | Loan status terminal: Late (default) vs Repaid |
| **Source** | [Bondora public loan book (Kaggle mirror marcobeyer/bondora-p2p-loans)](https://www.kaggle.com/datasets/bondora/peer-to-peer-loans) |
| **Fetched from** | Kaggle dataset marcobeyer/bondora-p2p-loans |
| **License** | CC0 1.0 (Kaggle reuploader) |
| **Harmonization notes** | Terminal loans only (Repaid/Late); post-outcome columns removed by the day-zero cut (PrincipalBalance, RecoveryStage, …). Bondora's origination estimates (ProbabilityOfDefault, LossGivenDefault) are kept — day-zero, 1-D AUC ≈ 0.52–0.58. Bondora's public loan book updates over time; the recipe pins the source that reproduces the benchmark snapshot (266,482 terminal loans, DR 0.41). |

### conorsully — Conor Sully Credit Score

| Property | Value |
|---|---|
| **Rows** | 1,000 |
| **Features** | 34 |
| **Defaults** | 284 (28.4%) |
| **Geography** | Synthetic / educational |
| **Product** | Consumer |
| **Target** | Default on personal loan |
| **Source** | [Kaggle (conorsully1/credit-score), synthetic educational dataset](https://www.kaggle.com/datasets/conorsully1/credit-score) |
| **Fetched from** | Kaggle dataset conorsully1/credit-score |
| **License** | CC0 1.0 |

### dish — Automobile Loan Default

| Property | Value |
|---|---|
| **Rows** | 121,856 |
| **Features** | 38 |
| **Defaults** | 9,845 (8.1%) |
| **Geography** | Unspecified |
| **Product** | Vehicle finance |
| **Target** | Auto loan default |
| **Source** | [Kaggle: saurabhbagchi/dish-network-hackathon](https://www.kaggle.com/datasets/saurabhbagchi/dish-network-hackathon) |
| **Fetched from** | Kaggle dataset saurabhbagchi/dish-network-hackathon |
| **License** | CC0 1.0 (Kaggle reuploader) |
| **Harmonization notes** | Despite the `dish-network` slug, the Kaggle dataset is titled "Automobile Loan Default Dataset" and its columns (Car_Owned, Bike_Owned, Loan_Annuity, …) are an auto-loan set — **not** Dish Network / telecom. Provenance unverified; the earlier "Dish Network / USA / Telecom" labeling was wrong. Kept under id `dish` for stability. |

### gastonstat — Gaston Sanchez Credit Scoring

| Property | Value |
|---|---|
| **Rows** | 4,454 |
| **Features** | 13 |
| **Defaults** | 1,254 (28.1%) |
| **Geography** | Unknown |
| **Product** | Consumer |
| **Target** | Loan default |
| **Source** | [GitHub (gastonstat/CreditScoring)](https://github.com/gastonstat/CreditScoring) |
| **Fetched from** | GitHub file https://raw.githubusercontent.com/gastonstat/CreditScoring/master/CreditScoring.csv |
| **License** | No license (GitHub, no LICENSE file) |

### gmsc — Give Me Some Credit

| Property | Value |
|---|---|
| **Rows** | 150,000 |
| **Features** | 10 |
| **Defaults** | 10,026 (6.7%) |
| **Geography** | USA |
| **Product** | Consumer revolving + installment |
| **Target** | 90+ days past due within 2 years on revolving/installment |
| **Source** | [Kaggle competition](https://www.kaggle.com/datasets/brycecf/give-me-some-credit-dataset) |
| **Fetched from** | Kaggle dataset brycecf/give-me-some-credit-dataset |
| **License** | Unknown (Kaggle reuploader) |


### laotse — Laotse Credit Risk

| Property | Value |
|---|---|
| **Rows** | 32,581 |
| **Features** | 11 |
| **Defaults** | 7,108 (21.8%) |
| **Geography** | Unspecified |
| **Product** | Consumer |
| **Target** | Loan default |
| **Source** | [Kaggle](https://www.kaggle.com/datasets/laotse/credit-risk-dataset) |
| **Fetched from** | Kaggle dataset laotse/credit-risk-dataset |
| **License** | CC0 1.0 (Kaggle reuploader) |

### lc_clean — Lending Club (cleaned, 2007–2014)

| Property | Value |
|---|---|
| **Rows** | 150,000 |
| **Features** | 19 |
| **Defaults** | 30,273 (20.2%) |
| **Geography** | USA |
| **Product** | P2P consumer |
| **Target** | Loan charged off |
| **Source** | [LendingClub loan data (cleaned third-party version)](https://huggingface.co/datasets/RPD123-byte/credit-risk-datasets) |
| **Fetched from** | Hugging Face dataset RPD123-byte/credit-risk-datasets |
| **License** | No license (HuggingFace repo, none stated) |

### lc_my — Lending Club (Malaysian variant)

| Property | Value |
|---|---|
| **Rows** | 100,000 |
| **Features** | 16 |
| **Defaults** | 22,639 (22.6%) |
| **Geography** | Unspecified |
| **Product** | Consumer |
| **Target** | Loan default |
| **Source** | [Kaggle](https://www.kaggle.com/datasets/zaurbegiev/my-dataset) |
| **Fetched from** | Kaggle dataset zaurbegiev/my-dataset |
| **License** | Unknown (Kaggle reuploader) |
| **Harmonization notes** | Credit Score column rescaled (subset had values inflated 10x) |

### lc_small — Lending Club (small, 9,578 loans)

| Property | Value |
|---|---|
| **Rows** | 9,578 |
| **Features** | 13 |
| **Defaults** | 1,533 (16.0%) |
| **Geography** | USA |
| **Product** | Consumer |
| **Target** | Not fully paid |
| **Source** | [Kaggle / educational](https://www.kaggle.com/datasets/itssuru/loan-data) |
| **Fetched from** | Kaggle dataset itssuru/loan-data |
| **License** | ODbL (Open Database License) |

### lt_vehicle — L&T Vehicle Loan Default

| Property | Value |
|---|---|
| **Rows** | 233,154 |
| **Features** | 36 |
| **Defaults** | 50,611 (21.7%) |
| **Geography** | India |
| **Product** | Vehicle finance |
| **Target** | Default on the first EMI of a vehicle loan (L&T Financial Services) |
| **Source** | [Kaggle competition (Analytics Vidhya / L&T)](https://www.kaggle.com/datasets/mamtadhaker/lt-vehicle-loan-default-prediction) |
| **Fetched from** | Kaggle dataset mamtadhaker/lt-vehicle-loan-default-prediction |
| **License** | Other (specified in description) |

### pakdd — PAKDD 2010 Credit Data

| Property | Value |
|---|---|
| **Rows** | 50,000 |
| **Features** | 43 |
| **Defaults** | 13,041 (26.1%) |
| **Geography** | Brazil |
| **Product** | Consumer |
| **Target** | Bad loan (PAKDD competition target) |
| **Source** | [PAKDD 2010 Data Mining Competition (JLZml GitHub mirror)](https://github.com/JLZml/Credit-Scoring-Data-Sets) |
| **Fetched from** | GitHub file https://raw.githubusercontent.com/JLZml/Credit-Scoring-Data-Sets/master/2.%20PAKDD%202009%20Data%20Mining%20Competition/PAKDD%202010.zip |
| **License** | No license (competition archive) |
| **Harmonization notes** | Raw modeling data (headerless) rebuilt with the 54 variable names from PAKDD2010_VariablesList; ID_CLIENT (sequential key) dropped |

### poland_1yr — Polish Companies Bankruptcy (1-year horizon)

| Property | Value |
|---|---|
| **Rows** | 7,027 |
| **Features** | 64 |
| **Defaults** | 271 (3.9%) |
| **Geography** | Poland |
| **Product** | Corporate |
| **Target** | Bankruptcy within 1 year (Polish firms, 2000–2013) |
| **Source** | [UCI ML Repository (Zieba et al. 2016)](https://archive.ics.uci.edu/dataset/365) |
| **Fetched from** | UCI archive https://archive.ics.uci.edu/static/public/365/polish+companies+bankruptcy+data.zip |
| **License** | CC-BY-4.0 (UCI) |
| **Harmonization notes** | ARFF read; 64 financial-ratio features (Attr1–Attr64), target `class`. Some ratios are undefined for distressed firms (division by a zero denominator) and come through as missing; `Attr27`'s missingness carries a moderate day-zero signal (isna-AUC ≈ 0.71) — a legitimate property of the data, not leakage. |

### poland_3yr — Polish Companies Bankruptcy (3-year horizon)

| Property | Value |
|---|---|
| **Rows** | 10,503 |
| **Features** | 64 |
| **Defaults** | 495 (4.7%) |
| **Geography** | Poland |
| **Product** | Corporate |
| **Target** | Bankruptcy within 3 years (Polish firms, 2000–2013) |
| **Source** | [UCI ML Repository (Zieba et al. 2016)](https://archive.ics.uci.edu/dataset/365) |
| **Fetched from** | UCI archive https://archive.ics.uci.edu/static/public/365/polish+companies+bankruptcy+data.zip |
| **License** | CC-BY-4.0 (UCI) |

### poland_5yr — Polish Companies Bankruptcy (5-year horizon)

| Property | Value |
|---|---|
| **Rows** | 5,910 |
| **Features** | 64 |
| **Defaults** | 410 (6.9%) |
| **Geography** | Poland |
| **Product** | Corporate |
| **Target** | Bankruptcy within 5 years (Polish firms, 2000–2013) |
| **Source** | [UCI ML Repository (Zieba et al. 2016)](https://archive.ics.uci.edu/dataset/365) |
| **Fetched from** | UCI archive https://archive.ics.uci.edu/static/public/365/polish+companies+bankruptcy+data.zip |
| **License** | CC-BY-4.0 (UCI) |

### prosper — Prosper Marketplace Loans

| Property | Value |
|---|---|
| **Rows** | 55,084 |
| **Features** | 58 |
| **Defaults** | 17,010 (30.9%) |
| **Geography** | USA |
| **Product** | P2P consumer |
| **Target** | Loan defaulted or charged off |
| **Source** | [Kaggle (Prosper loan data)](https://www.kaggle.com/datasets/henryokam/prosper-loan-data) |
| **Fetched from** | Kaggle dataset henryokam/prosper-loan-data |
| **License** | CC0 1.0 (Kaggle reuploader) |
| **Harmonization notes** | Post-outcome columns removed by the semantic day-zero cut; identifier keys (LoanKey/MemberKey/GroupKey) dropped |

### sba — U.S. SBA Loan Defaults

| Property | Value |
|---|---|
| **Rows** | 2,102 |
| **Features** | 19 |
| **Defaults** | 686 (32.6%) |
| **Geography** | USA |
| **Product** | Small business |
| **Target** | SBA-backed loan charged off |
| **Source** | [Li, Mickel & Taylor (2018), Journal of Statistics Education](https://www.kaggle.com/datasets/larsen0966/sba-loans-case-data-set) |
| **Fetched from** | Kaggle dataset larsen0966/sba-loans-case-data-set |
| **License** | CC0 1.0 (Kaggle reuploader) |
| **Harmonization notes** | Label-leaking columns removed: Term (Term%12 encodes the label), RealEstate (=Term≥240), daysterm (=Term×30); Name (surrogate); DisbursementDate/Recession (day-zero cut) |


### south_german — South German Credit (corrected)

| Property | Value |
|---|---|
| **Rows** | 1,000 |
| **Features** | 20 |
| **Defaults** | 300 (30.0%) |
| **Geography** | Germany |
| **Product** | Consumer |
| **Target** | Bad credit risk |
| **Source** | [Grömping (2019), Beuth University; UCI](https://archive.ics.uci.edu/dataset/573) |
| **Fetched from** | UCI archive https://archive.ics.uci.edu/static/public/573/south+german+credit+update.zip |
| **License** | CC-BY-4.0 (UCI) |

### taiwan — Taiwan Credit Card Default

| Property | Value |
|---|---|
| **Rows** | 30,000 |
| **Features** | 23 |
| **Defaults** | 6,636 (22.1%) |
| **Geography** | Taiwan |
| **Product** | Credit card |
| **Target** | Default on next month credit card payment |
| **Source** | [UCI Machine Learning Repository (Yeh & Lien 2009)](https://archive.ics.uci.edu/dataset/350) |
| **Fetched from** | Kaggle dataset uciml/default-of-credit-card-clients-dataset |
| **License** | CC0 1.0 (Kaggle “UCI ML” reuploader) |

---


## Harmonization pipeline

PaMIR ships no data. `pamir.download(id)` fetches the raw file from the
dataset's original source into a local cache and applies the same recipe
(`pamir.harmonize`), reproducibly, so every user gets the identical table:

1. Read the raw file (dataset-specific separator / encoding).
2. Derive the binary `__target__` (numeric parse, or a dataset target rule).
3. Drop the target, id, date/timestamp and unnamed-index columns.
4. Coerce numeric-looking string columns; drop constant / all-missing columns;
   drop surrogate-key string columns (one distinct value per row).
5. **Drop post-outcome columns by the elicitation's semantic cut** — every
   column marked `day_zero_available = false` (not known to the lender at the
   moment of the decision), plus any dataset-specific extra drops. This
   replaces the earlier statistical `AUC > 0.95` rule, which passed sets of
   individually-weak columns that together leak the outcome.
6. Shuffle rows with a fixed seed (file order is not an origination sequence).

The full per-dataset recipe (source, target rule, day-zero drop list, extra
drops, shuffle seed) is in each catalog entry's `download` and `harmonize`
fields; the harmonized result is validated against `expected.json`
(columns, row count, default rate).

## Known limitations

- **No calendar order.**  Rows are shuffled with a fixed seed; file order carries no origination time, and the `lag` parameter approximates maturation delay in row positions.
- **No timestamps.**  Most datasets lack origination or resolution dates. The `lag` parameter approximates maturation delay in row positions.
- **Heterogeneous quality.**  Some datasets are curated competition data (gmsc, pakdd); others are educational (conorsully, lc_small).

All 19 datasets can be evaluated under both protocols provided by PaMIR:
the streaming protocol with label-maturation delay (`evaluate`) and the
conventional i.i.d. train/test split (`evaluate_iid`).