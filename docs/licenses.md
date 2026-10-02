# Licenses & attribution

**PaMIR distributes no data.** `pamir.download(id)` fetches each dataset from its
**original source** into your local cache. You download it directly from that
source and are bound by **its** license and terms — below. The PaMIR **package
code** is Apache-2.0; that covers the code only, never the datasets.

> **Note on Kaggle licenses.** A license shown on a Kaggle dataset is almost
> always set by the *re-uploader*, not the rights-holder — a "CC0" there is the
> uploader's statement about their upload, and does not transfer rights in the
> underlying data. Since PaMIR does not redistribute the tables, the practical
> risk is low; the column below states what each *source* declares and what it
> obliges **you** to do.

Results (AUCs, tables, figures) may be published for all datasets; the
obligations concern **redistributing the tables themselves** and **attribution**.

## Per-dataset terms

| dataset | license (as declared by the downloaded source) | your obligation | origin |
|---|---|---|---|
| `bankruptcy` | Data files © Original Authors (Kaggle) | Primary source is UCI 572 (CC-BY-4.0) — attribution to Liang et al. (2016) required. | [source](https://archive.ics.uci.edu/dataset/572) |
| `bondora` | CC0 1.0 (Kaggle reuploader) | Set by the reuploader, not Bondora; underlying data is Bondora AS's public loan book — attribute Bondora AS. | [source](https://www.kaggle.com/datasets/bondora/peer-to-peer-loans) |
| `conorsully` | CC0 1.0 | No restrictions; synthetic dataset. | [source](https://github.com/conor-sully/credit-score) |
| `dish` | CC0 1.0 (Kaggle reuploader) | No stated restrictions; provenance unverified (Kaggle title: “Automobile Loan Default Dataset”). | [source](https://www.kaggle.com/datasets/saurabhbagchi/dish-network-hackathon) |
| `gastonstat` | No license (GitHub, no LICENSE file) | No rights granted — research use only; do not redistribute. | [source](https://github.com/gastonstat/CreditScoring) |
| `gmsc` | Unknown (Kaggle reuploader) | Primary source is the 2011 Kaggle competition — competition rules; do not redistribute. | [source](https://www.kaggle.com/datasets/brycecf/give-me-some-credit-dataset) |
| `laotse` | CC0 1.0 (Kaggle reuploader) | No stated restrictions. | [source](https://www.kaggle.com/datasets/laotse/credit-risk-dataset) |
| `lc_clean` | No license (HuggingFace repo, none stated) | LendingClub data via a third party — no rights granted; research use only. | [source](https://huggingface.co/datasets/RPD123-byte/credit-risk-datasets) |
| `lc_my` | Unknown (Kaggle reuploader) | No rights granted — research use only. | [source](https://www.kaggle.com/datasets/zaurbegiev/my-dataset) |
| `lc_small` | ODbL (Open Database License) | Share-alike + attribution: a derived database must also be ODbL. | [source](https://www.kaggle.com/datasets/itsmesunil/bank-loan-modelling) |
| `lt_vehicle` | Other (specified in description) | L&T hackathon archive — no rights granted; research use only. | [source](https://www.kaggle.com/datasets/mamtadhaker/lt-vehicle-loan-default-prediction) |
| `pakdd` | No license (competition archive) | PAKDD 2010 (NeuroTech / UFPE) — research use only. | [source](https://github.com/JLZml/Credit-Scoring-Data-Sets) |
| `poland_1yr` | CC-BY-4.0 (UCI) | Attribution to Zięba et al. (2016) required. | [source](https://archive.ics.uci.edu/dataset/365) |
| `poland_3yr` | CC-BY-4.0 (UCI) | Attribution to Zięba et al. (2016) required. | [source](https://archive.ics.uci.edu/dataset/365) |
| `poland_5yr` | CC-BY-4.0 (UCI) | Attribution to Zięba et al. (2016) required. | [source](https://archive.ics.uci.edu/dataset/365) |
| `prosper` | CC0 1.0 (Kaggle reuploader) | Set by the reuploader, not Prosper; underlying is Prosper Marketplace data. | [source](https://www.kaggle.com/datasets/henryokam/prosper-loan-data) |
| `sba` | CC0 1.0 (Kaggle reuploader) | Educational subset — cite Li, Mickel & Taylor (2018). | [source](https://www.kaggle.com/datasets/larsen0966/sba-loans-case-data-set) |
| `south_german` | CC-BY-4.0 (UCI) | Attribution to Grömping (2019) required. | [source](https://archive.ics.uci.edu/dataset/573) |
| `taiwan` | CC0 1.0 (Kaggle “UCI ML” reuploader) | Primary source is UCI 350 (CC-BY-4.0) — attribute Yeh & Lien (2009). | [source](https://archive.ics.uci.edu/dataset/350) |

Attribution is required by license for **bankruptcy, poland_1yr, poland_3yr,
poland_5yr, south_german, taiwan** (CC-BY-4.0 via their UCI primary source) and
**lc_small** (ODbL share-alike).

## Attribution & citations

Cite the source of each dataset you use — `pamir.dataset_info(id)["citation"]`:

- **bankruptcy** — Liang, D., Lu, C.-C., Tsai, C.-F., & Shih, G.-A. (2016). Financial ratios and corporate governance indicators in bankruptcy prediction: A comprehensive study. European Journal of Operational Research, 252(2), 561–572.
- **bondora** — Bondora AS public loan book (Estonia); CC-BY-SA-4.0 — share-alike + attribution
- **conorsully** — Conor Sully, synthetic credit-score dataset (Kaggle conorsully1/credit-score), MIT
- **gmsc** — Kaggle “Give Me Some Credit” competition (2011), Credit Fusion & Will Cukierski.
- **lc_small** — LendingClub via Kaggle (itssuru/loan-data); ODbL — share-alike + attribution required
- **pakdd** — PAKDD 2010 Data Mining Competition (NeuroTech Ltd. / UFPE). Also benchmarked in Lessmann, Baesens, Seow & Thomas (2015), EJOR 247(1), 124–136.
- **poland_1yr** — Zięba, M., Tomczak, S. K., & Tomczak, J. M. (2016). Ensemble boosted trees with synthetic features generation in application to bankruptcy prediction. Expert Systems with Applications, 58, 93–101.
- **poland_3yr** — Zięba, M., Tomczak, S. K., & Tomczak, J. M. (2016). Ensemble boosted trees with synthetic features generation in application to bankruptcy prediction. Expert Systems with Applications, 58, 93–101.
- **poland_5yr** — Zięba, M., Tomczak, S. K., & Tomczak, J. M. (2016). Ensemble boosted trees with synthetic features generation in application to bankruptcy prediction. Expert Systems with Applications, 58, 93–101.
- **sba** — Li, M., Mickel, A., & Taylor, S. (2018). “Should This Loan be Approved or Denied?”: A Large Dataset with Class Assignment Guidelines. Journal of Statistics Education, 26(1), 55–66.
- **south_german** — Grömping, U. (2019). South German Credit Data: Correcting a Widely Used Data Set. Report 4/2019, Beuth University of Applied Sciences Berlin.
- **taiwan** — Yeh, I.-C., & Lien, C.-H. (2009). The comparisons of data mining techniques for the predictive accuracy of probability of default of credit card clients. Expert Systems with Applications, 36(2), 2473–2480.

