# Credit Risk Scoring: Predicting Loan Default with Cost-Aware Thresholding

A credit risk scoring project that predicts which borrowers are likely to become seriously delinquent (90+ days past due) within two years, and translates model output into a decision a bank can act on.

Beyond comparing models, the project asks the question a lending team would ask: **where should the approval cutoff be, given that a missed default costs far more than a declined good customer?**

## Business Problem

A bank approving loans faces two kinds of mistakes:

- **Approving a borrower who later defaults** → lost principal (expensive).
- **Declining a borrower who would have repaid** → lost interest income (cheaper, but it adds up).

A model that only maximizes accuracy ignores this asymmetry. This project builds a risk score, explains what drives it, and chooses a cutoff based on cost.

## Dataset

[Give Me Some Credit](https://www.kaggle.com/c/GiveMeSomeCredit) (Kaggle): 150,000 borrowers, 10 features, binary target `SeriousDlqin2yrs`.

- **Class imbalance:** only about 6.7% of borrowers defaulted.
- **Missing values:** `MonthlyIncome` (~20% missing) and `NumberOfDependents` (~2.6% missing).

## Approach

1. **Split:** 80/20 stratified train/test split (`random_state=42`) so the default rate is preserved in both sets.
2. **Missing values:** median imputation fitted on the training set only (to avoid leakage), plus an `IncomeMissing` flag.
3. **Imbalance handling:** class weighting (`class_weight="balanced"`, `balanced_subsample`, `scale_pos_weight`). Accuracy is not used as a headline metric, since predicting "no default" for everyone would already score about 93%.
4. **Models compared:** Logistic Regression (baseline), Random Forest, XGBoost.
5. **Explainability:** SHAP values on the XGBoost model.
6. **Decision analysis:** cutoff selection by total expected cost under explicit cost assumptions.

## Results

### Model comparison

| Model | ROC-AUC |
|---|---|
| Logistic Regression (baseline) | 0.802 |
| Random Forest | 0.864 |
| XGBoost | 0.869 |

Tree-based models clearly outperform the linear baseline. The gap between Random Forest and XGBoost is small, and with a single train/test split it should not be read as a definitive ranking.

### What drives the risk score (SHAP)

Most influential features, in order:

1. `RevolvingUtilizationOfUnsecuredLines`: the share of available credit already used. Higher utilization pushes the score toward higher risk.
2. `NumberOfTime30-59DaysPastDueNotWorse`
3. `NumberOfTimes90DaysLate`
4. `age`: younger borrowers receive higher risk scores.

Delinquency history features show a sharp effect: borrowers with any past-due record are pushed strongly toward higher risk. `DebtRatio` shows no consistent pattern. `NumberOfDependents` and the `IncomeMissing` flag contribute very little.

**Business reading:** high credit utilization combined with early signs of late payment (30-59 days) are strong warning signals that appear well before a loan becomes seriously delinquent. This suggests a use as an **early-warning trigger**, for example proactive outreach or restructuring offers, rather than waiting for 90 days of delinquency.

### Choosing the approval cutoff

The model outputs a risk score; the cutoff is a business decision. Assuming (illustratively) that approving a borrower who defaults costs 10 times more than declining a good borrower:

| Metric | Value |
|---|---|
| Best cutoff | 0.55 |
| Missed defaults | 530 |
| Good borrowers declined | 4,526 |
| Recall / Precision at cutoff | 0.736 / 0.246 |
| Total cost with model | 9,826 |
| Total cost approving everyone | 20,050 |

**Cost is roughly 51% lower than approving every application** under this assumption.

The cost curve is U-shaped and fairly flat between 0.50 and 0.65, so the exact cutoff is not fragile.

### Sensitivity to the cost assumption

| Cost ratio (missed default : declined good borrower) | Best cutoff | Cost with model | Cost approving everyone | Reduction |
|---|---|---|---|---|
| 3 : 1 | 0.85 | 4,491 | 6,015 | ~25% |
| 5 : 1 | 0.75 | 6,347 | 10,025 | ~37% |
| 10 : 1 | 0.55 | 9,826 | 20,050 | ~51% |
| 20 : 1 | 0.40 | 13,897 | 40,100 | ~65% |

The more costly a missed default is, the lower (stricter) the optimal cutoff, and the more valuable the model becomes. Cutoffs were evaluated on a grid with steps of 0.05, and the curve is flat near its minimum (for example, at 3 : 1 the cutoffs 0.75 to 0.85 give nearly identical cost). The cost ratios are illustrative; a real institution would estimate them from its own loss and margin data.

## Limitations

- Cost ratios are assumptions, not bank data.
- The cutoff was selected on the same test set used for reporting, so the reported savings are somewhat optimistic. A separate validation set (or cross-validation) would be the cleaner approach.
- Model comparison uses a single split; cross-validation would show whether the Random Forest / XGBoost difference is real.
- Model scores are risk rankings (class weighting distorts them), not calibrated default probabilities.
- `age` influences the score. In real lending, using age in credit decisions can be restricted by fairness and regulatory requirements and would need review.
- No hyperparameter tuning beyond reasonable defaults.

## Next Steps

- Cross-validation and a proper train/validation/test split for cutoff selection.
- Probability calibration.
- Fairness check across age groups.
- Dashboard of risk bands (Power BI) for non-technical stakeholders.

## Tech Stack

Python, pandas, scikit-learn, XGBoost, SHAP, matplotlib. Developed in Google Colab.

## Repository Structure

```
credit-risk-scoring/
├── Credit_Risk_Scoring.ipynb
├── README.md
└── requirements.txt
```

## Author

Leo Titok Lubis, Information Systems student, interested in data science for banking and risk.
