# 📊 Statistical Analysis of Brazilian E-Commerce (Olist)

## Project Overview
End-to-end statistical analysis of 100,000+ real e-commerce orders
from Olist, Brazil's largest department store marketplace. Covers
descriptive statistics, hypothesis testing, A/B testing and
correlation analysis — with a focus on actionable business insights.

---

## Project Structure
```
project3-stats-analysis/
│
├── data/
│   ├── raw/                        ← original Olist CSV files
│   └── processed/                  ← SQLite database (gitignored)
│
├── notebooks/
│   └── stats_analysis.ipynb        ← main analysis notebook
│
├── src/
│   ├── __init__.py
│   ├── data_loader.py              ← loads CSVs into SQLite
│   └── stats_helpers.py            ← 9 reusable statistical functions
│
└── README.md
```

---

## Dataset
**Source:** [Olist Brazilian E-Commerce Dataset](https://www.kaggle.com/datasets/olistbr/brazilian-ecommerce)

| Table | Rows | Description |
|---|---|---|
| orders | 99,441 | Order status and timestamps |
| order_items | 112,650 | Products and prices per order |
| payments | 103,886 | Payment type and installments |
| reviews | 99,224 | Customer review scores |
| customers | 99,441 | Customer city and state |
| products | 32,951 | Product categories |
| sellers | 3,095 | Seller locations |
| geolocation | 1,000,163 | ZIP code coordinates |

---

## Statistical Tests Implemented

### Reusable Functions (`src/stats_helpers.py`)
| Function | Type |
|---|---|
| `descriptive_summary()` | Descriptive statistics |
| `test_normality()` | Shapiro-Wilk / D'Agostino-Pearson + Q-Q plot |
| `confidence_interval()` | 95% CI for the mean |
| `two_sample_ttest()` | Welch's t-test |
| `students_ttest()` | Student's t-test with Levene's pre-test |
| `paired_ttest()` | Paired t-test for dependent samples |
| `mann_whitney_test()` | Non-parametric, 2 groups + effect size |
| `kruskal_wallis_test()` | Non-parametric, 3+ groups + Bonferroni post-hoc |
| `chi_square_test()` | Independence between categorical variables |
| `one_way_anova()` | Parametric, 3+ groups |
| `correlation_significance()` | Pearson/Spearman with p-values |

---

## Key Findings

### Descriptive Statistics
- Median order value (BRL 91.79) more representative than mean
  (BRL 139.75) due to extreme right skew (skewness=7.51)
- Log transformation reduces skewness to 0.587
- Large sample problem demonstrated — normality tests unreliable
  at n > 5,000; visual inspection + CLT preferred

### Hypothesis Testing

**Test 1 — High Value Orders Take Longer (Welch's t-test) ✅**
- Low value: 11.01 days vs High value: 12.95 days
- Statistically significant — ~2 day difference

**Test 2 — Review Score Differs by Payment Type (Kruskal-Wallis) ✅**
- All payment types share median score of 5
- Debit card significantly differs from credit card, voucher and boleto
- Same median can mask meaningful distributional differences

**Test 3 — Late Delivery Depends on Order Value (Chi-square) ✅**
- Q4 (highest value) orders have highest late delivery rate
- Corroborates Test 1 — high value orders consistently
  underperform on delivery metrics

**Test 4 — Order Values Differ Across States (ANOVA + Kruskal-Wallis) ✅**
- Both parametric and non-parametric tests agree
- PB highest, SP lowest mean order value
- Counterintuitive: SP (wealthiest state) orders less per transaction
  due to higher volume of small/impulse purchases

**A/B Test — Q4 Promotion Had Negligible Effect ❌**
- Not statistically significant (p > 0.05)
- Cohen's d = 0.0045 → negligible practical effect
- Observational data caveat — seasonal confounders may exist

### Correlation Analysis
- price ↔ total_order_value: very strong positive (expected)
- freight_value ↔ total_order_value: strong positive
- delivery_days ↔ estimated_days: strong positive
- delivery_days ↔ review_score: weak negative ⚠️
  → longer delivery correlates with lower satisfaction
- total_order_value ↔ review_score: not significant

### Three-Test Narrative
Tests 1, 3 and correlation analysis converge:
> High value orders → longer delivery → more late deliveries → lower reviews
>
> **Recommendation:** Dedicated fast-track fulfillment for Q4 orders

---

## Technical Notes

### Why Spearman over Pearson?
Data is non-normal (confirmed by normality testing). Spearman uses
rank ordering — no normality assumption required. Both computed and
compared; largest disagreement on freight_value ↔ delivery_days
suggesting a non-linear relationship.

### Why Kruskal-Wallis over ANOVA for Review Scores?
Review scores are ordinal (1-5) — violates ANOVA's continuous,
normally distributed assumption. Kruskal-Wallis tests rank
distributions without parametric assumptions.

### Large Sample Problem
At n=114,859 the D'Agostino-Pearson test rejects normality even
on perfectly simulated normal data. Statistical significance
diverges from practical significance at large n. Decision rule:

| Sample size | Approach |
|---|---|
| n < 30 | Normality test result critical |
| 30 < n < 5,000 | Visual inspection + skewness/kurtosis |
| n > 5,000 | Ignore p-value, use CLT + visual inspection |

### Why Both ANOVA and Kruskal-Wallis for State Analysis?
Order values non-normal but CLT justifies parametric ANOVA on
state means at large n. Running both validates finding is robust
to distributional assumptions — both tests agreed.

---

## Tools & Libraries
| Library | Purpose |
|---|---|
| pandas | Data manipulation |
| numpy | Numerical operations |
| matplotlib / seaborn | Visualization |
| scipy.stats | Statistical tests |
| sqlalchemy | Python-SQL connection |

---

## How to Run
1. Clone the repository
2. Download Olist dataset from Kaggle and place CSVs in `data/raw/`
3. Install dependencies:
   `pip install pandas numpy matplotlib seaborn scipy sqlalchemy`
4. Open `notebooks/stats_analysis.ipynb` in VS Code or Jupyter
5. Run all cells top to bottom — database auto-generates on first run

---

## Dataset Citation
Olist. (2018). Brazilian E-Commerce Public Dataset by Olist.
Retrieved from https://www.kaggle.com/datasets/olistbr/brazilian-ecommerce