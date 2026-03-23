# src/stats_helpers.py
# ──────────────────────────────────────────────────────────────
# Reusable statistical helper functions
# ──────────────────────────────────────────────────────────────

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats

# ──────────────────────────────────────────────────────────────
# 1. DESCRIPTIVE STATISTICS
# ──────────────────────────────────────────────────────────────
def descriptive_summary(df, col, label=None):
    '''Returns a clean descriptive statistics summary for a column.'''
    label = label or col
    s = df[col].dropna()
    
    summary = pd.DataFrame({
        'Metric': ['Count', 'Mean', 'Median', 'Std Dev',
                   'Min', '25th Pct', '75th Pct', 'Max',
                   'Skewness', 'Kurtosis'],
        'Value': [
            f'{len(s):,}',
            f'{s.mean():.2f}',
            f'{s.median():.2f}',
            f'{s.std():.2f}',
            f'{s.min():.2f}',
            f'{s.quantile(0.25):.2f}',
            f'{s.quantile(0.75):.2f}',
            f'{s.max():.2f}',
            f'{s.skew():.3f}',
            f'{s.kurtosis():.3f}'
        ]
    })
    print(f'──── Descriptive Statistics: {label} ────')
    return summary

# ──────────────────────────────────────────────────────────────
# 2. NORMALITY TEST
# ──────────────────────────────────────────────────────────────
def test_normality(df, col, label=None, significance_level=0.05):
    '''
    Tests normality using Shapiro-Wilk (n<5000) or
    D'Agostino-Pearson (n>=5000).
    Optionally plots histogram + KDE + Q-Q plot.
    '''
    label = label or col
    s     = df[col].dropna()

    if len(s) < 5000:
        stat, p   = stats.shapiro(s.sample(min(len(s), 5000), random_state=42))
        test_name = "Shapiro-Wilk"
    else:
        stat, p   = stats.normaltest(s)
        test_name = "D'Agostino-Pearson"

    result = 'Normal ✅' if p > significance_level else 'Not Normal ❌'

    print(f'──── Normality Test: {label} ────')
    print(f'Test:        {test_name}')
    print(f'Statistic:   {stat:.4f}')
    print(f'P-value:     {p:.4f}')
    print(f'Result:      {result} (α={significance_level})')
    print()

    return stat, p

# ──────────────────────────────────────────────────────────────
# 3. CONFIDENCE INTERVAL
# ──────────────────────────────────────────────────────────────
def confidence_interval(df, col, confidence=0.95, label=None):
    '''Calculates confidence interval for the mean.'''
    label = label or col
    s     = df[col].dropna()
    
    mean  = s.mean()
    se    = stats.sem(s) # Standard error of the mean
    ci    = stats.t.interval(confidence, df=len(s)-1, loc=mean, scale=se)
    
    print(f'──── {int(confidence*100)}% Confidence Interval: {label} ────')
    print(f'Sample Mean:  {mean:.2f}')
    print(f'CI:           ({ci[0]:.2f}, {ci[1]:.2f})')
    print(f'Interpretation: We are {int(confidence*100)}% confident the true')
    print(f'               population mean lies between {ci[0]:.2f} and {ci[1]:.2f}')
    print()
    
    return ci

# ──────────────────────────────────────────────────────────────
# 4. TWO-SAMPLE T-TEST
# ──────────────────────────────────────────────────────────────
def two_sample_ttest(group1, group2, label1='Group 1', label2='Group 2', significance_level=0.05):
    '''
    Performs Welch's t-test (does not assume equal variances).
    More robust than Student's t-test for real-world data.
    '''

    # Levene's test to verify equal variance assumption first
    lev_stat, lev_p = stats.levene(group1.dropna(), group2.dropna())
    variance_equal   = lev_p > 0.05

    if variance_equal: #  Student's t-test,
        stat, p = stats.ttest_ind(group1.dropna(), group2.dropna(), 
                              equal_var=True # True for Student's t-tes
                              )
    else: # Welch's t-test
        stat, p = stats.ttest_ind(group1.dropna(), group2.dropna(), 
                                equal_var=False #False for Welch's t-test 
                                )
    result  = 'Significant ✅' if p < significance_level else 'Not Significant ❌'
    
    test_name = "Student's t-test" if variance_equal else "Welch's t-test"
    print(f'──── {test_name}: {label1} vs {label2} ────')
    print(f'{label1} mean:  {group1.mean():.2f}')
    print(f'{label2} mean:  {group2.mean():.2f}')
    print(f'Difference:    {group1.mean() - group2.mean():.2f}')
    print(f'T-statistic:   {stat:.4f}')
    print(f'P-value:       {p:.4f}')
    print(f'Result:        {result} (α={significance_level})')
    print()
    
    return stat, p

# ──────────────────────────────────────────────────────────────
# 5. CHI-SQUARE TEST
# ──────────────────────────────────────────────────────────────
def chi_square_test(df, col1, col2, significance_level=0.05):
    '''Tests independence between two categorical variables.'''
    contingency = pd.crosstab(df[col1], df[col2])
    chi2, p, dof, expected = stats.chi2_contingency(contingency)
    result = 'Dependent ✅' if p < significance_level else 'Independent ❌'
    
    print(f'──── Chi-Square Test: {col1} vs {col2} ────')
    print(f'Chi2 Statistic: {chi2:.4f}')
    print(f'Degrees of Freedom: {dof}')
    print(f'P-value:        {p:.4f}')
    print(f'Result:         {result} (α={significance_level})')
    print()
    
    return chi2, p, contingency

# ──────────────────────────────────────────────────────────────
# 6. ONE-WAY ANOVA
# ──────────────────────────────────────────────────────────────
def one_way_anova(*groups, labels=None, significance_level=0.05):
    '''Tests if means differ significantly across 3+ groups.'''
    stat, p = stats.f_oneway(*[g.dropna() for g in groups])
    result  = 'Significant difference ✅' if p < significance_level else 'No significant difference ❌'
    labels  = labels or [f'Group {i+1}' for i in range(len(groups))]
    
    print('──── One-Way ANOVA ────')
    for label, group in zip(labels, groups):
        print(f'{label}: mean={group.mean():.2f}, n={len(group):,}')
    print(f'F-statistic: {stat:.4f}')
    print(f'P-value:     {p:.4f}')
    print(f'Result:      {result} (α={significance_level})')
    print()
    
    return stat, p

# ──────────────────────────────────────────────────────────────
# 7. MANN-WHITNEY U TEST (non-parametric alternative to Welch's)
# ──────────────────────────────────────────────────────────────
def mann_whitney_test(group1, group2, label1='Group 1', label2='Group 2', significance_level=0.05):
    '''
    Non-parametric alternative to independent samples t-test.
    Use when:
    - Data is not normally distributed
    - Ordinal data (e.g. review scores 1-5)
    - Small sample sizes
    - Presence of outliers

    Tests whether one group tends to have higher values than the other.
    Does NOT assume normality — uses rank ordering instead.
    '''
    g1 = group1.dropna()
    g2 = group2.dropna()

    stat, p = stats.mannwhitneyu(g1, g2, alternative='two-sided')
    result  = 'Significant ✅' if p < significance_level else 'Not Significant ❌'

    # Effect size — rank biserial correlation
    # ranges from -1 to 1, like Cohen's d but for non-parametric
    n1, n2      = len(g1), len(g2)
    effect_size = 1 - (2 * stat) / (n1 * n2)

    print(f'──── Mann-Whitney U Test: {label1} vs {label2} ────')
    print(f'{label1}:')
    print(f'  n={n1:,}, median={g1.median():.2f}, mean={g1.mean():.2f}')
    print(f'{label2}:')
    print(f'  n={n2:,}, median={g2.median():.2f}, mean={g2.mean():.2f}')
    print(f'U-statistic:   {stat:.4f}')
    print(f'P-value:       {p:.4f}')
    print(f'Effect size:   {effect_size:.4f} (rank biserial correlation)')
    print(f'Result:        {result} (α={significance_level})')
    print()

    return stat, p, effect_size

# ──────────────────────────────────────────────────────────────
# 8. KRUSKAL-WALLIS TEST (non-parametric alternative to ANOVA)
# ──────────────────────────────────────────────────────────────
def kruskal_wallis_test(*groups, labels=None, significance_level=0.05):
    '''
    Non-parametric alternative to one-way ANOVA.
    Use when:
    - Data is not normally distributed
    - Comparing 3+ independent groups
    - Ordinal data
    - Unequal group sizes with non-normal distributions

    Tests whether at least one group tends to have 
    different values than the others.
    Does NOT tell you WHICH groups differ — use 
    post-hoc Dunn test for that.
    '''
    cleaned = [g.dropna() for g in groups]
    labels  = labels or [f'Group {i+1}' for i in range(len(groups))]

    stat, p = stats.kruskal(*cleaned)
    result  = 'Significant difference ✅' if p < significance_level else 'No significant difference ❌'

    print('──── Kruskal-Wallis Test ────')
    for label, group in zip(labels, cleaned):
        print(f'{label}:')
        print(f'  n={len(group):,}, median={group.median():.2f}, mean={group.mean():.2f}')

    print(f'\nH-statistic: {stat:.4f}')
    print(f'P-value:     {p:.4f}')
    print(f'Result:      {result} (α={significance_level})')

    # Post-hoc: if significant, run pairwise Mann-Whitney with Bonferroni correction
    if p < significance_level:
        print('\n──── Post-hoc: Pairwise Mann-Whitney (Bonferroni corrected) ────')
        n_comparisons = len(groups) * (len(groups) - 1) / 2
        adjusted_alpha = significance_level / n_comparisons  # Bonferroni correction

        for i in range(len(groups)):
            for j in range(i + 1, len(groups)):
                u_stat, u_p = stats.mannwhitneyu(
                    cleaned[i], cleaned[j],
                    alternative='two-sided'
                )
                sig = '✅' if u_p < adjusted_alpha else '❌'
                print(f'  {labels[i]} vs {labels[j]}: '
                      f'p={u_p:.4f} {sig} '
                      f'(adjusted α={adjusted_alpha:.4f})')
    print()

    return stat, p

# ──────────────────────────────────────────────────────────────
# 9. PLOT DISTRIBUTION
# ──────────────────────────────────────────────────────────────
def plot_distribution(df, col, label=None, bins=50):
    '''Plots histogram + KDE + normal curve overlay.'''
    label = label or col
    s     = df[col].dropna()

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    # ── Left: Histogram + KDE ────────────────────────────────
    sns.histplot(s, bins=bins, kde=True, color='#3498db',
                 ax=axes[0], stat='density')
    
    # Overlay normal curve
    x    = np.linspace(s.min(), s.max(), 100)
    axes[0].plot(x, stats.norm.pdf(x, s.mean(), s.std()),
                 'r--', linewidth=2, label='Normal curve')
    axes[0].axvline(s.mean(),   color='red',    linestyle='--',
                    linewidth=1, label=f'Mean: {s.mean():.2f}')
    axes[0].axvline(s.median(), color='green',  linestyle='--',
                    linewidth=1, label=f'Median: {s.median():.2f}')
    axes[0].set_title(f'Distribution: {label}', fontweight='bold')
    axes[0].legend()
    
    # ── Right: Q-Q Plot ───────────────────────────────────────
    stats.probplot(s, dist='norm', plot=axes[1])
    axes[1].set_title(f'Q-Q Plot: {label}', fontweight='bold')
    axes[1].get_lines()[0].set(color='#3498db', markersize=3)
    axes[1].get_lines()[1].set(color='red', linewidth=2)
    
    plt.tight_layout()
    plt.show()

