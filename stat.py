import numpy as np
import pandas as pd
from scipy.stats import mannwhitneyu, ttest_ind
import scipy.stats as stats

df = pd.read_csv('log/result.csv')

subject_groups = {subject: df[df['Adaptive_Scheme'] == subject]['Iteration'].tolist() 
                  for subject in df['Adaptive_Scheme'].unique()}

first_subject = list(subject_groups.keys())[0]
other_subjects = list(subject_groups.keys())[1:]

results = {}

for subject in other_subjects:
    first_values = subject_groups[first_subject]
    second_values = subject_groups[subject]

    t_stat, p_value = stats.ttest_ind(first_values, second_values, equal_var=False, alternative='greater')

    # p_value_one_tailed = p_value / 2 if t_stat > 0 else 1 - (p_value / 2)
    results[subject] = {
        't_statistic': t_stat,
        'p_value': p_value
    }

# Print the results
for subject, result in results.items():
    print(f"Test comparing {first_subject} vs {subject}:")
    print(f"  t-statistic: {result['t_statistic']:.4f}, p-value: {result['p_value']:.4f}")
