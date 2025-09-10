import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import statistics
from scipy.stats import chi2_contingency, ttest_ind
from statsmodels.stats.proportion import proportion_confint
from statsmodels.stats.weightstats import _zconfint_generic


df1 = pd .read_csv("dataset1.csv")
df2 = pd .read_csv("dataset2.csv")

print(df1.info())
print(df2.info())


missing1 = (df1.isnull().sum() / len(df1)) * 100
print("Missing % in Dataset1:")
print(missing1)

missing2 = (df2.isnull().sum() / len(df2)) * 100
print("Missing % in Dataset2:")
print(missing2)

df1['habit'] = df1['habit'].fillna("Unknown")

#Convert datatypes
# For Dataset1
for col in ['start_time', 'rat_period_start', 'rat_period_end', 'sunset_time']:
    df1[col] = pd.to_datetime(df1[col], errors='coerce', dayfirst=True)

# For Dataset2
df2['time'] = pd.to_datetime(df2['time'], errors='coerce', dayfirst=True)


#Handle duplicates
df1 = df1.drop_duplicates()
df2 = df2.drop_duplicates()

# Example variables
cols1 = ['bat_landing_to_food', 'seconds_after_rat_arrival', 'hours_after_sunset']
cols2 = ['bat_landing_number', 'food_availability', 'rat_minutes', 'rat_arrival_number']

for col in cols1:
    print(f"\nDataset1 - {col}")
    print("Mean:", statistics.mean(df1[col]))
    print("Median:", statistics.median(df1[col]))
    try:
        print("Mode:", statistics.mode(df1[col]))
    except:
        print("Mode: No unique mode")
    print("Min:", np.min(df1[col]))
    print("Max:", np.max(df1[col]))
    print("Variance:", np.var(df1[col]))
    print("Std Dev:", np.std(df1[col]))
    print("25th percentile:", np.percentile(df1[col], 25))
    print("75th percentile:", np.percentile(df1[col], 75))

for col in cols2:
    print(f"\nDataset2 - {col}")
    print("Mean:", statistics.mean(df2[col]))
    print("Median:", statistics.median(df2[col]))
    try:
        print("Mode:", statistics.mode(df2[col]))
    except:
        print("Mode: No unique mode")
    print("Min:", np.min(df2[col]))
    print("Max:", np.max(df2[col]))
    print("Variance:", np.var(df2[col]))
    print("Std Dev:", np.std(df2[col]))
    print("25th percentile:", np.percentile(df2[col], 25))
    print("75th percentile:", np.percentile(df2[col], 75))

print(df1['risk'].value_counts())



sns.countplot(x='risk', data=df1)
plt.title("Risk-taking vs. Risk-avoidance in bats")
plt.show()

# Vigilance: time to approach food
sns.boxplot(x='risk', y='bat_landing_to_food', data=df1)
plt.title("Hesitation Time vs Risk-taking")
plt.show()

# Distribution of rat arrivals
sns.histplot(df2['rat_arrival_number'], bins=20, kde=False)
plt.title("Distribution of Rat Arrivals per 30-min block")
plt.show()

# Rat minutes vs Bat landings
sns.scatterplot(x='rat_minutes', y='bat_landing_number', data=df2)
plt.title("Bat landings vs Rat presence")
plt.show()

# Hypothesis: Bat risk-taking is linked to reward
contingency = pd.crosstab(df1['risk'], df1['reward'])
chi2, p, dof, expected = chi2_contingency(contingency)
print("Chi-square Test between Risk and Reward")
print("Chi2 =", chi2, "p =", p)

# Hypothesis: Risk-taking bats have lower hesitation (bat_landing_to_food)
risk0 = df1[df1['risk']==0]['bat_landing_to_food']
risk1 = df1[df1['risk']==1]['bat_landing_to_food']

tstat, pval = ttest_ind(risk0, risk1)
print("T-test Risk vs Hesitation Time: t =", tstat, "p =", pval)


