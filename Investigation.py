
# --- Importing Libraries ---
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from scipy.stats import ttest_ind, chi2_contingency
import statsmodels.stats.api as sms
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    mean_absolute_error, mean_squared_error, r2_score,
    accuracy_score
)

# --- Load Datasets ---
df1 = pd.read_csv("dataset1.csv")
df2 = pd.read_csv("dataset2.csv")


# --- Handle Missing Values & Convert Dates ---
df1.fillna({'habit': 'Unknown'}, inplace=True)
for col in ['start_time', 'rat_period_start', 'rat_period_end', 'sunset_time']:
    df1[col] = pd.to_datetime(df1[col], errors='coerce', dayfirst=True)
df2['time'] = pd.to_datetime(df2['time'], errors='coerce', dayfirst=True)


print("Dataset 1 Shape:", df1.shape)
print("Dataset 2 Shape:", df2.shape)
print("\nMissing Values (%):")
print(df1.isna().mean() * 100)

# INVESTIGATION A: Do bats perceive rats as predators?

print("\n--- INVESTIGATION A: Do bats perceive rats as predators? ---")

# --- Descriptive Statistics ---
print("\n--- Descriptive Statistics ---\n")
print("\n\nbat_landing_to_food:")
print(f"Mean: {df1['bat_landing_to_food'].mean():.2f}")
print(f"Median: {df1['bat_landing_to_food'].median():.2f}")
print(f"Std Dev: {df1['bat_landing_to_food'].std():.2f}")
print(f"Min: {df1['bat_landing_to_food'].min()}, Max: {df1['bat_landing_to_food'].max()}")

print("\nseconds_after_rat_arrival:")
print(f"Mean: {df1['seconds_after_rat_arrival'].mean():.2f}")
print(f"Median: {df1['seconds_after_rat_arrival'].median():.2f}")
print(f"Std Dev: {df1['seconds_after_rat_arrival'].std():.2f}")
print(f"Min: {df1['seconds_after_rat_arrival'].min()}, Max: {df1['seconds_after_rat_arrival'].max()}")

# --- Chi-Square and T-test ---
contingency = pd.crosstab(df1['risk'], df1['reward'])
chi2, p, _, _ = chi2_contingency(contingency)
print(f"\nChi-Square Test (Risk vs Reward): Chi2 = {chi2:.3f}, p = {p:.5f}")

risk0 = df1[df1['risk'] == 0]['bat_landing_to_food']
risk1 = df1[df1['risk'] == 1]['bat_landing_to_food']
t_stat, p_val = ttest_ind(risk0, risk1, equal_var=False)
print(f"T-test between Risk groups: t = {t_stat:.3f}, p = {p_val:.5f}")

# INVESTIGATION B: Seasonal Behaviour

print("\n--- INVESTIGATION B: Seasonal Behaviour ---")
season_stats = df1.groupby('season')[['bat_landing_to_food','risk','reward']].agg(['mean','std','count'])
print("\nSeasonal Summary:")
print(season_stats)

# --- Seasonal t-test ---
winter = df1.loc[df1['season'] == 0, 'bat_landing_to_food']
spring = df1.loc[df1['season'] == 1, 'bat_landing_to_food']
t_stat, p_val = ttest_ind(winter, spring, equal_var=False)
print(f"T-test (Winter vs Spring): t={t_stat:.3f}, p={p_val:.5f}")

# Outlier Detection & Correlation

print("\n--- Outlier Detection & Correlation Analysis ---")
Q1 = df1['bat_landing_to_food'].quantile(0.25)
Q3 = df1['bat_landing_to_food'].quantile(0.75)
IQR = Q3 - Q1
filtered = df1[~((df1['bat_landing_to_food'] < (Q1 - 1.5 * IQR)) | (df1['bat_landing_to_food'] > (Q3 + 1.5 * IQR)))]
corr = filtered[['bat_landing_to_food','seconds_after_rat_arrival','hours_after_sunset','risk','reward']].corr()
sns.heatmap(corr, annot=True, cmap='coolwarm')
plt.title("FIGURE 1: Correlation Matrix (Cleaned Data)")
plt.show()

# Confidence Interval

print("\n--- Confidence Interval for Hesitation Time ---")
cm = sms.DescrStatsW(df1['bat_landing_to_food'].dropna())
ci_low, ci_high = cm.tconfint_mean()
print(f"95% CI for hesitation time: {ci_low:.2f} to {ci_high:.2f}")


# Regression Modelling
print("\n--- Regression Modelling ---")
X = df1[['seconds_after_rat_arrival']]
y = df1['bat_landing_to_food']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
lr = LinearRegression().fit(X_train, y_train)
y_pred = lr.predict(X_test)
print(f"Slope: {lr.coef_[0]:.3f}, Intercept: {lr.intercept_:.3f}")
print(f"MAE: {mean_absolute_error(y_test, y_pred):.3f}")
print(f"RMSE: {np.sqrt(mean_squared_error(y_test, y_pred)):.3f}")
print(f"R²: {r2_score(y_test, y_pred):.3f}")

# --- Multiple Regression ---
df1_encoded = pd.get_dummies(df1, columns=['season'], drop_first=True)
X = df1_encoded[['seconds_after_rat_arrival','hours_after_sunset','season_1']]
y = df1_encoded['risk']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
model = LinearRegression().fit(X_train, y_train)
pred = model.predict(X_test)
adj_r2 = 1 - (1 - r2_score(y_test, pred)) * (len(y_test)-1)/(len(y_test)-X_test.shape[1]-1)
print(f"Adjusted R²: {adj_r2:.3f}")
print(f"MSE Simple: {mean_squared_error(y_test, y_pred):.3f} | MSE Multiple: {mean_squared_error(y_test, pred):.3f}")


# Merge & Feature Engineering

print("\n--- Merging Datasets & Feature Engineering ---")
merged = pd.merge(df1, df2, on=['month','hours_after_sunset'], how='left')
merged['rat_intensity'] = merged['rat_minutes'] / (merged['rat_arrival_number'] + 1)
merged['bat_activity'] = merged['bat_landing_number'] / (merged['food_availability'] + 1)

# Probability Insight

present = df1[df1['seconds_after_rat_arrival'] > 0]
p_risk = (present['risk'] == 1).mean()
print(f"\nP(Risk | Rat present) = {p_risk:.3f}")


# Distribution of Hesitation Time
plt.figure(figsize=(8,5))
sns.histplot(df1['bat_landing_to_food'], bins=30, kde=True, color='skyblue')
plt.title("FIGURE 2: Distribution of Bat Hesitation Time")
plt.xlabel("Hesitation Time (seconds)")
plt.ylabel("Frequency")
plt.show()

# Risk vs Reward Relationship
plt.figure(figsize=(6,4))
sns.countplot(x='risk', hue='reward', data=df1, palette='viridis')
plt.title("FIGURE 3: Frequency of Risk vs Reward Outcomes")
plt.xlabel("Risk Level (0=Low, 1=High)")
plt.ylabel("Count")
plt.legend(title='Reward (0/1)')
plt.show()

# Seasonal Comparison of Bat Activity
plt.figure(figsize=(8,5))
sns.barplot(x='season', y='bat_landing_to_food', data=df1, ci=None, palette='coolwarm')
plt.title("FIGURE 4: Average Bat Hesitation by Season")
plt.xlabel("Season")
plt.ylabel("Mean Hesitation Time (s)")
plt.show()

#  Scatter Plot of Rat Arrival vs Hesitation
plt.figure(figsize=(8,5))
sns.scatterplot(x='seconds_after_rat_arrival', y='bat_landing_to_food', hue='risk', data=df1, palette='cool')
plt.title("FIGURE 5: Hesitation Time vs Rat Arrival (by Risk Level)")
plt.xlabel("Seconds After Rat Arrival")
plt.ylabel("Bat Hesitation Time (s)")
plt.show()

#  Pairplot for Key Metrics
sns.pairplot(df1[['bat_landing_to_food','seconds_after_rat_arrival','hours_after_sunset','risk','reward']],
             diag_kind='kde', corner=True)
plt.suptitle("FIGURE 6: Pairwise Relationships Between Key Variables", y=1.02)
plt.show()

#  Rat Intensity vs Bat Activity (Merged Data)
sns.scatterplot(x='rat_intensity', y='bat_activity', hue='season', data=merged)
plt.title("FIGURE 7: Rat Intensity vs Bat Activity by Season")
plt.show()

print("\nExtended Analysis Complete.")
