import pandas as pd
from scipy.stats import chi2_contingency
import matplotlib.pyplot as plt
import seaborn as sns
from tabulate import tabulate

file_path = 'Result_test.csv'
df = pd.read_csv(file_path)

# Remove horizontally split images

stim_columns = ['Stimulus 1', 'Stimulus 8', 'Stimulus 10', 'Stimulus 14', 'Stimulus 17', 'Stimulus 18', 'Stimulus 19', 'Stimulus 20', 'Stimulus 21', 'Stimulus 22']
trial_columns = ['Stimulus 2', 'Stimulus 3', 'Stimulus 4', 'Stimulus 5', 'Stimulus 6', 'Stimulus 7', 'Stimulus 9', 'Stimulus 11', 'Stimulus 12', 'Stimulus 13', 'Stimulus 15', 'Stimulus 16' ]  # Example trial stimuli to exclude
df_cleaned = df.drop(columns=trial_columns)

df_trial = df.drop(columns=stim_columns)

grouped_trial = df_trial.groupby("Group").sum(numeric_only=True).T
grouped_data = df_cleaned.groupby("Group").sum(numeric_only=True).T

# Mapping
group_mapping = {"A": "Left-to-Right", "B": "Right-to-Left", "C": "Bidirectional"}
df_cleaned["Group"] = df_cleaned["Group"].map(group_mapping)
df_trial["Group"] = df_trial["Group"].map(group_mapping)


df_cleaned["Total Black Figure"] = df_cleaned.iloc[:, 2:].sum(axis=1)
df_cleaned["Total White Figure"] = df_cleaned.iloc[:, 2:].shape[1] - df_cleaned["Total Black Figure"]

group_summary = df_cleaned.groupby("Group")[["Total Black Figure", "Total White Figure"]].mean()

# 1. Bar Plot: Group-wise proportion of responses
group_summary.plot(kind="bar", figsize=(8, 6), color=["black", "gray"], width=0.8)
plt.title("Group-Wise Proportion of Responses")
plt.ylabel("Average Count")
plt.xlabel("Group")
plt.xticks(rotation=0)
plt.legend(["Black Figure", "White Figure"], title="Response")
plt.tight_layout()
#plt.show()

# 2. Distribution of responses (Stacked Bar Chart)
response_data = df_cleaned.melt(
    id_vars=["Participant", "Group"],
    var_name="Stimuli",
    value_name="Response"
)

# Summarize data for plotting
response_summary = (
    response_data.groupby(["Stimuli", "Group"])["Response"]
    .value_counts(normalize=False)  
    .unstack(fill_value=0)
    .reset_index()
)

# --- Visualization ---
plt.figure(figsize=(12, 8))
sns.barplot(
    data=response_summary,
    x="Stimuli",
    y=1,  # Count of "Black Figure"
    hue="Group",
    palette=["green", "blue", "orange"]
)
plt.title("Distribution of Responses Across Groups (Stimuli)")
plt.ylabel("Count of 'Black Figure' Responses")
plt.xlabel("Stimuli")
plt.legend(title="Group")
plt.xticks(rotation=45, ha="right")
plt.tight_layout()
#plt.show()

# 3. Participant-Level Response Heatmap
plt.figure(figsize=(12, 8))
sns.heatmap(
    df_cleaned.iloc[:, 2:-2].T,
    cmap="coolwarm",
    cbar=True,
    xticklabels=df_cleaned["Participant"],
    yticklabels=df_cleaned.columns[2:-2],
)
plt.title("Participant-Level Responses Across Stimuli")
plt.xlabel("Participants")
plt.ylabel("Stimuli")
plt.tight_layout()
#plt.show()

# Display descriptive statistics

print("Participant-Level Responses Summary:")
print(df_cleaned[["Participant", "Group"]])

print("\nGroup-Level Averages:")
print(group_summary)

summary = grouped_data.apply(lambda x: [x.sum(), len(x) * len(df.columns[2:]) - x.sum()], axis=0).T
summary.columns = ["Black Figure, White Background", "White Figure, Black Background"]

#Perform the Chi-Square Test of Independence
chi2, p, dof, expected = chi2_contingency(summary.values)


# Print the results
print("Observed Frequencies (Summary Table):")
print(summary)
print("\nChi-Square Statistic:", chi2)
print("p-value:", p)
print("Degrees of Freedom:", dof)

#print(tabulate(df_cleaned.drop(columns=stim_columns), headers = 'keys', tablefmt = 'pretty', showindex=False))
