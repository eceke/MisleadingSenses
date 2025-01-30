import pandas as pd
from scipy.stats import chi2_contingency
import matplotlib.pyplot as plt
import seaborn as sns

# Load main dataset (figure-ground responses)
file_path_main = "Result_test.csv"
df_main = pd.read_csv(file_path_main)

# Load left-right response dataset
file_path_lr = "Left_Right_Responses.csv"
df_lr = pd.read_csv(file_path_lr)

# Define stimulus columns for analysis
stim_columns = ['Stimulus 1', 'Stimulus 8', 'Stimulus 10', 'Stimulus 14', 
                'Stimulus 17', 'Stimulus 18', 'Stimulus 19', 'Stimulus 20', 
                'Stimulus 21', 'Stimulus 22']
trial_columns = ['Stimulus 2', 'Stimulus 3', 'Stimulus 4', 'Stimulus 5', 
                 'Stimulus 6', 'Stimulus 7', 'Stimulus 9', 'Stimulus 11', 
                 'Stimulus 12', 'Stimulus 13', 'Stimulus 15', 'Stimulus 16']

# --- DATA CLEANING AND FILTERING ---
# Ensure only the relevant columns are used
df_cleaned = df_main[['Group', 'Participant'] + stim_columns]

# Mapping reading/writing direction groups
group_mapping = {"A": "Left-to-Right", "B": "Right-to-Left", "C": "Bidirectional"}
df_cleaned["Group"] = df_cleaned["Group"].map(group_mapping)

# Compute total black & white figure responses
df_cleaned["Total Black Figure"] = df_cleaned[stim_columns].sum(axis=1)
df_cleaned["Total White Figure"] = len(stim_columns) - df_cleaned["Total Black Figure"]

# --- Visualization: Group-Wise Proportion of Responses ---
group_summary = df_cleaned.groupby("Group")[["Total Black Figure", "Total White Figure"]].mean()
group_summary.plot(kind="bar", figsize=(8, 6), color=["black", "gray"], width=0.8)
plt.title("Group-Wise Proportion of Responses")
plt.ylabel("Average Count")
plt.xlabel("Group")
plt.xticks(rotation=0)
plt.legend(["Black Figure", "White Figure"], title="Response")
plt.tight_layout()
plt.show()

# --- Visualization: Participant-Level Response Heatmap ---
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
plt.show()

# --- LEFT/RIGHT RESPONSE ANALYSIS ---
df_lr = df_lr.drop(columns=['Unnamed: 0'], errors='ignore')  # Remove unnecessary index column

# Merge df_cleaned (main responses) with df_lr (left/right responses) based on "Participant"
df_combined = pd.merge(df_cleaned, df_lr, on="Participant", suffixes=('_figure', '_side'))

# Define new column names for left/right responses
stim_columns_side = [col + "_side" for col in stim_columns]

# Convert "Left"/"Right" responses to numerical values (1 = Right, 0 = Left)
for col in stim_columns:
    df_combined[col + "_side"] = df_combined[col + "_side"].map({"Right": 1, "Left": 0})

# --- Right vs. Left Comparison (Counts) ---
right_vs_left_summary = df_combined.groupby("Group")[stim_columns_side].sum()
right_vs_left_summary["Total Right Choices"] = right_vs_left_summary.sum(axis=1)
right_vs_left_summary["Total Left Choices"] = len(stim_columns) * df_combined.groupby("Group").size() - right_vs_left_summary["Total Right Choices"]

# --- Right vs. Left Comparison (Percentage) ---
right_vs_left_summary["Right Choice %"] = (right_vs_left_summary["Total Right Choices"] / (right_vs_left_summary["Total Right Choices"] + right_vs_left_summary["Total Left Choices"])) * 100
right_vs_left_summary["Left Choice %"] = 100 - right_vs_left_summary["Right Choice %"]

# --- Visualization: Right vs. Left (Counts) ---
right_vs_left_summary[["Total Right Choices", "Total Left Choices"]].plot(kind="bar", stacked=True, figsize=(8,6), color=["purple", "orange"])
plt.title("Total Right vs. Left Choices per Group")
plt.ylabel("Total Count")
plt.xlabel("Group")
plt.xticks(rotation=0)
plt.legend(["Right Choices", "Left Choices"], title="Choice")
plt.tight_layout()
plt.show()

# --- Visualization: Right vs. Left (Percentage) ---
right_vs_left_summary["Right Choice %"].plot(kind="bar", figsize=(8,6), color=["purple", "orange", "blue"], width=0.8)
plt.title("Percentage of Right-Side Choices per Group")
plt.ylabel("Right-Side Choice (%)")
plt.xlabel("Group")
plt.xticks(rotation=0)
plt.ylim(0, 100)
plt.axhline(y=50, color='red', linestyle='--', label="Neutral (50%)")
plt.legend(["Right-Side Preference", "Neutral Line"])
plt.tight_layout()
plt.show()

# --- Chi-Square Tests ---
# Chi-Square for Black vs. White Figure
# Use only the stimulus columns for the observed frequencies
grouped_data = df_cleaned.groupby("Group")[stim_columns].sum()
summary = grouped_data.apply(lambda x: pd.Series([x.sum(), len(x) * len(stim_columns) - x.sum()]), axis=1)
summary.columns = ["Black Figure, White Background", "White Figure, Black Background"]

# Debugging: Print the observed frequencies for Black vs. White Figure
print("\nObserved Frequencies (Black vs. White):")
print(summary)

chi2_black_white, p_black_white, dof_black_white, _ = chi2_contingency(summary.values)

# Chi-Square for Right vs. Left Choice (Counts)
chi2_pref, p_pref, dof_pref, _ = chi2_contingency(right_vs_left_summary[["Total Right Choices", "Total Left Choices"]].values)

# --- Table for Chi-Square Results ---
chi_square_results = pd.DataFrame({
    "Test": ["Black vs. White Figure", "Right vs. Left Preference"],
    "Chi-Square Statistic": [chi2_black_white, chi2_pref],
    "p-value": [p_black_white, p_pref],
    "Degrees of Freedom": [dof_black_white, dof_pref]
})

# Display Chi-Square Results
print("\nChi-Square Test Results:")
print(chi_square_results)

# --- Visualization: Chi-Square p-values ---
sns.barplot(
    x=chi_square_results["Test"], 
    y=chi_square_results["p-value"], 
    palette=["black", "purple"]
)
plt.axhline(y=0.05, color='red', linestyle='--', label="Significance Threshold (p=0.05)")
plt.title("Chi-Square Test p-values")
plt.ylabel("p-value")
plt.xticks(rotation=15, ha="right")
plt.legend()
plt.tight_layout()
plt.show()

# --- Stacked Bar Chart: Distribution of Responses Across Groups (Per Stimulus) ---
response_data = df_cleaned.melt(
    id_vars=["Participant", "Group"],
    var_name="Stimuli",
    value_name="Response"
)

response_summary = (
    response_data.groupby(["Stimuli", "Group"])["Response"]
    .value_counts(normalize=False)  
    .unstack(fill_value=0)
    .reset_index()
)

# --- Visualization: Stacked Bar Chart ---
plt.figure(figsize=(12, 8))
sns.barplot(
    data=response_summary,
    x="Stimuli",
    y=1,  # Count of "Black Figure"
    hue="Group",
    palette=["green", "blue", "orange"]
)
plt.title("Distribution of Responses Across Groups (Per Stimulus)")
plt.ylabel("Count of 'Black Figure' Responses")
plt.xlabel("Stimuli")
plt.legend(title="Group")
plt.xticks(rotation=45, ha="right")
plt.tight_layout()
plt.show()
