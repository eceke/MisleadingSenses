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

# Remove trial stimuli from the main dataset
df_cleaned = df_main.drop(columns=trial_columns)

# Mapping reading/writing direction groups
group_mapping = {"A": "Left-to-Right", "B": "Right-to-Left", "C": "Bidirectional"}
df_cleaned["Group"] = df_cleaned["Group"].map(group_mapping)

# Compute total black & white figure responses
df_cleaned["Total Black Figure"] = df_cleaned[stim_columns].sum(axis=1)
df_cleaned["Total White Figure"] = len(stim_columns) - df_cleaned["Total Black Figure"]

# Compute group-wise average figure-ground responses
group_summary = df_cleaned.groupby("Group")[["Total Black Figure", "Total White Figure"]].mean()

# --- Visualization: Group-Wise Proportion of Responses ---
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

# --- Chi-Square Test for Figure-Ground Responses ---
grouped_data = df_cleaned.groupby("Group").sum(numeric_only=True).T
summary = grouped_data.apply(lambda x: [x.sum(), len(x) * len(df_main.columns[2:]) - x.sum()], axis=0).T
summary.columns = ["Black Figure, White Background", "White Figure, Black Background"]

chi2, p, dof, expected = chi2_contingency(summary.values)

# Print Chi-Square Results
print("Observed Frequencies (Summary Table):")
print(summary)
print("\nChi-Square Statistic:", chi2)
print("p-value:", p)
print("Degrees of Freedom:", dof)

# --- LEFT/RIGHT RESPONSE ANALYSIS ---
# Remove any unnecessary index column from df_lr
df_lr = df_lr.drop(columns=['Unnamed: 0'], errors='ignore')

# Merge df_cleaned (main responses) with df_lr (left/right responses) based on "Participant"
df_combined = pd.merge(df_cleaned, df_lr, on="Participant", suffixes=('_figure', '_side'))

# Convert "Left"/"Right" responses to numerical values (1 = Right, 0 = Left)
for col in stim_columns:
    df_combined[col + "_side"] = df_combined[col + "_side"].map({"Right": 1, "Left": 0})

# Calculate the total number of "Right" choices per participant
df_combined["Total Right-Side Choices"] = df_combined[[col + "_side" for col in stim_columns]].sum(axis=1)

# Compute group-wise average of right-side choices
right_side_summary = df_combined.groupby("Group")["Total Right-Side Choices"].mean()

# --- Visualization: Right-Side Choices per Group ---
plt.figure(figsize=(8, 6))
right_side_summary.plot(kind="bar", color=["purple", "orange", "blue"], width=0.8)
plt.title("Average Right-Side Choices per Group")
plt.ylabel("Average Count")
plt.xlabel("Group")
plt.xticks(rotation=0)
plt.tight_layout()
plt.show()

# --- Chi-Square Test for Right-Side Preference ---
right_side_contingency = df_combined.groupby("Group")["Total Right-Side Choices"].sum().to_frame()
right_side_contingency["Total Left-Side Choices"] = (len(stim_columns) * df_combined.groupby("Group").size()) - right_side_contingency["Total Right-Side Choices"]

chi2_right, p_right, dof_right, expected_right = chi2_contingency(right_side_contingency.values)

# Print Chi-Square Test Results for Right-Side Preferences
print("\nRight-Side vs. Left-Side Choice Summary:")
print(right_side_contingency)
print("\nChi-Square Statistic for Right-Side Bias:", chi2_right)
print("p-value:", p_right)
print("Degrees of Freedom:", dof_right)

# --- Stacked Bar Chart: Distribution of Responses Across Groups (Per Stimulus) ---
# Reshape data for plotting
response_data = df_cleaned.melt(
    id_vars=["Participant", "Group"],
    var_name="Stimuli",
    value_name="Response"
)

# Summarize data: count responses per stimulus & group
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
