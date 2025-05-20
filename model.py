import pandas as pd
import numpy as np
import re
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.impute import SimpleImputer
import datetime

"""# Loading the data"""

data = pd.read_csv('Final_insurance_fraud.csv')
target_column = 'Fraud_Label'
description_column = 'Claim_Description'

feature_columns = [col for col in data.columns if col != target_column]  # Listing all feature columns (excluding the target column)

"""
# Claim Validation using Simple NLP"""

def validate_claim(description):
    if not isinstance(description, str) or len(description.strip()) == 0:
        return False, "Empty description"

    words = description.split()
    if len(words) < 10:
        return False, "Description too short"

    if not re.search(r'\d+', description):
        return False, "No numerical data found"

    return True, "Valid claim"

if description_column in data.columns:
    data['is_valid_claim'], data['validation_message'] = zip(*data[description_column].apply(validate_claim))
else:
    data['is_valid_claim'] = True
    data['validation_message'] = "No description to validate"

valid_data = data[data['is_valid_claim'] == True].copy()
print(f"Number of valid claims after filtering: {len(valid_data)}")

if valid_data.empty:
    print("Warning: No valid claims after filtering. Reverting to full dataset.")
    valid_data = data.copy()

valid_data = valid_data.dropna(subset=[target_column])
print(f"Number of samples after dropping missing targets: {len(valid_data)}")

sns.countplot(x=data['Fraud_Label'], palette='coolwarm')
plt.title('Fraud vs. Non-Fraud Claims')
plt.xlabel('Fraud Reported (0 = No, 1 = Yes)')
plt.ylabel('Count')
plt.show()

"""# Feature Engineering & Encoding
### For categorical features (excluding text description) using Label Encoding
"""

for col in feature_columns:
    if col != description_column and valid_data[col].dtype == 'object':
        le = LabelEncoder()
        valid_data[col] = le.fit_transform(valid_data[col].astype(str))

if description_column in valid_data.columns:
    model_features = [col for col in feature_columns if col != description_column]
else:
    model_features = feature_columns

if 'Claim_ID' in model_features:
    model_features.remove('Claim_ID')

X = valid_data[model_features]
y = valid_data[target_column]

"""# Handling Missing Values and Splitting Data
### Impute missing values in features: use mean imputation since features are now numeric.
"""

imputer = SimpleImputer(strategy='mean')
X_imputed = imputer.fit_transform(X)
X = pd.DataFrame(X_imputed, columns=X.columns)

if len(X) < 5:
    raise ValueError("Not enough samples for splitting the data.")

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

model = LogisticRegression(random_state=42, max_iter=1000)
model.fit(X_train_scaled, y_train)

"""# Evaluation and Flagging for Human Review
### Predict and get probabilities for the test set
"""

y_pred = model.predict(X_test_scaled)
y_prob = model.predict_proba(X_test_scaled)

accuracy = accuracy_score(y_test, y_pred)
conf_matrix = confusion_matrix(y_test, y_pred)
class_report = classification_report(y_test, y_pred)

print("\nAccuracy:", accuracy)
print("\nConfusion Matrix:")
print(conf_matrix)
print("\nClassification Report:")
print(class_report)

# Visualize the confusion matrix
plt.figure(figsize=(6, 4))
sns.heatmap(conf_matrix, annot=True, fmt="d", cmap="Blues")
plt.title("Confusion Matrix")
plt.ylabel("Actual")
plt.xlabel("Predicted")
plt.show()

"""
# Flag cases for human review:
## Flag cases with predicted fraud probability near 0.5 (e.g., between 0.4 and 0.6)"""

review_indices = []
threshold_lower, threshold_upper = 0.4, 0.6
for idx, prob in enumerate(y_prob):
    if threshold_lower < prob[1] < threshold_upper:
        review_indices.append(idx)

print(f"\nNumber of cases flagged for human review: {len(review_indices)}")

def generate_summary(claim_row):
    """
    Generates a simple summary for a claim.
    Adjust this function to integrate with a generative AI for more advanced summaries.
    """
    summary = f"Claim ID: {claim_row.get('Claim_ID', 'N/A')}\n"
    summary += f"Claim Amount: {claim_row.get('Claim_Amount', 'N/A')}\n"
    summary += f"Description: {claim_row.get('Claim_Description', 'N/A')[:100]}...\n"
    summary += f"Predicted Fraud Probability: {claim_row.get('fraud_prob', 'N/A'):.2f}\n"
    summary += f"Validation: {claim_row.get('validation_message', 'N/A')}\n"
    return summary

# Generate sample summaries for flagged cases if the claim description column exists
flagged_summaries = []
if description_column in data.columns:
    test_indices = X_test.index.tolist() # Reset the test set index to map back to original data rows
    for idx in review_indices:
        orig_idx = test_indices[idx]
        claim_details = valid_data.iloc[orig_idx].to_dict()
        claim_details['fraud_prob'] = y_prob[idx][1]
        summary = generate_summary(claim_details)
        flagged_summaries.append(summary)

    print("\nSample Summaries for Human Review:")
    for summ in flagged_summaries[:3]:
        print("-" * 40)
        print(summ)
else:
    print("\nNo claim description column available to generate summaries.")

flagged_summaries = []
if description_column in data.columns:
    test_indices = X_test.index.tolist() # Reset the test set index to map back to original data rows
    for idx in review_indices:
        orig_idx = test_indices[idx]
        claim_details = valid_data.iloc[orig_idx].to_dict()
        claim_details['fraud_prob'] = y_prob[idx][1]
        summary = generate_summary(claim_details)
        flagged_summaries.append(summary)

    print("\nSample Summaries for Human Review:")
    for summ in flagged_summaries[:3]:
        print("-" * 40)
        print(summ)
else:
    print("\nNo claim description column available to generate summaries.")


