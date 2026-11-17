import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score
import pickle
import os

print("--- Starting Model Training ---")

# Step 1: Load the datasets
train_df = pd.read_csv('Training.csv')
test_df = pd.read_csv('Testing.csv')
print("✅ Data loaded successfully!")

# Step 2: Validate the symptom data
if 'Unnamed: 133' in train_df.columns:
    train_df = train_df.drop('Unnamed: 133', axis=1)
    print("✅ Dropped 'Unnamed: 133' column from training data.")

for dataset_name, dataset in [('training', train_df), ('testing', test_df)]:
    feature_data = dataset.drop('prognosis', axis=1)
    if feature_data.isna().any().any():
        raise ValueError(
            f"{dataset_name.title()} data contains unknown symptom values. "
            "The current model requires explicit 0/1 values; do not treat unknown as absent."
        )
    if not feature_data.isin([0, 1]).all().all():
        raise ValueError(f"{dataset_name.title()} symptom values must be 0 or 1.")

print("✅ Symptom data validated as explicit 0/1 values.")

# Step 3: Encode labels (convert categorical target variable to numbers)
le = LabelEncoder()
train_df['prognosis'] = le.fit_transform(train_df['prognosis'])
test_df['prognosis'] = le.transform(test_df['prognosis'])
print("✅ Encoded target labels successfully.")

# Step 4: Split into features (X) and labels (y)
X_train = train_df.drop('prognosis', axis=1)
y_train = train_df['prognosis']
X_test = test_df.drop('prognosis', axis=1)
y_test = test_df['prognosis']

print(f"📊 Training model with {len(X_train.columns)} features and {len(X_train)} samples.")

# Step 5: Train the model
rf_model = RandomForestClassifier(n_estimators=100, random_state=42)
rf_model.fit(X_train, y_train)
print("✅ Model training complete!")

# Step 6: Make predictions and evaluate performance
y_pred = rf_model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print(f"✅ Model Accuracy: {accuracy * 100:.2f}%")

# Step 7: Save model and label encoder
if not os.path.exists('models'):
    os.makedirs('models')

with open('models/disease_predictor.pkl', 'wb') as file:
    pickle.dump(rf_model, file)

with open('models/label_encoder.pkl', 'wb') as file:
    pickle.dump(le, file)

