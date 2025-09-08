import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
import joblib

# Create test data with meaningful column names
data = pd.DataFrame({
    'age': [25, 30, 35, 40, 45, 50, 55, 60, 28, 32, 38, 42],
    'income': [30000, 50000, 70000, 90000, 110000, 130000, 150000, 170000, 35000, 55000, 75000, 95000],
    'category': ['A', 'B', 'A', 'C', 'B', 'A', 'C', 'B', 'A', 'B', 'C', 'A'],
    'score': [0.2, 0.8, 0.6, 0.9, 0.7, 0.8, 0.9, 0.7, 0.3, 0.6, 0.8, 0.5]
})
target = [0, 1, 0, 1, 1, 1, 1, 0, 0, 1, 1, 0]

# Encode categorical data
le = LabelEncoder()
data['category_encoded'] = le.fit_transform(data['category'])
data = data.drop('category', axis=1)

# Train model
X_train, X_test, y_train, y_test = train_test_split(data, target, test_size=0.5, random_state=42)
model = RandomForestClassifier(random_state=42)
model.fit(X_train, y_train)

# Save model
joblib.dump(model, 'test_model_with_features.pkl')

# Save test dataset
test_df = X_test.copy()
test_df['target'] = y_test
test_df.to_csv('test_dataset_with_target.csv', index=False)

print('Created test model and dataset with meaningful feature names!')
print(f'Model features: {list(X_train.columns)}')
print(f'Test dataset shape: {test_df.shape}')
print(f'Model accuracy: {model.score(X_test, y_test):.3f}')
