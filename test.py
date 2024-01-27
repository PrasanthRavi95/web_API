from sklearn.metrics import accuracy_score
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
import pickle
import pandas as pd

# Creating a DataFrame with the provided data
data = {
    'income': [50000, 60000, 75000, 30000, 45000, 55000, 70000, 40000],
    'credit_score': [700, 720, 800, 620, 650, 680, 750, 630],
    'loan_amount': [5000, 6000, 8000, 3000, 4000, 5500, 7000, 3500],
    'loan_approval': [1, 1, 1, 0, 0, 1, 1, 0]
}

data = pd.DataFrame(data)


# Assuming you have a dataset with features (income, credit_score, loan_amount) and labels (loan_approval)
# Replace 'your_dataset.csv' with the actual file name or provide your own dataset


# Assume 'loan_approval' is the target variable
X = data[['income', 'credit_score', 'loan_amount']]
y = data['loan_approval']

# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42)

# Train a Logistic Regression model
model = LogisticRegression()
model.fit(X_train, y_train)

# Evaluate the model on the test set
y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print(f'Model Accuracy: {accuracy}')

# Save the trained model as a pickle file
# with open('loan_approval_model.pkl', 'wb') as model_file:
#    pickle.dump(model, model_file)
# Example input features for prediction
input_features = [[10, 700, 50000]]  # Adjust these values as needed

# Make predictions
predictions = model.predict(input_features)

# Print the prediction
print("Loan Approval Prediction:",
      "Approved" if predictions[0] == 1 else "Not Approved")
