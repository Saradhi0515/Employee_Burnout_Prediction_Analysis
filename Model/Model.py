from sklearn.ensemble import RandomForestClassifier
from sklearn.datasets import load_iris
import pickle

# Load dataset
data = load_iris()
X, y = data.data, data.target


# Train a Random Forest model
model = RandomForestClassifier()
model.fit(X, y)

# Save the model to a file
with open('X:\Employee_Burnout_Prediction_Analysis\Model\model.pkl', 'wb') as file:
    pickle.dump(model, file)