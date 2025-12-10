import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import mean_absolute_error
import joblib  # This saves the trained model to a file

# 1. Load the Data
df = pd.read_csv("traffic_dataset.csv")

# 2. Prepare the Data
# X = The input features (Car_Count)
# y = The target we want to predict (Wait_Time)
X = df[['Car_Count']] 
y = df['Wait_Time']

# 3. Split into "Study" data and "Exam" data
# We train on 80% of the data and test on the unseen 20% to be fair.
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 4. Initialize the Model (The Decision Tree)
print("🧠 Training the model...")
model = DecisionTreeRegressor()

# 5. Train the Model (The "Study" Phase)
model.fit(X_train, y_train)

# 6. Test the Model (The "Exam" Phase)
predictions = model.predict(X_test)

# Calculate error (How wrong were we on average?)
error = mean_absolute_error(y_test, predictions)
print(f"✅ Model Trained!")
print(f"📉 Average Error: {error:.2f} seconds") 
# (If error is 0.0, it means it learned the rules perfectly!)

# 7. Save the Model
# We pack the trained model into a file so our App can use it later
joblib.dump(model, 'traffic_model.pkl')
print("💾 Model saved as 'traffic_model.pkl'")