import pandas as pd
import matplotlib.pyplot as plt

# 1. Load the Data
# We use Pandas to read the CSV file into a "DataFrame" (like a super-powered Excel sheet)
df = pd.read_csv("traffic_dataset.csv")

# 2. Peek at the data
print("--- First 5 Rows of Data ---")
print(df.head()) # Shows the top 5 rows

print("\n--- Basic Statistics ---")
print(df.describe()) # Shows count, mean, max, min, etc.

# 3. Visualize the Relationship
# We want to see if "Car Count" affects "Wait Time"
plt.figure(figsize=(10, 6))

# Create a Scatter Plot (Dots on a graph)
# X-axis = How many cars
# Y-axis = How long the wait is
plt.scatter(df['Car_Count'], df['Wait_Time'], color='blue', alpha=0.5)

plt.title("Traffic Analysis: Car Count vs. Wait Time")
plt.xlabel("Number of Cars Detected")
plt.ylabel("Estimated Wait Time (seconds)")
plt.grid(True)

print("\n📊 Displaying Graph... (Check the popup window)")
plt.show()