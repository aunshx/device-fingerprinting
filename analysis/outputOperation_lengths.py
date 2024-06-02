import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import json

# Load data from a JSON file (ensure the file contains 56,000 entries)
print("Loading data...")
with open('./data/formatted_data_for_dtw.json', 'r') as file:
    data = json.load(file)

# Convert the data into a pandas DataFrame
df = pd.DataFrame(data)

# Convert the operationOutput from JSON string to a list of floats
print("Converting operationOutput strings to lists...")
df['operationOutput'] = df['operationOutput'].apply(lambda x: np.fromstring(x.strip('[]'), sep=','))

# Calculate the lengths of the operationOutput arrays
print("Calculating the lengths of the operationOutput arrays...")
df['length'] = df['operationOutput'].apply(len)

# Plot the distribution of lengths
plt.figure(figsize=(12, 6))
sns.histplot(df['length'], bins=50, kde=True)
plt.title('Distribution of OperationOutput Lengths')
plt.xlabel('Length')
plt.ylabel('Frequency')
plt.grid(True)
plt.show()

# Calculate basic statistics
mean_length = df['length'].mean()
median_length = df['length'].median()
percentile_90 = np.percentile(df['length'], 90)
percentile_95 = np.percentile(df['length'], 95)

print(f"Mean length: {mean_length}")
print(f"Median length: {median_length}")
print(f"90th percentile length: {percentile_90}")
print(f"95th percentile length: {percentile_95}")
