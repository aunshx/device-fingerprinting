import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
import json
import matplotlib.pyplot as plt

# Load data from a JSON file
print("Loading data...")
with open('./data/formatted_data_for_dtw.json', 'r') as file:
    data = json.load(file)

# get only 1000 entries of each platform
new_data = []
for platform in ['Linux', 'Windows', 'iOS', 'macOS']:
    platform_data = [item for item in data if item["platform"] == platform]
    new_data.extend(platform_data[:1000])

data = new_data

# Convert the data to a DataFrame
df = pd.DataFrame(data)

# Define the fixed length for padding/truncation
fixed_length = 263

def pad_or_truncate(arr, length):
    if len(arr) > length:
        return arr[:length]
    elif len(arr) < length:
        return np.pad(arr, (0, length - len(arr)), 'constant')
    return arr

# Apply padding/truncation to each operationOutput
df['operationOutput'] = df['operationOutput'].apply(lambda x: pad_or_truncate(x, fixed_length))

# Create feature matrix
print("Creating feature matrix...")
X = np.vstack(df['operationOutput'].values)

# Standardize features
scaler = StandardScaler()
X = scaler.fit_transform(X)

# Perform K-Means clustering
print("Performing K-Means clustering...")
kmeans = KMeans(n_clusters=3, random_state=42)
clusters = kmeans.fit_predict(X)

# Add cluster labels to the DataFrame
df['cluster'] = clusters

# Visualize the clusters using PCA
print("Visualizing clusters...")
pca = PCA(n_components=2)
X_pca = pca.fit_transform(X)

plt.figure(figsize=(10, 8))
plt.scatter(X_pca[:, 0], X_pca[:, 1], c=clusters, cmap='viridis')
plt.xlabel('PCA Component 1')
plt.ylabel('PCA Component 2')
plt.title('K-Means Clusters (PCA Projection)')
plt.colorbar(label='Cluster Label')
plt.show()

# # Calculate the distribution of platforms within each cluster
# cluster_counts = df.groupby(['cluster', 'platform']).size().unstack(fill_value=0)
# cluster_proportions = cluster_counts.div(cluster_counts.sum(axis=1), axis=0)

# # Visualize the distribution of platforms within each cluster
# print("Visualizing platform distribution within clusters...")
# cluster_proportions.plot(kind='bar', stacked=True, figsize=(10, 8))
# plt.xlabel('Cluster')
# plt.ylabel('Proportion')
# plt.title('Platform Distribution within Clusters')
# plt.legend(title='Platform')
# plt.show()