import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

# Load the dataset
data = {
    "algorithm": ["sha224", "sha512", "md5", "blake2b", "sha1", "sha256"],
    "avg_hamming_distance": [52.5004024, 119.9983504, 30.0015976, 120.0027828, 37.50467467, 60.00387788],
    "std_hamming_distance": [1.811452234, 2.744986016, 1.367919828, 2.736565193, 1.529207195, 1.941630476]
}

df = pd.DataFrame(data)

# Set up the bar width and positions
bar_width = 0.35
index = np.arange(len(df["algorithm"]))

# Create the multi-bar plot
fig, ax = plt.subplots(figsize=(12, 6))

bar1 = ax.bar(index, df["avg_hamming_distance"], bar_width, label='Average Hamming Distance', color='b', alpha=0.7)
bar2 = ax.bar(index + bar_width, df["std_hamming_distance"], bar_width, label='Standard Deviation of Hamming Distance', color='r', alpha=0.7)

# Add labels, title, and legend
ax.set_xlabel('Hashing Algorithm')
ax.set_ylabel('Distance')
ax.set_title('Comparison of Average Hamming Distance and Std Dev of Hamming Distance Across Hashing Algorithms')
ax.set_xticks(index + bar_width / 2)
ax.set_xticklabels(df["algorithm"])
ax.legend()

# Display the plot
plt.tight_layout()
plt.show()
