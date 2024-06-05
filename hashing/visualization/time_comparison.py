import pandas as pd
import matplotlib.pyplot as plt

# Create a DataFrame with the compute time values
data = {
    'Algorithm': ['sha224', 'sha512', 'md5', 'blake2b', 'sha1', 'sha256'],
    'Compute Time (ms)': [526.1334, 570.8342, 454.8989, 578.812, 449.7988, 588.8356]
}

df = pd.DataFrame(data)

# Plot the line graph
plt.figure(figsize=(10, 6))
plt.plot(df['Algorithm'], df['Compute Time (ms)'], marker='o', linestyle='-', color='b')

# Add labels and title
plt.xlabel('Algorithm')
plt.ylabel('Compute Time (ms)')
plt.title('Algorithm vs Compute Time')
plt.grid(True)
plt.xticks(rotation=45)
plt.ylim(0, df['Compute Time (ms)'].max() + 50)  # Start y-axis from 0

# Save the plot as an image file
plt.savefig('algorithm_vs_compute_time_line.png')

# Show the plot
plt.show()
