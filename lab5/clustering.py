# Import of libraries
import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import matplotlib.patches as mpatches
from matplotlib import cm
from matplotlib.colors import ListedColormap

from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score
from sklearn.utils import check_random_state

from google.colab import drive
drive.mount('/content/drive')

cwd = 'drive/MyDrive/...' # Set your current working directory where the csv file is located

# Check if file exists
file_path = cwd + '/air_quality.csv'
if not os.path.exists(file_path):
    raise FileNotFoundError(f"The CSV file was not found at the path: {file_path}")

# Load dataset (first 1,000,000 rows for testing)
df = pd.read_csv(file_path, low_memory=False, na_values=['-', 'NA', 'n/a', 'ND',''], nrows=1000000)

# Let's check that the column type has been read correctly
print(df.dtypes)

# display the first 5 rows
df.head()

# Let's remove the unnecessary columns
df = df.drop(columns=["sitename", "county", 'aqi',"siteid","pollutant","date"])

#  Drops any column that contains only missing values
df = df.dropna(axis=1, how="all")

# Drops any row that has at least one missing value in any remaining column
df = df.dropna(axis=0, how="any")

# Show the distinct values of the 'status' column
print(df['status'].unique())

# Show number of distinct values of the 'status' column
print("Different values:", df['status'].nunique())

# Select only the numerical columns for the scaler
df_numeric = df.select_dtypes(include=[np.number])

# Fit the scaler on the full numeric data and transform it to a NumPy array
scaler = StandardScaler()
df_scaled = scaler.fit_transform(df_numeric)

# Silhouette-based model selection for K (number of clusters)
scores = {}

# Try different k values and compute the silhouette score on the sampled subset
for k in range(2, 6):
    kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
    labels = kmeans.fit_predict(df_scaled)
    score = silhouette_score()
    scores[k] = score
    print(f"Silhouette score for k={k}: {score:.4f}")

# Pick the k with the highest silhouette score
best_k = max(scores, key=scores.get)
print(f"\nAccording to the silhouette analysis, the optimal number of clusters would be: k={best_k}")

# Count how many unique labels/classes are present in 'status'
num_clusters = df['status'].nunique()

# Create a KMeans model with k = num_clusters clusters.
kmeans = KMeans(n_clusters=num_clusters, random_state=42)
# Fit the model on the standardized numeric features and assign each sample to its nearest centroid.
labels = kmeans.fit_predict(df_scaled)

# Build a contingency table between the true labels ('status')
# and the unsupervised cluster assignments 'labels'.
ct = pd.crosstab(df['status'], labels, colnames=['cluster'])
print("Contingency: status vs cluster")
display(ct)

# For each cluster, pick the most frequent 'status'.
# This yields a majority-vote mapping: cluster_id --> predicted status label.
cluster_to_status = ct.idxmax(axis=0).to_dict()
print("Mapping cluster --> status:", cluster_to_status)

# Convert each sample's cluster id into a predicted status via the majority-vote mapping.
# Align the Series index with df to keep row order consistent.
pred_status = pd.Series(labels, index=df.index).map(cluster_to_status)

# Compare predicted status vs. true status, elementwise; boolean Series of correct predictions.
is_correct = pred_status.eq(df['status'])

# Summarize performance per true status:
#  number of correct predictions
#  total samples
summary = (
    pd.DataFrame({'status': df['status'], 'correct': is_correct})
      .groupby('status')['correct']
      .agg(correct='sum', tot='count')
)

# Add errors and per-class accuracy (%).
summary['incorrect'] = summary['tot'] - summary['correct']
summary['accuracy_%'] = (summary['correct'] / summary['tot'] * 100).round(2)

print("\nSummary by status (correct/incorrect/accuracy):")
display(summary.sort_values('accuracy_%', ascending=False))

# Overall accuracy = mean correctness across all samples.
overall_acc = is_correct.mean()
print(f"Global accuracy (cluster --> status): {overall_acc*100:.2f}%")

# 2D visualization with PCA
# Fit PCA on the scaled features and project data to 2 principal components
pca = PCA(n_components=2)
reduced = pca.fit_transform(df_scaled)

# Select the base colormap "tab10"
base_cmap = plt.colormaps["tab10"]

# Take only the first 'num_clusters' colors
colors = base_cmap.colors[:num_clusters]

# Create a discrete colormap with these selected colors
cmap = ListedColormap(colors)

# Scatter plot of the 2D projection, colored by KMeans cluster labels
plt.figure(figsize=(8,6))
sc1 = plt.scatter(reduced[:,0], reduced[:,1], c=labels, cmap=cmap, alpha=0.6, s=12)

# Get the unique cluster IDs found by KMeans
unique_clusters = np.unique(labels)

# Create a colored patch for each cluster
colors = [cmap(i) for i in range(num_clusters)]
patches = [mpatches.Patch(color=colors[i], label=f'Cluster {cl}')
           for i, cl in enumerate(unique_clusters)]

# Add the legend to the plot
plt.legend(handles=patches, title="Labels", loc='lower left', frameon=True)

# Compute cluster sizes to display in the title
sizes = np.bincount(labels)
sizes_txt = ", ".join(f"{i}:{sizes[i]}" for i in range(len(sizes)))

# Title with k, variance explained by PCs, and cluster sizes
plt.title(f'KMeans — PCA 2D (k={num_clusters}) | size [{sizes_txt}]')

plt.tight_layout()
plt.show()

status_labels = df['status'].replace({'Moderate': 0, 'Good': 1, 'Unhealthy for Sensitive Groups': 2, 'Unhealthy': 3, 'Very Unhealthy': 4}, inplace=False).infer_objects(copy=False)

# Scatter plot of the 2D projection, colored by KMeans cluster labels
plt.figure(figsize=(8,6))
sc1 = plt.scatter(reduced[:,0], reduced[:,1], c=status_labels, cmap=cmap, alpha=0.6, s=12)

# Create a colored patch for each cluster
colors = [cmap(i) for i in range(num_clusters)]
patches = [mpatches.Patch(color=colors[i], label=cl)
           for i, cl in enumerate(df['status'].unique())]

# Add the legend to the plot
plt.legend(handles=patches, title="Labels", loc='lower left', frameon=True)

# Title with k, variance explained by PCs, and cluster sizes
plt.title(f'Labels distribution')

plt.tight_layout()
plt.show()

# Color points by correctness: 'orange' if cluster --> status mapping matches the true status, else 'red'
colors = np.where(is_correct.values, 'orange', 'red')

#  2D PCA scatter colored by match/mismatch
plt.figure(figsize=(8,6))
plt.scatter(reduced[:,0], reduced[:,1], c=colors, alpha=0.6, s=12)

# Legend using colored patches
patches = [
    mpatches.Patch(color='orange', label='Match cluster=status'),
    mpatches.Patch(color='red', label='Mismatch')
]
plt.legend(handles=patches, title="Labels", loc='lower left', frameon=True)

# Title shows overall accuracy computed earlier; axis labels include variance explained by PCs
plt.title(f'PCA comparison — Match (orange) vs Mismatch (red) | Acc {overall_acc*100:.1f}%')

plt.tight_layout()
plt.show()

# Attach cluster assignments to the original data for interpretation
df_with_clusters = df.copy()
df_with_clusters["Cluster"] = labels

# Compute per-cluster averages
# This summarizes each cluster's centroid in the original feature space,
# which is easier to interpret than scaled values.
cluster_summary = df_with_clusters.groupby('Cluster').mean(numeric_only=True)

# Display the table
display(cluster_summary)

