import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans, AgglomerativeClustering
from sklearn.metrics import silhouette_score
from sklearn.decomposition import PCA

from scipy.cluster.hierarchy import dendrogram, linkage

# Load dataset
df = pd.read_csv("dataset/Mall_Customers.csv")

# Display first five rows
print("First five rows:")
print(df.head())

# Display dataset information
print("\nDataset Information:")
print(df.info())

# Display dataset shape
print("\nDataset Shape:")
print(df.shape)

# Check missing values
print("\nMissing Values:")
print(df.isnull().sum())

# Check duplicate records
print("\nDuplicate Records:")
print(df.duplicated().sum())

# Statistical summary
print("\nStatistical Summary:")
print(df.describe())

# Gender distribution
print("\nGender Distribution:")
print(df["Gender"].value_counts())

# Age distribution
plt.figure(figsize=(7, 4))
sns.histplot(df["Age"], kde=True)
plt.title("Age Distribution")
plt.savefig("outputs/age_distribution.png")
plt.show()

# Annual income distribution
plt.figure(figsize=(7, 4))
sns.histplot(df["Annual Income (k$)"], kde=True)
plt.title("Annual Income Distribution")
plt.savefig("outputs/income_distribution.png")
plt.show()

# Spending score distribution
plt.figure(figsize=(7, 4))
sns.histplot(df["Spending Score (1-100)"], kde=True)
plt.title("Spending Score Distribution")
plt.savefig("outputs/spending_distribution.png")
plt.show()

# Gender count
plt.figure(figsize=(6, 4))
sns.countplot(data=df, x="Gender")
plt.title("Gender Distribution")
plt.savefig("outputs/gender_distribution.png")
plt.show()

# Select relevant features
X = df[[
    "Age",
    "Annual Income (k$)",
    "Spending Score (1-100)"
]]

# Check selected features
print("\nSelected Features:")
print(X.head())

# Initialize scaler
scaler = StandardScaler()

# Scale the features
X_scaled = scaler.fit_transform(X)

print("\nScaled Data:")
print(X_scaled[:5])

wcss = []

for i in range(1, 11):

    kmeans = KMeans(
        n_clusters=i,
        init="k-means++",
        random_state=42,
        n_init=10
    )

    kmeans.fit(X_scaled)

    wcss.append(kmeans.inertia_)

# Plot Elbow graph
plt.figure(figsize=(8, 5))

plt.plot(range(1, 11), wcss, marker="o")

plt.title("Elbow Method")
plt.xlabel("Number of Clusters")
plt.ylabel("WCSS")

plt.savefig("outputs/elbow_method.png")
plt.show()

silhouette_scores = []

for i in range(2, 11):

    kmeans = KMeans(
        n_clusters=i,
        random_state=42,
        n_init=10
    )

    labels = kmeans.fit_predict(X_scaled)

    score = silhouette_score(X_scaled, labels)

    silhouette_scores.append(score)

    print("Clusters:", i, "Score:", score)

# Plot silhouette scores
plt.figure(figsize=(8, 5))

plt.plot(
    range(2, 11),
    silhouette_scores,
    marker="o"
)

plt.title("Silhouette Score Analysis")
plt.xlabel("Number of Clusters")
plt.ylabel("Silhouette Score")

plt.savefig("outputs/silhouette_score.png")
plt.show()

# Apply K-Means
kmeans = KMeans(
    n_clusters=5,
    init="k-means++",
    random_state=42,
    n_init=10
)

# Predict cluster labels
df["KMeans_Cluster"] = kmeans.fit_predict(X_scaled)

print("\nK-Means Cluster Results:")
print(df[[
    "Age",
    "Annual Income (k$)",
    "Spending Score (1-100)",
    "KMeans_Cluster"
]].head(10))

# Cluster count
print("\nNumber of Customers in Each Cluster:")
print(df["KMeans_Cluster"].value_counts().sort_index())

pca = PCA(n_components=2)

X_pca = pca.fit_transform(X_scaled)

plt.figure(figsize=(9, 6))

sns.scatterplot(
    x=X_pca[:, 0],
    y=X_pca[:, 1],
    hue=df["KMeans_Cluster"],
    palette="viridis",
    s=80
)

plt.title("Customer Segmentation Using K-Means")
plt.xlabel("Principal Component 1")
plt.ylabel("Principal Component 2")

plt.savefig("outputs/kmeans_clusters.png")
plt.show()

# Perform hierarchical linkage
linked = linkage(X_scaled, method="ward")

# Plot dendrogram
plt.figure(figsize=(12, 6))

dendrogram(
    linked,
    truncate_mode="lastp",
    p=30
)

plt.title("Hierarchical Clustering Dendrogram")
plt.xlabel("Customer Groups")
plt.ylabel("Distance")

plt.savefig("outputs/dendrogram.png")
plt.show()

# Apply Hierarchical Clustering
hc = AgglomerativeClustering(
    n_clusters=5,
    linkage="ward"
)

# Generate cluster labels
df["Hierarchical_Cluster"] = hc.fit_predict(X_scaled)

print("\nHierarchical Clustering Results:")
print(df[[
    "Age",
    "Annual Income (k$)",
    "Spending Score (1-100)",
    "Hierarchical_Cluster"
]].head(10))

plt.figure(figsize=(9, 6))

sns.scatterplot(
    x=X_pca[:, 0],
    y=X_pca[:, 1],
    hue=df["Hierarchical_Cluster"],
    palette="Set2",
    s=80
)

plt.title("Hierarchical Clustering Results")
plt.xlabel("Principal Component 1")
plt.ylabel("Principal Component 2")

plt.savefig("outputs/hierarchical_clusters.png")
plt.show()

# Evaluate K-Means
kmeans_score = silhouette_score(
    X_scaled,
    df["KMeans_Cluster"]
)

# Evaluate Hierarchical Clustering
hc_score = silhouette_score(
    X_scaled,
    df["Hierarchical_Cluster"]
)

print("\nModel Comparison")

print("K-Means Silhouette Score:", kmeans_score)

print("Hierarchical Silhouette Score:", hc_score)

if kmeans_score > hc_score:
    print("K-Means gives better-separated clusters.")

elif hc_score > kmeans_score:
    print("Hierarchical Clustering gives better-separated clusters.")

else:
    print("Both algorithms have equal Silhouette Scores.")

# Group customers by K-Means cluster
cluster_analysis = df.groupby("KMeans_Cluster").agg({
    "Age": "mean",
    "Annual Income (k$)": "mean",
    "Spending Score (1-100)": "mean",
    "CustomerID": "count"
})

cluster_analysis.rename(
    columns={"CustomerID": "Customer_Count"},
    inplace=True
)

print("\nCustomer Segment Analysis:")
print(cluster_analysis.round(2))

cluster_analysis[
    ["Age", "Annual Income (k$)", "Spending Score (1-100)"]
].plot(
    kind="bar",
    figsize=(12, 6)
)

plt.title("Customer Segment Comparison")
plt.xlabel("Cluster")
plt.ylabel("Average Value")
plt.xticks(rotation=0)
plt.legend()

plt.savefig("outputs/cluster_comparison.png")
plt.show()

df.to_csv(
    "outputs/customer_segmentation_results.csv",
    index=False
)

print("\nCustomer segmentation results saved successfully!")