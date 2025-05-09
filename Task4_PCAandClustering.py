import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans, DBSCAN

# -------------------------------
# 1. Load and preprocess data
# -------------------------------
df = pd.read_csv("ResearchOutputs.csv", low_memory=False)
# df = pd.read_csv("fused_outputs.csv", low_memory=False)
df.columns = [col.strip().lower() for col in df.columns]

features = ['projectstatus', 'projectrdc', 'outputtype', 'projectstartyear', 'projectendyear', 'outputyear']
df = df[df['outputtitle'].notna()]
df = df[features + ['outputtitle']].copy()

# Fill missing values
df['projectstatus'] = df['projectstatus'].fillna('Unknown')
df['projectrdc'] = df['projectrdc'].fillna('Unknown')
df['outputtype'] = df['outputtype'].fillna('Unknown')
for col in ['projectstartyear', 'projectendyear', 'outputyear']:
    df[col] = df[col].fillna(df[col].median())

# One-hot encoding
df_encoded = pd.get_dummies(df[['projectstatus', 'projectrdc', 'outputtype']], drop_first=True)

# TF-IDF on outputtitle → reduce with PCA
vectorizer = TfidfVectorizer(stop_words='english', max_features=200)
X_title_tfidf = vectorizer.fit_transform(df['outputtitle'].astype(str))
title_pca = PCA(n_components=3)
X_title_pca = title_pca.fit_transform(X_title_tfidf.toarray())

# Combine structured + title features
X_combined = pd.concat([
    df[['projectstartyear', 'projectendyear', 'outputyear']].reset_index(drop=True),
    df_encoded.reset_index(drop=True),
    pd.DataFrame(X_title_pca, columns=['title_pc1', 'title_pc2', 'title_pc3'])
], axis=1)

# Normalize
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X_combined)

# -------------------------------
# 2. PCA for visualization
# -------------------------------
pca = PCA(n_components=2)
X_pca = pca.fit_transform(X_scaled)

plt.figure(figsize=(8, 5))
plt.bar(range(1, 3), pca.explained_variance_ratio_, alpha=0.7)
plt.xlabel('Principal Component')
plt.ylabel('Explained Variance Ratio')
plt.title('PCA - Combined Features')
plt.tight_layout()
plt.show()

# Top contributing features per PC
components = pca.components_
feature_names = X_combined.columns
for i, comp in enumerate(components):
    top_indices = np.argsort(np.abs(comp))[::-1][:3]
    print(f"\nPC{i+1} explains {pca.explained_variance_ratio_[i]:.2%}")
    for idx in top_indices:
        print(f"  {feature_names[idx]:<25} weight={comp[idx]:.4f}")

# -------------------------------
# 3. KMeans clustering (optional)
# -------------------------------
k = 2
kmeans = KMeans(n_clusters=k, random_state=42)
cluster_labels_kmeans = kmeans.fit_predict(X_pca)

plt.figure(figsize=(8, 6))
sns.scatterplot(x=X_pca[:, 0], y=X_pca[:, 1], hue=cluster_labels_kmeans, palette='tab10')
plt.title(f'KMeans Clustering (k={k}) on PCA-Reduced Data')
plt.xlabel('PC1')
plt.ylabel('PC2')
plt.tight_layout()
plt.show()

# -------------------------------
# 4. DBSCAN clustering
# -------------------------------
dbscan = DBSCAN(eps=0.2, min_samples=5)
cluster_labels = dbscan.fit_predict(X_pca)

plt.figure(figsize=(8, 6))
sns.scatterplot(x=X_pca[:, 0], y=X_pca[:, 1], hue=cluster_labels, palette='tab10')
plt.title('DBSCAN Clustering on PCA-Reduced Data')
plt.xlabel('PC1')
plt.ylabel('PC2')
plt.tight_layout()
plt.show()

unique, counts = np.unique(cluster_labels, return_counts=True)
print("\nCluster label distribution:")
for u, c in zip(unique, counts):
    label = f"Noise (-1)" if u == -1 else f"Cluster {u}"
    print(f"{label}: {c} samples")

# -------------------------------
# 5. Analyze Cluster 2 and 5 //will be different when run several times
# -------------------------------
df_clusters = df.reset_index(drop=True).copy()
df_clusters['cluster'] = cluster_labels
df_c2 = df_clusters[df_clusters['cluster'] == 2]
df_c5 = df_clusters[df_clusters['cluster'] == 5]

def analyze_cluster(df_subset, name):
    print(f"\n=== {name} ===")
    print("Output Type:")
    print(df_subset['outputtype'].value_counts(), "\n")
    print("Project RDC:")
    print(df_subset['projectrdc'].value_counts(), "\n")
    print("Mean of Years:")
    print(df_subset[['projectstartyear', 'projectendyear', 'outputyear']].mean(), "\n")

    # Year histogram
    df_subset['outputyear'].hist(bins=15, alpha=0.7)
    plt.title(f"{name} - Output Year Distribution")
    plt.xlabel("Output Year")
    plt.ylabel("Frequency")
    plt.show()

    # Top words
    print("Top words in outputtitle:")
    vec = CountVectorizer(stop_words='english', max_features=15)
    vec.fit(df_subset['outputtitle'].astype(str))
    print(vec.get_feature_names_out(), "\n")

analyze_cluster(df_c2, "Cluster 2")
analyze_cluster(df_c5, "Cluster 5")
