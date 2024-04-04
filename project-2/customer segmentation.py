import pandas as pd
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score

# Step 1: Data Loading
data = pd.read_csv("marketing_data.csv")  
print(data.head())  

# Step 2: Data Exploration and Cleaning
print(data.isnull().sum())

# Handle missing values if any
data.dropna(inplace=True)

# Step 3: Feature Selection
features = ['MntWines', 'MntFruits', 'MntMeatProducts', 'MntFishProducts', 'MntSweetProducts', 'MntGoldProds']

# Step 4: Feature Scaling
scaler = StandardScaler()
scaled_data = scaler.fit_transform(data[features])

# Step 5: Choosing the Number of Clusters (K)
# Use the Elbow Method to determine the optimal number of clusters
inertia = []
silhouette_scores = []
for k in range(2, 11):
    kmeans = KMeans(n_clusters=k, random_state=42)
    kmeans.fit(scaled_data)
    inertia.append(kmeans.inertia_)
    silhouette_scores.append(silhouette_score(scaled_data, kmeans.labels_))

# Plotting the Elbow Method
plt.figure(figsize=(10, 6))
plt.plot(range(2, 11), inertia, marker='o')
plt.title('Elbow Method')
plt.xlabel('Number of Clusters (K)')
plt.ylabel('Inertia')
plt.show()

# Plotting Silhouette Score
plt.figure(figsize=(10, 6))
plt.plot(range(2, 11), silhouette_scores, marker='o')
plt.title('Silhouette Score')
plt.xlabel('Number of Clusters (K)')
plt.ylabel('Silhouette Score')
plt.show()

# Step 6: Model Training and Prediction
optimal_k = 3  

# Train the K-means model with the optimal number of clusters
kmeans = KMeans(n_clusters=optimal_k, random_state=42)
kmeans.fit(scaled_data)

# Step 7: Visualization
# Visualize the clusters
data['Cluster'] = kmeans.labels_
plt.figure(figsize=(10, 6))
for i in range(optimal_k):
    cluster_data = data[data['Cluster'] == i]
    plt.scatter(cluster_data['MntWines'], cluster_data['MntFruits'], label=f'Cluster {i+1}')
plt.scatter(kmeans.cluster_centers_[:, 0], kmeans.cluster_centers_[:, 1], marker='*', s=200, c='black', label='Centroids')
plt.title('Customer Segmentation')
plt.xlabel('Spending on Wines')
plt.ylabel('Spending on Fruits')
plt.legend()
plt.show()

# Step 8: Insights and Recommendations
# Analyze characteristics of each cluster and provide insights and recommendations
cluster_means = data.groupby('Cluster')[features].mean()
print("Cluster Means:")
print(cluster_means)

import warnings
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.simplefilter(action='ignore', category=FutureWarning)