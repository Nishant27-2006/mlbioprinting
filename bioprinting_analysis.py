
import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
import seaborn as sns
from skimage import io
from skimage.transform import resize

# Load images from the data folder
image_paths = ["data/image1.tif", "data/image2.tif", "data/image3.tif", ...]  # Add all image paths
target_size = (256, 256)  # Standard size for resizing

# Preprocess images
preprocessed_images = []
for image_path in image_paths:
    image = io.imread(image_path)
    resized_image = resize(image, target_size, anti_aliasing=True)
    preprocessed_images.append(resized_image.flatten())

# Convert list of images to numpy array for analysis
image_array = np.array(preprocessed_images)

# 1. PCA Analysis
pca = PCA(n_components=2)
pca_result = pca.fit_transform(image_array)

# 2. K-Means Clustering
kmeans = KMeans(n_clusters=3, random_state=42)
kmeans_result = kmeans.fit_predict(image_array)

# 3. Pairwise Correlation
correlation_matrix = np.corrcoef(image_array)

# Visualizations
# Pairwise Correlation Heatmap
plt.figure(figsize=(8, 6))
sns.heatmap(correlation_matrix, annot=False, cmap='coolwarm')
plt.title('Pairwise Correlation of Image Intensities')
plt.show()

# PCA 2D Plot
plt.figure(figsize=(8, 6))
plt.scatter(pca_result[:, 0], pca_result[:, 1], c=kmeans_result, cmap='viridis')
plt.title('PCA 2D Projection with K-Means Clusters')
plt.xlabel('Principal Component 1')
plt.ylabel('Principal Component 2')
plt.show()

# Elbow Method for K-Means
inertia = []
k_values = list(range(1, 6))
for k in k_values:
    kmeans_model = KMeans(n_clusters=k, random_state=42)
    kmeans_model.fit(image_array)
    inertia.append(kmeans_model.inertia_)

# Elbow Plot
plt.figure(figsize=(8, 6))
plt.plot(k_values, inertia, 'bo-', linewidth=2)
plt.title('Elbow Method for Optimal K')
plt.xlabel('Number of Clusters (k)')
plt.ylabel('Inertia')
plt.grid(True)
plt.show()
