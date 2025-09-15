from genedyn.src import *
from genedyn import io
import multiprocessing as mp
import time
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score
from sklearn.decomposition import PCA
import umap
import matplotlib.pyplot as plt
import plotly.express as px


def generate_data():
    """Generate parameter sets and collect data"""
    start = time.time()
    total_stable_points = 512
    sp = int(total_stable_points / mp.cpu_count())
    args = [(i, sp) for i in range(mp.cpu_count())]
    with mp.Pool(processes=mp.cpu_count()) as pool:
        pool.starmap(loop, args)
    end = time.time()
    print('\n')
    print('-' * 40)
    print(f"Elapsed time: {end - start:.2f} seconds")
    print('-' * 40)


def clustering_fixed_points():
    """Clustering with K-means and cluster analysis"""
    df = io.load_data()
    print(f"\nGenerated {len(df)} valid parameter sets.")
    print(df.head())

    # Print distribution of dynamics
    dynamics_counts = df['dynamics'].value_counts(normalize=True)
    print("\nDistribution of dynamics:")
    print(dynamics_counts)

    # Check standard deviation of fixed points
    print("\nStandard deviation of fixed points:")
    for col in ['CEBPA', 'SPI1', 'MYB', 'RUNX1', 'HOXA9', 'MEIS1']:
        print(f" {col}: {df[col].std():.4f}")

    # Select fixed points for clustering
    fix_points = df[['CEBPA', 'SPI1', 'MYB', 'RUNX1', 'HOXA9', 'MEIS1']]

    # Normalize data
    scaler = StandardScaler()
    fix_points_scaled = scaler.fit_transform(fix_points)

    # Select number of clusters using Silhouette Score
    sil_scores = []
    K = range(2, 10)
    for k in K:
        kmeans = KMeans(n_clusters=k, random_state=42)
        kmeans.fit(fix_points_scaled)
        if k > 1:
            sil_score = silhouette_score(fix_points_scaled, kmeans.labels_)
            sil_scores.append(sil_score)
        else:
            sil_scores.append(0)

    # Find optimal k based on Silhouette Score
    best_k = K[np.argmax(sil_scores[1:]) + 1]
    print(f"\nOptimal number of clusters (based on Silhouette Score): {best_k}")

    # Cluster with the optimal k
    kmeans = KMeans(n_clusters=best_k, random_state=42)
    clusters = kmeans.fit_predict(fix_points_scaled)
    df['cluster'] = clusters

    # Analyze clusters
    print("\nCluster analysis:")
    for cluster in range(best_k):
        cluster_data = df[df['cluster'] == cluster]
        print(f"\nCluster {cluster}:")
        print(f" Number of samples: {len(cluster_data)}")
        print(" Mean fixed points:")
        for col in ['CEBPA', 'SPI1', 'MYB', 'RUNX1', 'HOXA9', 'MEIS1']:
            print(f" {col}: {cluster_data[col].mean():.4f} ± {cluster_data[col].std():.4f}")
        print(" Dynamics distribution:")
        print(cluster_data['dynamics'].value_counts(normalize=True))
        print(" Mean interaction parameters:")
        for col in ['cc', 'rc', 'hc', 'cs', 'ss', 'hs', 'mymy', 'hmy', 'memy',
                    'rr', 'hr', 'rh', 'hh', 'meh', 'hme', 'meme', 'cr', 'sr', 'myh']:
            print(f" {col}: {cluster_data[col].mean():.4f} ± {cluster_data[col].std():.4f}")

    # Display 2D UMAP
    umap_reducer_2d = umap.UMAP(n_neighbors=30, min_dist=0.5, n_components=2, random_state=42)
    umap_embedding_2d = umap_reducer_2d.fit_transform(fix_points_scaled)

    plt.figure(figsize=(8, 6))
    scatter = plt.scatter(umap_embedding_2d[:, 0], umap_embedding_2d[:, 1], c=clusters, cmap='Spectral', s=50)
    plt.colorbar(scatter, label='Cluster')
    plt.title('UMAP 2D Clustering of Fixed Points')
    plt.xlabel('UMAP 1')
    plt.ylabel('UMAP 2')
    plt.show()

    # Display 3D UMAP
    umap_reducer_3d = umap.UMAP(n_neighbors=30, min_dist=0.5, n_components=3, random_state=42)
    umap_embedding_3d = umap_reducer_3d.fit_transform(fix_points_scaled)

    fig = px.scatter_3d(
        x=umap_embedding_3d[:, 0],
        y=umap_embedding_3d[:, 1],
        z=umap_embedding_3d[:, 2],
        color=clusters,
        labels={'x': 'UMAP 1', 'y': 'UMAP 2', 'z': 'UMAP 3'},
        title='UMAP 3D Clustering of Fixed Points',
        color_continuous_scale='Spectral'
    )
    fig.update_traces(marker=dict(size=5))
    fig.show()

    # Display 2D PCA
    pca_2d = PCA(n_components=2, random_state=42)
    pca_embedding_2d = pca_2d.fit_transform(fix_points_scaled)

    plt.figure(figsize=(8, 6))
    scatter = plt.scatter(pca_embedding_2d[:, 0], pca_embedding_2d[:, 1], c=clusters, cmap='Spectral', s=50)
    plt.colorbar(scatter, label='Cluster')
    plt.title('PCA 2D Clustering of Fixed Points')
    plt.xlabel('PCA 1')
    plt.ylabel('PCA 2')
    plt.show()

    # Display 3D PCA
    pca_3d = PCA(n_components=3, random_state=42)
    pca_embedding_3d = pca_3d.fit_transform(fix_points_scaled)

    fig = px.scatter_3d(
        x=pca_embedding_3d[:, 0],
        y=pca_embedding_3d[:, 1],
        z=pca_embedding_3d[:, 2],
        color=clusters,
        labels={'x': 'PCA 1', 'y': 'PCA 2', 'z': 'PCA 3'},
        title='PCA 3D Clustering of Fixed Points',
        color_continuous_scale='Spectral'
    )
    fig.update_traces(marker=dict(size=5))
    fig.show()


def learning_dynamics():
    df = io.load_data()
    print(f"\nGenerated {len(df)} valid parameter sets.")
    print(df.head())

    # Print distribution of dynamics
    dynamics_counts = df['dynamics'].value_counts(normalize=True)
    print("\nDistribution of dynamics:")
    print(dynamics_counts)

    # Check standard deviation of fixed points
    print("\nStandard deviation of fixed points:")
    for col in ['CEBPA', 'SPI1', 'MYB', 'RUNX1', 'HOXA9', 'MEIS1']:
        print(f" {col}: {df[col].std():.4f}")

    # Prepare data for training
    X = df.drop(['dynamics'], axis=1)
    y = df['dynamics']

    # Split data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Scale features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # Train Random Forest
    model = RandomForestClassifier(random_state=42)
    model.fit(X_train_scaled, y_train)

    # Evaluate model
    y_pred = model.predict(X_test_scaled)
    print("\nAccuracy:", model.score(X_test_scaled, y_test))
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred))

    # Parameter analysis for each dynamic
    print("\nParameter analysis for each dynamic:")
    for dynamic in df['dynamics'].unique():
        dynamic_data = df[df['dynamics'] == dynamic]
        print(f"\nDynamic {dynamic}:")
        print(f" Number of samples: {len(dynamic_data)}")
        print(" Mean fixed points:")
        for col in ['CEBPA', 'SPI1', 'MYB', 'RUNX1', 'HOXA9', 'MEIS1']:
            print(f" {col}: {dynamic_data[col].mean():.4f} ± {dynamic_data[col].std():.4f}")
        print(" Mean interaction parameters:")
        for col in ['cc', 'rc', 'hc', 'cs', 'ss', 'hs', 'mymy', 'hmy', 'memy',
                    'rr', 'hr', 'rh', 'hh', 'meh', 'hme', 'meme', 'cr', 'sr', 'myh']:
            print(f" {col}: {dynamic_data[col].mean():.4f} ± {dynamic_data[col].std():.4f}")

    # Save model and scaler
    with open('rf_model.pkl', 'wb') as f:
        pickle.dump(model, f)
    with open('scaler.pkl', 'wb') as f:
        pickle.dump(scaler, f)
    print("Model and scaler saved: rf_model.pkl, scaler.pkl")
