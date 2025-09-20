from genedyn.src import *
from genedyn import io
import multiprocessing as mp
import time
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report
from umap import UMAP
import matplotlib.pyplot as plt
import plotly.express as px
from sklearn.metrics import confusion_matrix, balanced_accuracy_score
from sklearn.utils import resample
import numpy as np
import pandas as pd


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

    # Print distribution of dynamics
    dynamics_counts = df['dynamics'].value_counts(normalize=True)
    print("\nDistribution of dynamics:")
    print(dynamics_counts)

    # Check standard deviation of fixed points
    print("\nStandard deviation of fixed points:")
    for col in ['CEBPA', 'SPI1', 'MYB', 'RUNX1', 'HOXA9', 'MEIS1']:
        print(f" {col}: {df[col].std():.4f}")
    # Separate each class
    stable_df = df[df['dynamics'] == 'Stable']
    nzfp_df = df[df['dynamics'] == 'Non-zero Fixed Points']
    saddle_df = df[df['dynamics'] == 'Saddle']

    # Choose how many Saddle samples to keep (preserving original order)
    n_saddle = int(stable_df.shape[0] * 1.2)  # adjust this number as needed
    saddle_df_sampled = saddle_df.head(n_saddle)

    # Combine all parts
    custom_df = pd.concat([stable_df, nzfp_df, saddle_df_sampled])

    # Select fixed points for clustering
    fix_points = custom_df[['CEBPA', 'SPI1', 'MYB', 'RUNX1', 'HOXA9', 'MEIS1']]

    # Normalize data
    scaler = StandardScaler()
    fix_points_scaled = scaler.fit_transform(fix_points)

    # Select number of clusters using Silhouette Score
    sil_scores = []
    K = range(2, 10)
    for k in K:
        kmeans = KMeans(n_clusters=k)
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
    kmeans = KMeans(n_clusters=best_k)
    clusters = kmeans.fit_predict(fix_points_scaled)
    custom_df['cluster'] = clusters

    # Analyze clusters
    print("\nCluster analysis:")
    for cluster in range(best_k):
        cluster_data = custom_df[custom_df['cluster'] == cluster]
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
    umap_reducer_2d = UMAP(n_neighbors=30, min_dist=0.5, n_components=2)
    umap_embedding_2d = umap_reducer_2d.fit_transform(fix_points_scaled)

    plt.figure(figsize=(8, 6))
    scatter = plt.scatter(umap_embedding_2d[:, 0], umap_embedding_2d[:, 1], c=clusters, cmap='Spectral', s=8, alpha=0.8)
    plt.colorbar(scatter, label='Cluster')
    plt.title('UMAP 2D Clustering of Fixed Points')
    plt.xlabel('UMAP 1')
    plt.ylabel('UMAP 2')
    plt.savefig("fig/UMAP-2D.png", dpi=300, bbox_inches='tight')
    plt.show()

    # Display 3D UMAP
    umap_reducer_3d = UMAP(n_neighbors=30, min_dist=0.5, n_components=3)
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
    fig.update_traces(
        marker=dict(
            size=5,
            opacity=0.8,
            line=dict(width=0)
        )
    )
    fig.update_layout(
        width=800,
        height=800,
        margin=dict(l=60, r=60, t=70, b=70),
        scene_aspectmode='cube',
        scene=dict(aspectratio=dict(x=1, y=1, z=1))
    )
    fig.write_image("fig/UMAP-3D.png", width=1000, height=1000, scale=1)
    fig.show()

    # Display 2D PCA
    pca_2d = PCA(n_components=2)
    pca_embedding_2d = pca_2d.fit_transform(fix_points_scaled)

    plt.figure(figsize=(8, 6))
    scatter = plt.scatter(pca_embedding_2d[:, 0], pca_embedding_2d[:, 1], c=clusters, cmap='Spectral', s=8, alpha=0.8)
    plt.colorbar(scatter, label='Cluster')
    plt.title('PCA 2D Clustering of Fixed Points')
    plt.xlabel('PCA 1')
    plt.ylabel('PCA 2')
    plt.savefig("fig/PCA-2D.png", dpi=300, bbox_inches='tight')
    plt.show()

    # Display 3D PCA
    pca_3d = PCA(n_components=3)
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
    fig.update_traces(
        marker=dict(
            size=5,
            opacity=0.8,
            line=dict(width=0)
        )
    )
    fig.update_layout(
        width=800,
        height=800,
        margin=dict(l=60, r=60, t=70, b=70),
        scene_aspectmode='cube',
        scene=dict(aspectratio=dict(x=1, y=1, z=1))
    )
    fig.write_image("fig/PCA-3D.png", width=1000, height=1000, scale=1)
    fig.show()


def learning_dynamics_custom_sampling():
    df = io.load_data()
    print(f"\nGenerated {len(df)} valid parameter sets.")

    # Class distribution
    dynamics_counts = df['dynamics'].value_counts(normalize=True)
    print("\nDistribution of dynamics:")
    print(dynamics_counts)

    # Std dev of fixed points
    print("\nStandard deviation of fixed points:")
    for col in ['CEBPA', 'SPI1', 'MYB', 'RUNX1', 'HOXA9', 'MEIS1']:
        print(f" {col}: {df[col].std():.4f}")

    # --- Custom undersampling ---
    # Separate each class
    stable_df = df[df['dynamics'] == 'Stable']
    nzfp_df = df[df['dynamics'] == 'Non-zero Fixed Points']
    saddle_df = df[df['dynamics'] == 'Saddle']

    # Choose how many Saddle samples to keep (preserving original order)
    n_saddle = int(stable_df.shape[0] * 1.2)  # adjust this number as needed
    saddle_df_sampled = saddle_df.head(n_saddle)

    # Combine all parts
    custom_df = pd.concat([stable_df, nzfp_df, saddle_df_sampled])
    preferred_df = pd.concat([saddle_df])
    print('\nEvaluate the model on a balanced dataset:')
    print(f"Final dataset size: {len(custom_df)}")
    print(custom_df['dynamics'].value_counts())

    # Prepare data
    X = custom_df.drop(['dynamics'], axis=1)
    y = custom_df['dynamics']

    # Train/test split (stratified to preserve class ratios)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, stratify=y
    )

    # Scale features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # Train model = RandomForestClassifier
    model = RandomForestClassifier(
        n_estimators=100,  # keep or lower
        max_depth=5,  # limit tree depth
        min_samples_split=5,  # require more samples to split
        min_samples_leaf=2  # require more samples per leaf
    )
    model.fit(X_train_scaled, y_train)

    # Evaluate
    y_pred = model.predict(X_test_scaled)
    print("\nAccuracy:", model.score(X_test_scaled, y_test))
    print("Balanced Accuracy:", balanced_accuracy_score(y_test, y_pred))
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred, digits=4))
    print("\nConfusion Matrix:")
    print(confusion_matrix(y_test, y_pred))

    train_acc = model.score(X_train_scaled, y_train)
    test_acc = model.score(X_test_scaled, y_test)

    # Parameter analysis
    print("\nParameter analysis for each dynamic:")
    for dynamic in custom_df['dynamics'].unique():
        dynamic_data = custom_df[custom_df['dynamics'] == dynamic]
        print(f"\nDynamic {dynamic}:")
        print(f" Number of samples: {len(dynamic_data)}")
        print(" Mean fixed points:")
        for col in ['CEBPA', 'SPI1', 'MYB', 'RUNX1', 'HOXA9', 'MEIS1']:
            print(f" {col}: {dynamic_data[col].mean():.4f} ± {dynamic_data[col].std():.4f}")
        print(" Mean interaction parameters:")
        for col in ['cc', 'rc', 'hc', 'cs', 'ss', 'hs', 'mymy', 'hmy', 'memy',
                    'rr', 'hr', 'rh', 'hh', 'meh', 'hme', 'meme', 'cr', 'sr', 'myh']:
            print(f" {col}: {dynamic_data[col].mean():.4f} ± {dynamic_data[col].std():.4f}")

    # --- Evaluate model on all Saddle samples ---

    print('\nEvaluate the model on all Saddle samples:')
    X_saddle = preferred_df.drop(['dynamics'], axis=1)
    X_saddle_scaled = scaler.transform(X_saddle)
    y_saddle = preferred_df['dynamics']
    y_saddle_pred = model.predict(X_saddle_scaled)
    true_saddle = y_saddle.values
    pred_saddle = y_saddle_pred

    # Boolean mask for correct predictions
    correct_mask = (true_saddle == pred_saddle)

    # Accuracy
    saddle_accuracy = np.mean(correct_mask)
    print(f"\n✅ Saddle Accuracy: {saddle_accuracy:.4f} ({np.sum(correct_mask)} / {len(true_saddle)} correct)")

    # Misclassified samples
    misclassified = pd.Series(pred_saddle[~correct_mask])
    misclass_counts = misclassified.value_counts()

    if not misclass_counts.empty:
        print("\n❌ Misclassifications (Saddle predicted as):")
        for label, count in misclass_counts.items():
            print(f"  {label}: {count} samples")
    else:
        print("\n🎯 All Saddle samples were correctly classified.")


def learning_dynamics_undersampling():
    df = io.load_data()
    print(f"\nGenerated {len(df)} valid parameter sets.")
    print(df.head())

    # Class distribution
    dynamics_counts = df['dynamics'].value_counts(normalize=True)
    print("\nDistribution of dynamics:")
    print(dynamics_counts)

    # Std dev of fixed points
    print("\nStandard deviation of fixed points:")
    for col in ['CEBPA', 'SPI1', 'MYB', 'RUNX1', 'HOXA9', 'MEIS1']:
        print(f" {col}: {df[col].std():.4f}")

    # --- Random undersampling ---
    min_class_size = df['dynamics'].value_counts().min()
    balanced_df = pd.concat([
        resample(group, replace=False, n_samples=min_class_size)
        for _, group in df.groupby('dynamics')
    ])
    print(f"\nAfter undersampling: {len(balanced_df)} samples total")
    print(balanced_df['dynamics'].value_counts())

    # Prepare data
    X = balanced_df.drop(['dynamics'], axis=1)
    y = balanced_df['dynamics']

    # Train/test split (stratified to preserve class ratios)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, stratify=y
    )

    # Scale features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # Train model = RandomForestClassifier
    model = RandomForestClassifier(
        n_estimators=100,  # keep or lower
        max_depth=5,  # limit tree depth
        min_samples_split=5,  # require more samples to split
        min_samples_leaf=2  # require more samples per leaf
    )
    model.fit(X_train_scaled, y_train)

    # Evaluate
    y_pred = model.predict(X_test_scaled)
    print("\nAccuracy:", model.score(X_test_scaled, y_test))
    print("Balanced Accuracy:", balanced_accuracy_score(y_test, y_pred))
    print("\nClassification Report (macro avg is key):")
    print(classification_report(y_test, y_pred, digits=4))
    print("\nConfusion Matrix:")
    print(confusion_matrix(y_test, y_pred))

    train_acc = model.score(X_train_scaled, y_train)
    test_acc = model.score(X_test_scaled, y_test)
    print(train_acc, test_acc)

    # Parameter analysis
    print("\nParameter analysis for each dynamic:")
    for dynamic in balanced_df['dynamics'].unique():
        dynamic_data = balanced_df[balanced_df['dynamics'] == dynamic]
        print(f"\nDynamic {dynamic}:")
        print(f" Number of samples: {len(dynamic_data)}")
        print(" Mean fixed points:")
        for col in ['CEBPA', 'SPI1', 'MYB', 'RUNX1', 'HOXA9', 'MEIS1']:
            print(f" {col}: {dynamic_data[col].mean():.4f} ± {dynamic_data[col].std():.4f}")
        print(" Mean interaction parameters:")
        for col in ['cc', 'rc', 'hc', 'cs', 'ss', 'hs', 'mymy', 'hmy', 'memy',
                    'rr', 'hr', 'rh', 'hh', 'meh', 'hme', 'meme', 'cr', 'sr', 'myh']:
            print(f" {col}: {dynamic_data[col].mean():.4f} ± {dynamic_data[col].std():.4f}")

    X_full = df.drop(['dynamics'], axis=1)
    X_full_scaled = scaler.transform(X_full)
    y_full = df['dynamics']

    y_full_pred = model.predict(X_full)
    acc_all = model.score(X_full_scaled, y_full_pred)
    print('Accuracy on full samples:', acc_all)
    print("Balanced Accuracy on full samples:", balanced_accuracy_score(y_full, y_full_pred))
    print("\nClassification Report on full samples:")
    print(classification_report(y_full, y_full_pred, digits=4))
    print("\nConfusion Matrix of full samples:")
    print(confusion_matrix(y_full, y_full_pred))
