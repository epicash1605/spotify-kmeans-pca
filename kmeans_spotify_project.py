# K-Means clustering on Spotify audio features (PCA + metrics + plots)

import os, shutil, warnings
warnings.filterwarnings("ignore", category=FutureWarning)

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.preprocessing import MinMaxScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import RFE
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score, davies_bouldin_score
from sklearn.decomposition import PCA
from kneed import KneeLocator

# ---- CONFIG -----------------------------------------------------------------
CSV_PATH = r"C:\Users\Ashwi\OneDrive - GEMS EDUCATION\K CLUSTERING PROJECT\spotify_songs.csv"
SAMPLE_N = 2000
N_FEATURES_TO_SELECT = 6
RANDOM_STATE = 42
FORCE_K = 4          # <<< set to 4 to match notebook; set to None for auto-pick

# Detect Colab (optional upload)
IN_COLAB = False
try:
    from google.colab import files as _colab_files  # type: ignore
    IN_COLAB = True
except Exception:
    pass

sns.set_theme(style="whitegrid")


def ensure_csv_available(csv_path: str) -> str:
    if os.path.exists(csv_path):
        return csv_path
    if IN_COLAB:
        print("Pick your CSV (e.g., spotify_songs.csv) and wait for upload to finish…")
        uploaded = _colab_files.upload()
        if not uploaded:
            raise RuntimeError("No file uploaded.")
        name = next(iter(uploaded))
        if name != csv_path:
            shutil.copy(name, csv_path)
        return csv_path
    raise FileNotFoundError(f"Couldn't find {csv_path}. Put it in the same folder or update CSV_PATH.")


def plot_grid_hists(df: pd.DataFrame, cols, title_prefix: str):
    cols = list(cols)
    n = len(cols)
    rows = max(int(np.ceil(n / 3)), 1)
    fig, axes = plt.subplots(nrows=rows, ncols=3, figsize=(16, 9), sharex=False, sharey=False)
    axes = np.array(axes).reshape(-1)
    for i, c in enumerate(cols):
        sns.histplot(df[c], bins=30, kde=True, ax=axes[i])
        axes[i].set_title(f"Distribution of {c} ({title_prefix})", fontsize=12, pad=8)
        axes[i].set_xlabel(c, fontsize=10)
        axes[i].set_ylabel("Count", fontsize=10)
    for j in range(i + 1, len(axes)):
        axes[j].axis("off")
    plt.tight_layout(h_pad=2.0, w_pad=2.0)
    plt.show()


def main():
    # ------------------ LOAD ---------------------------------------------------
    csv = ensure_csv_available(CSV_PATH)
    spotify_songs = pd.read_csv(csv)
    print(spotify_songs.shape)
    try:
        display(spotify_songs.head(3))  # type: ignore
    except Exception:
        print(spotify_songs.head(3))

    # Optional sampling
    if len(spotify_songs) > SAMPLE_N:
        spotify_songs = spotify_songs.sample(n=SAMPLE_N, random_state=RANDOM_STATE).reset_index(drop=True)

    # ------------------ FEATURES ----------------------------------------------
    features = [
        "danceability","energy","loudness","speechiness",
        "acousticness","instrumentalness","liveness","valence","tempo"
    ]
    features = [c for c in features if c in spotify_songs.columns]
    if not features:
        raise RuntimeError("None of the expected audio features were found in the CSV.")

    # ------------------ RAW HISTOGRAMS ----------------------------------------
    plot_grid_hists(spotify_songs, features, "raw")

    # ------------------ SCALING ------------------------------------------------
    X_raw = spotify_songs[features].apply(pd.to_numeric, errors="coerce")
    X_raw = X_raw.fillna(X_raw.median(numeric_only=True))
    scaler = MinMaxScaler()
    spotify_songs_scaled = pd.DataFrame(
        scaler.fit_transform(X_raw), columns=features, index=spotify_songs.index
    )

    try:
        display(spotify_songs_scaled.head())  # type: ignore
    except Exception:
        print(spotify_songs_scaled.head())

    plot_grid_hists(spotify_songs_scaled, features, "scaled")

    # ------------------ RFE ----------------------------------------------------
    dummy_y = np.ones(spotify_songs_scaled.shape[0], dtype=int)
    estimator = RandomForestClassifier(n_estimators=100, random_state=RANDOM_STATE, n_jobs=-1)
    selector = RFE(estimator, n_features_to_select=N_FEATURES_TO_SELECT)
    X_selected_np = selector.fit_transform(spotify_songs_scaled, dummy_y)
    selected_features = np.array(features)[selector.support_]
    print("Selected features (RFE w/ dummy y):", list(selected_features))

    X_selected = pd.DataFrame(X_selected_np, columns=selected_features, index=spotify_songs.index)
    try:
        display(X_selected.head())  # type: ignore
    except Exception:
        print(X_selected.head())

    # ------------------ ELBOW --------------------------------------------------
    interia = []
    k_range = range(1, 11)
    for k in k_range:
        kmeans = KMeans(n_clusters=k, random_state=RANDOM_STATE, n_init=10)
        kmeans.fit(X_selected)
        interia.append(kmeans.inertia_)
    print("inertia list:", interia)

    plt.figure(figsize=(8, 5))
    plt.plot(list(k_range), interia, marker='o', linestyle='-', color='b')
    plt.xlabel('Number of Clusters (k)')
    plt.ylabel('Inertia')
    plt.title('Elbow Method for Optimal k')
    plt.grid()
    plt.show()

    kl = KneeLocator(list(k_range), interia, curve='convex', direction='decreasing')
    optimal_k = int(kl.elbow) if kl.elbow is not None else None
    print("optimal_k (from elbow):", optimal_k)

    # ------------------ FINAL K-MEANS & METRICS --------------------------------
    if FORCE_K is not None:
        optimal_k = int(FORCE_K)
        print(f"FORCE_K is set -> using k={optimal_k}")
    else:
        if optimal_k is None or optimal_k < 2:
            from sklearn.metrics import silhouette_score as _sil
            sil = []
            for k in range(2, 11):
                km = KMeans(n_clusters=k, random_state=RANDOM_STATE, n_init=10).fit(X_selected)
                sil.append(_sil(X_selected, km.labels_))
            optimal_k = int(np.arange(2, 11)[np.argmax(sil)])
            print("Elbow not found; using best silhouette k:", optimal_k)

    final_kmeans = KMeans(n_clusters=optimal_k, random_state=RANDOM_STATE, n_init=10)
    final_labels = final_kmeans.fit_predict(X_selected)
    print("cluster labels (first 20):", final_labels[:20])
    # quick counts per cluster
    print("cluster sizes:", pd.Series(final_labels).value_counts().sort_index().to_dict())

    silhoutte_avg = silhouette_score(X_selected, final_labels)
    davies_avg = davies_bouldin_score(X_selected, final_labels)
    print("Silhouette:", silhoutte_avg)
    print("Davies–Bouldin:", davies_avg)

    # ------------------ PCA 2D -------------------------------------------------
    pca_2d = PCA(n_components=2, random_state=RANDOM_STATE)
    X_pca_2d = pca_2d.fit_transform(X_selected)
    loadings = pd.DataFrame(pca_2d.components_.T, columns=['PC1', 'PC2'], index=X_selected.columns)
    plt.figure(figsize=(8, 5))
    sns.heatmap(loadings, annot=True, cmap='coolwarm', center=0)
    plt.xlabel('Principal Components')
    plt.ylabel('Features')
    plt.title("PCA Loadings")
    plt.show()

    df_pca_2d = pd.DataFrame(X_pca_2d, columns=['PC1', 'PC2'])
    df_pca_2d['Cluster'] = final_labels
    plt.figure(figsize=(10, 6))
    sns.scatterplot(data=df_pca_2d, x='PC1', y='PC2', hue='Cluster', palette='Set1', alpha=0.8)
    plt.title('PCA 2D Visualization')
    plt.xlabel('Principal Component 1')
    plt.ylabel('Principal Component 2')
    plt.show()

    # ------------------ PCA 3D (Plotly) ---------------------------------------
    try:
        import plotly.express as px
        pca_3d = PCA(n_components=3, random_state=RANDOM_STATE)
        X_pca_3d = pca_3d.fit_transform(X_selected)

        loadings_3d = pd.DataFrame(pca_3d.components_.T, columns=['PC1', 'PC2', 'PC3'], index=X_selected.columns)
        plt.figure(figsize=(8, 5))
        sns.heatmap(loadings_3d, annot=True, cmap='coolwarm', center=0)
        plt.xlabel('Principal Components')
        plt.ylabel('Features')
        plt.title("PCA Loadings")
        plt.show()

        df_pca_3d = pd.DataFrame(X_pca_3d, columns=['PC1', 'PC2', 'PC3'])
        df_pca_3d['Cluster'] = final_labels
        fig = px.scatter_3d(
            df_pca_3d, x='PC1', y='PC2', z='PC3',
            color=df_pca_3d['Cluster'].astype(str),
            title='PCA 3D Visualization'
        )
        fig.write_html("pca_3d.html", include_plotlyjs="cdn", auto_open=True)
        print("Wrote pca_3d.html and opened it in your browser.")
    except Exception as e:
        print("Skipping Plotly 3D (install with `pip install plotly` if you want it). Error:", e)

# Save results (CSV + cluster profiles + selected features)
    out_csv = "spotify_clustered.csv"
    out_profiles = "cluster_profiles.csv"
    out_selected = "selected_features.txt"

    out_df = spotify_songs.copy()
    out_df["cluster"] = final_labels
    out_df.to_csv(out_csv, index=False)

    # Avoid column overlap: concatenate instead of join
    profiles = (
        pd.concat([out_df[["cluster"]], spotify_songs_scaled[features]], axis=1)
          .groupby("cluster")
          .mean()
          .round(3)
    )
    profiles.to_csv(out_profiles)

    with open(out_selected, "w") as f:
        f.write("\n".join(map(str, selected_features)))

    print(f"\nSaved:\n  {out_csv}\n  {out_profiles}\n  {out_selected}\n")


if __name__ == "__main__":
    main()

