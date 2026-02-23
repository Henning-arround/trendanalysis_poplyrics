import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
import umap
import os

# Configure plot style for LaTeX
mpl.rcParams.update({
    'font.family': 'serif',
    'font.size': 12,
    'axes.titlesize': 14,
    'axes.labelsize': 12,
    'xtick.labelsize': 10,
    'ytick.labelsize': 10,
    'legend.fontsize': 10,
    'figure.titlesize': 16,
    'text.usetex': True
})

# Setup paths
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
EMBEDDINGS_PATH = os.path.join(BASE_DIR, "../../../bertopic/dtm/embedding/embeddings.npy")
OUTPUT_DIR = os.path.join(BASE_DIR, "../../../visualisations/topic_model")

os.makedirs(OUTPUT_DIR, exist_ok=True)

def load_embeddings(path):
    print(f"Loading embeddings from {path}...")
    return np.load(path)

def plot_triple_3d(embeddings):
    print("Running UMAP 3D...")
    # Use same parameters as original script for consistency, added min_dist
    reducer = umap.UMAP(n_neighbors=70, n_components=3, min_dist=0.5, random_state=42, n_jobs=1, low_memory=False)
    embedding_3d = reducer.fit_transform(embeddings)
    
    print("Creating triple view plot...")
    # Create figure with 3 subplots - 2 in top row, 1 in bottom row
    fig = plt.figure(figsize=(15, 12))
    gs = fig.add_gridspec(2, 2)
    
    # First view - Standard isometric-like view
    ax1 = fig.add_subplot(gs[0, 0], projection='3d')
    sc1 = ax1.scatter(embedding_3d[:, 0], embedding_3d[:, 1], embedding_3d[:, 2], s=1, alpha=0.5, c=embedding_3d[:, 2], cmap='viridis')
    ax1.set_title('Ansicht 1', fontsize=14)
    ax1.set_xlabel('UMAP 1')
    ax1.set_ylabel('UMAP 2')
    ax1.set_zlabel('UMAP 3')
    ax1.view_init(elev=30, azim=45)
    
    # Second view - Rotated 90 degrees
    ax2 = fig.add_subplot(gs[0, 1], projection='3d')
    sc2 = ax2.scatter(embedding_3d[:, 0], embedding_3d[:, 1], embedding_3d[:, 2], s=1, alpha=0.5, c=embedding_3d[:, 2], cmap='viridis')
    ax2.set_title('Ansicht 2 (rotiert 90°)', fontsize=14)
    ax2.set_xlabel('UMAP 1')
    ax2.set_ylabel('UMAP 2')
    ax2.set_zlabel('UMAP 3')
    ax2.view_init(elev=30, azim=135)

    # Third view - UMAP 1 vs UMAP 3 (Side view)
    ax3 = fig.add_subplot(gs[1, :], projection='3d')
    sc3 = ax3.scatter(embedding_3d[:, 0], embedding_3d[:, 1], embedding_3d[:, 2], s=1, alpha=0.5, c=embedding_3d[:, 2], cmap='viridis')
    # Title moved closer with negative padding
    ax3.set_title('Ansicht 3 (UMAP 1 vs UMAP 3)', fontsize=14, y=0.95)
    ax3.set_xlabel('UMAP 1')
    
    # UMAP 2: Remove tick labels, keep axis label with padding
    ax3.set_ylabel('UMAP 2', labelpad=15)
    ax3.set_yticklabels([])
    
    ax3.set_zlabel('UMAP 3')
    # elev=0 looks from the side (at Z axis), azim=-90 positions X axis horizontally
    ax3.view_init(elev=0, azim=-90)
    
    # Add a colorbar
    # plt.colorbar(sc2, ax=[ax1, ax2], label='UMAP 3 Dimension', shrink=0.7)

    plt.tight_layout()
    
    output_path = os.path.join(OUTPUT_DIR, "embeddings_umap_3d_triple_view.png")
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"Saved {output_path}")

def main():
    if not os.path.exists(EMBEDDINGS_PATH):
        print(f"Error: Embeddings file not found at {EMBEDDINGS_PATH}")
        return

    embeddings = load_embeddings(EMBEDDINGS_PATH)
    print(f"Embeddings shape: {embeddings.shape}")
    
    plot_triple_3d(embeddings)

if __name__ == "__main__":
    main()
