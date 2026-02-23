import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
import matplotlib.animation as animation
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
    'text.usetex': False
})

# Setup paths
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
EMBEDDINGS_PATH = os.path.join(BASE_DIR, "../../../bertopic/dtm/embedding/embeddings.npy")
OUTPUT_DIR = os.path.join(BASE_DIR, "../../../visualisations/topic_model")

os.makedirs(OUTPUT_DIR, exist_ok=True)

def load_embeddings(path):
    print(f"Loading embeddings from {path}...")
    return np.load(path)

def plot_2d(embeddings):
    print("Running UMAP 2D...")
    reducer = umap.UMAP(n_neighbors=70, n_components=2, random_state=42, n_jobs=1, low_memory=False)
    embedding_2d = reducer.fit_transform(embeddings)
    
    plt.figure(figsize=(12, 10))
    plt.scatter(embedding_2d[:, 0], embedding_2d[:, 1], s=1, alpha=0.5)
    plt.title('UMAP Projektion der Embeddings (2D)', fontsize=16)
    plt.xlabel('UMAP 1')
    plt.ylabel('UMAP 2')
    plt.tight_layout()
    
    output_path = os.path.join(OUTPUT_DIR, "embeddings_umap_2d.png")
    plt.savefig(output_path, dpi=300)
    print(f"Saved {output_path}")

def plot_3d(embeddings):
    print("Running UMAP 3D...")
    reducer = umap.UMAP(n_neighbors=70, n_components=3, random_state=42, n_jobs=1, low_memory=False)
    embedding_3d = reducer.fit_transform(embeddings)
    
    fig = plt.figure(figsize=(12, 10))
    ax = fig.add_subplot(111, projection='3d')
    ax.scatter(embedding_3d[:, 0], embedding_3d[:, 1], embedding_3d[:, 2], s=1, alpha=0.5)
    
    ax.set_title('UMAP Projektion der Embeddings (3D)', fontsize=16)
    ax.set_xlabel('UMAP 1')
    ax.set_ylabel('UMAP 2')
    ax.set_zlabel('UMAP 3')
    plt.tight_layout()
    
    output_path = os.path.join(OUTPUT_DIR, "embeddings_umap_3d.png")
    plt.savefig(output_path, dpi=300)
    print(f"Saved {output_path}")

    # Create GIF
    def rotate(angle):
        ax.view_init(elev=10, azim=angle)

    print("Creating 3D rotation GIF...")
    # Frames: 0 to 360 degrees
    ani = animation.FuncAnimation(fig, rotate, frames=np.arange(0, 360, 2), interval=50)
    gif_path = os.path.join(OUTPUT_DIR, "embeddings_umap_3d.gif")
    ani.save(gif_path, writer='pillow', fps=20)
    print(f"Saved {gif_path}")

def main():
    if not os.path.exists(EMBEDDINGS_PATH):
        print(f"Error: Embeddings file not found at {EMBEDDINGS_PATH}")
        return

    embeddings = load_embeddings(EMBEDDINGS_PATH)
    print(f"Embeddings shape: {embeddings.shape}")
    
    output_2d = os.path.join(OUTPUT_DIR, "embeddings_umap_2d.png")
    if not os.path.exists(output_2d):
        plot_2d(embeddings)
    else:
        print(f"Skipping 2D plot, {output_2d} already exists.")

    plot_3d(embeddings)

if __name__ == "__main__":
    main()
