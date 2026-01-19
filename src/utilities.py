import torch
import gc
import faiss
import random
import sys
import os

import streamlit as st
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import numpy as np

from torchvision import transforms
from tqdm import tqdm
from PIL import Image

# Add parent directory to path for imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
try:
    from joan_kusuma_model import JoanKusumaEmbeddingModel as FeaturizerModel
except ImportError:
    from featurizer_model import FeaturizerModel

class ExactIndex():
    def __init__(self, vectors, img_paths):
        self.dimension = vectors.shape[1]
        self.vectors = vectors.astype('float32')
        self.img_paths = img_paths

    def build(self):
        self.index = faiss.IndexFlatL2(self.dimension)
        self.index.add(self.vectors)

    def query(self, vectors, k=10):
        distances, indices = self.index.search(vectors, k)
        return [self.img_paths[i] for i in indices[0]]
    
    def save(self, filename):
        faiss.write_index(self.index, filename)
    
    @classmethod
    def load(cls, vectors, img_paths, filename):
        instance = cls(vectors, img_paths)
        instance.index = faiss.read_index(filename)
        return instance

def display_image(file_name):
    try:
        img = Image.open(file_name)
        plt.imshow(img)
        plt.axis('off')
        plt.show()
    except Exception as e:
        print(f"An error occured: {e}")


def visualize_nearest_neighbors(selected_img_path, nearest_neighbor_paths):
    # Create a figure with two columns
    fig, axs = plt.subplots(5, 2, figsize=(10, 8))

    plt.suptitle("Recommended Items based on your selection", fontsize=16, y=1.03)

    # Display the item selected in the first column (column 0)
    selected_img = mpimg.imread(selected_img_path)
    axs[0, 0].imshow(selected_img)
    axs[0, 0].set_title("Item selected")
    axs[0, 0].axis('off')

    # Limit the number of displayed neighbors to a maximum of 10
    num_neighbors = min(len(nearest_neighbor_paths), 10)

    # Loop through the recommended items (nearest neighbors) and display them in the second column (column 1)
    for i, ax in enumerate(axs[1:].flatten(), 1):
        if i <= num_neighbors:
            neighbor_path = nearest_neighbor_paths[i - 1]
            img = mpimg.imread(neighbor_path)
            ax.imshow(img)
            ax.set_title(f"Recommended Item {i}")
            ax.axis('off')

    # Hide the axis line in the second column of the first row
    for i in range(5):
        axs[i, 0].axis('off')
        axs[i, 1].axis('off')

    # Show the images
    return fig

# Global model cache to avoid reloading
_model_cache = None

@st.cache_resource
def load_featurizer_model():
    """Load and cache the featurizer model"""
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Try multiple paths for the model file
    model_paths = [
        'featurizer-model-1.pt',
        'featurizer-model-best.pt',
        './featurizer-model-1.pt',
        './models/featurizer-model-1.pt',
        os.path.join(os.getcwd(), 'featurizer-model-1.pt')
    ]
    
    model_loaded = False
    model = None
    
    for path in model_paths:
        if os.path.exists(path):
            try:
                # Check if file is actually a PyTorch model (not Git LFS pointer)
                with open(path, 'rb') as f:
                    first_bytes = f.read(20)
                    if b'version https://git-lfs' in first_bytes:
                        print(f"Skipping Git LFS pointer file: {path}")
                        continue
                
                # Fix for PyTorch 2.6+ - use weights_only=False for compatibility
                checkpoint = torch.load(path, map_location=device, weights_only=False)
                
                # Use the existing FeaturizerModel architecture
                model = FeaturizerModel().to(device)
                
                # Load state dict
                if 'model_state_dict' in checkpoint:
                    model.load_state_dict(checkpoint['model_state_dict'], strict=False)
                else:
                    model.load_state_dict(checkpoint, strict=False)
                
                model_loaded = True
                print(f"Model loaded from: {path}")
                break
                
            except Exception as e:
                print(f"Failed to load model from {path}: {e}")
                continue
    
    if not model_loaded:
        print("No valid model found. Creating a basic model for inference...")
        # Create a basic model if no trained model is found
        model = FeaturizerModel().to(device)
        
        # Initialize with random weights (not ideal, but allows app to run)
        print("WARNING: Using randomly initialized model. For best results, train the model first.")
    
    model.eval()
    return model, device

def extract_img(image, transformation):
    """Extract features from image using cached model"""
    try:
        model, device = load_featurizer_model()
        
        # Convert image to PIL and preprocess
        if isinstance(image, np.ndarray):
            pil_image = Image.fromarray(image)
        else:
            pil_image = image
            
        # Ensure image is in RGB mode and resize
        pil_image = pil_image.convert("RGB").resize((128, 128))
        tensor = transformation(pil_image).to(device)
        
        # Extract features using encoder only
        with torch.no_grad():
            latent_feature = model.encoder(tensor.unsqueeze(0)).cpu().detach().numpy()
        
        # Clean up
        del tensor
        torch.cuda.empty_cache() if torch.cuda.is_available() else None
        
        return np.array(latent_feature)
    
    except Exception as e:
        st.error(f"Error extracting features: {e}")
        return None

def similar_img_search(query_vector, index, k=6):
    """Search for similar images using the index"""
    try:
        if query_vector is None:
            return []
            
        query_vector = query_vector.reshape(1, -1)
        nearest_neighbors = index.query(query_vector, k=k)
        
        # Skip the first result (usually the query itself) and return the rest
        selected_neighbors_paths = nearest_neighbors[1:] if len(nearest_neighbors) > 1 else nearest_neighbors
        return selected_neighbors_paths
        
    except Exception as e:
        st.error(f"Error in similarity search: {e}")
        return []

def visualize_outfits(boards, max_display=8):
    """Visualize recommended outfits with better error handling"""
    if not boards:
        fig, ax = plt.subplots(1, 1, figsize=(6, 4))
        ax.text(0.5, 0.5, 'No recommendations found', 
                horizontalalignment='center', verticalalignment='center', 
                transform=ax.transAxes, fontsize=14)
        ax.axis('off')
        return fig
    
    # Determine grid size based on number of items
    num_items = min(len(boards), max_display)
    cols = 4 if num_items > 4 else num_items
    rows = (num_items + cols - 1) // cols
    
    fig, axs = plt.subplots(rows, cols, figsize=(cols * 2.5, rows * 3))
    
    # Ensure axs is always a list for consistency
    if rows == 1 and cols == 1:
        axs = [axs]
    elif rows == 1:
        axs = axs.flatten()
    else:
        axs = axs.flatten()

    plt.suptitle(f"Top {num_items} Recommended Fashion Items", fontsize=14, y=0.98)
    
    # Randomly select items to display for variety
    display_paths = random.sample(boards, num_items) if len(boards) > num_items else boards

    # Display images
    for i in range(len(axs)):
        ax = axs[i]
        if i < len(display_paths):
            try:
                img_path = display_paths[i]
                if os.path.exists(img_path):
                    img = mpimg.imread(img_path)
                    ax.imshow(img)
                    ax.set_title(f"Item {i+1}", fontsize=10)
                else:
                    ax.text(0.5, 0.5, 'Image not found', 
                           horizontalalignment='center', verticalalignment='center', 
                           transform=ax.transAxes, fontsize=10)
            except Exception as e:
                ax.text(0.5, 0.5, f'Error loading\nimage', 
                       horizontalalignment='center', verticalalignment='center', 
                       transform=ax.transAxes, fontsize=8)
        ax.axis('off')
    
    plt.tight_layout()
    return fig

def viz_thumbnail(im_path, tn_sz):
    a_img = mpimg.imread(im_path)
    # Get the dimensions of the original image
    img_height, img_width, _ = a_img.shape

    # Calculate the padding needed to make the image square
    max_dim = max(img_height, img_width)
    pad_vert = (max_dim - img_height) // 2
    pad_horiz = (max_dim - img_width) // 2

    # Create new image with padding
    padded_img = np.pad(a_img, ((pad_vert, pad_vert), (pad_horiz, pad_horiz), (0, 0)), mode='constant', constant_values=255)

    # Create fig and axis
    fig, ax = plt.subplots(figsize=tn_sz)

    ax.imshow(padded_img)

    # remove axes ticks and labels for a cleaner look
    ax.axis('off')

    return fig
