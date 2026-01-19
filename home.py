import streamlit as st
import numpy as np
import pickle
import matplotlib.pyplot as plt
import matplotlib.image as mpimg

from obj_detection import ObjDetection
from PIL import Image
from torchvision import transforms

from src.utilities import ExactIndex, extract_img, similar_img_search, display_image, visualize_nearest_neighbors, visualize_outfits


# --- Modern UI Configurations --- #
st.set_page_config(
    page_title="Élégant Fashion AI",
    page_icon="✨",
    layout="wide",
    initial_sidebar_state="expanded"
)

# --- Custom CSS for Modern, Elegant Design --- #
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Playfair+Display:wght@300;400;500;600;700&family=Inter:wght@200;300;400;500&display=swap');
    
    /* Main container styling */
    .stApp {
        background: linear-gradient(135deg, #fdf5f4 0%, #fff1ef 100%);
        font-family: 'Inter', sans-serif;
    }
    
    /* Header styling */
    .main-header {
        text-align: center;
        padding: 2rem 0;
        margin-bottom: 2rem;
    }
    
    .elegant-title {
        font-family: 'Playfair Display', serif;
        font-size: 3.5rem;
        font-weight: 600;
        color: #8b7355;
        text-align: center;
        margin-bottom: 0.5rem;
        letter-spacing: 2px;
        text-shadow: 1px 1px 2px rgba(0,0,0,0.1);
    }
    
    .elegant-subtitle {
        font-family: 'Inter', sans-serif;
        font-size: 1.2rem;
        font-weight: 500;
        color: #9d8671;
        text-align: center;
        margin-bottom: 2rem;
        letter-spacing: 1px;
    }
    
    /* Card styling */
    .elegant-card {
        background: rgba(255, 241, 239, 0.7);
        backdrop-filter: blur(10px);
        border-radius: 20px;
        border: 1px solid rgba(239, 226, 224, 0.3);
        padding: 2rem;
        margin: 1rem 0;
        box-shadow: 0 8px 32px rgba(194, 173, 161, 0.1);
    }
    
    .upload-card {
        background: linear-gradient(145deg, #fff1ef, #efe2e0);
        border-radius: 24px;
        padding: 3rem;
        text-align: center;
        border: 2px dashed #dfd4cc;
        transition: all 0.3s ease;
    }
    
    .upload-card:hover {
        border-color: #c2ada1;
        transform: translateY(-2px);
        box-shadow: 0 12px 40px rgba(194, 173, 161, 0.15);
    }
    
    /* Button styling */
    .stButton > button {
        background: linear-gradient(135deg, #c2ada1, #dfd4cc);
        color: #fdf5f4;
        border: none;
        border-radius: 50px;
        padding: 0.75rem 2rem;
        font-family: 'Inter', sans-serif;
        font-weight: 500;
        font-size: 1rem;
        letter-spacing: 0.5px;
        transition: all 0.3s ease;
        box-shadow: 0 4px 16px rgba(194, 173, 161, 0.3);
    }
    
    .stButton > button:hover {
        background: linear-gradient(135deg, #dfd4cc, #c2ada1);
        transform: translateY(-2px);
        box-shadow: 0 8px 24px rgba(194, 173, 161, 0.4);
    }
    
    /* File uploader styling */
    .stFileUploader {
        background: transparent;
    }
    
    .stFileUploader > div {
        background: rgba(253, 245, 244, 0.8);
        border-radius: 16px;
        border: 2px dashed #efe2e0;
        padding: 2rem;
    }
    
    /* Text styling */
    .elegant-text {
        font-family: 'Inter', sans-serif;
        font-weight: 500;
        color: #6b5b4a;
        line-height: 1.6;
        font-size: 1.1rem;
    }
    
    .feature-text {
        font-family: 'Inter', sans-serif;
        font-weight: 500;
        color: #8b7355;
        font-size: 1rem;
        text-align: center;
    }
    
    /* Sidebar styling */
    .css-1d391kg {
        background: linear-gradient(180deg, #fdf5f4, #fff1ef);
    }
    
    /* Success/Error messages */
    .stSuccess {
        background: rgba(194, 173, 161, 0.1);
        border-left: 4px solid #c2ada1;
        color: #c2ada1;
    }
    
    .stError {
        background: rgba(223, 212, 204, 0.1);
        border-left: 4px solid #dfd4cc;
        color: #dfd4cc;
    }
    
    /* Spinner styling */
    .stSpinner {
        color: #c2ada1;
    }
    
    /* Image container */
    .image-container {
        border-radius: 16px;
        overflow: hidden;
        box-shadow: 0 8px 32px rgba(194, 173, 161, 0.15);
        margin: 1rem 0;
    }
    
    /* Recommendations section */
    .recommendations-header {
        font-family: 'Playfair Display', serif;
        font-size: 2.2rem;
        font-weight: 600;
        color: #6b5b4a;
        text-align: center;
        margin: 2rem 0 1rem 0;
        letter-spacing: 1px;
        text-shadow: 1px 1px 2px rgba(0,0,0,0.1);
    }
</style>
""", unsafe_allow_html=True)

# --- Elegant Header --- #
st.markdown("""
<div class="main-header">
    <h1 class="elegant-title">Élégant Fashion AI</h1>
    <p class="elegant-subtitle">Discover your perfect style with intelligent fashion curation</p>
</div>
""", unsafe_allow_html=True)

# --- Welcome Message --- #
st.markdown("""
<div class="elegant-card">
    <p class="elegant-text">Welcome to the future of fashion discovery. Our AI-powered system analyzes your style preferences and curates personalized recommendations with sophisticated computer vision technology.</p>
    <p class="feature-text">✨ Upload an image and let our Joan Kusuma +Layer CNN model find your perfect matches</p>
</div>
""", unsafe_allow_html=True)

# --- Load Model and Data --- #
@st.cache_resource
def load_models_and_data():
    """Load YOLO model and search index with caching"""
    try:
        # Load YOLO model
        yolo = ObjDetection(onnx_model='./models/best.onnx',
                            data_yaml='./models/data.yaml')
        
        # Load image paths and embeddings
        with open("img_paths.pkl", "rb") as im_file:
            image_paths = pickle.load(im_file)

        with open("embeddings.pkl", "rb") as file:
            embeddings = pickle.load(file)
        
        # Load search index
        index_path = "flatIndex.index"
        loaded_idx = ExactIndex.load(embeddings, image_paths, index_path)
        
        return yolo, loaded_idx
        
    except Exception as e:
        st.error(f"Error loading models: {e}")
        return None, None

with st.spinner('Please wait while your model is loading'):
    yolo, loaded_idx = load_models_and_data()
    
if yolo is None or loaded_idx is None:
    st.error("Failed to load required models. Please check your model files.")
    st.stop()

# --- Image Functions --- #
transformations = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        ])

def upload_image():
    st.markdown("""
    <div class="upload-card">
        <h3 style="font-family: 'Playfair Display', serif; color: #6b5b4a; font-weight: 600; margin-bottom: 1rem; text-shadow: 1px 1px 2px rgba(0,0,0,0.1);">Share Your Style</h3>
        <p style="font-family: 'Inter', sans-serif; color: #8b7355; font-weight: 500; font-size: 1.1rem; margin-bottom: 2rem;">Upload a fashion image to discover curated recommendations</p>
    </div>
    """, unsafe_allow_html=True)
    
    image_file = st.file_uploader(
        label='Choose your fashion image',
        type=['png', 'jpg', 'jpeg'],
        help="For best results, use images with clear backgrounds"
    )
    
    if image_file is not None:
        if image_file.type in ('image/png', 'image/jpeg'):
            st.markdown("""
            <div style="background: linear-gradient(135deg, #c2ada1, #dfd4cc); 
                        color: white; padding: 0.5rem 1rem; border-radius: 20px; 
                        text-align: center; font-family: 'Inter', sans-serif; 
                        font-size: 0.9rem; margin: 1rem 0;">
                ✨ Perfect! Your image is ready for analysis
            </div>
            """, unsafe_allow_html=True)
            return image_file
        else:
            st.markdown("""
            <div style="background: rgba(223, 212, 204, 0.5); 
                        color: #8b7355; padding: 0.5rem 1rem; border-radius: 20px; 
                        text-align: center; font-family: 'Inter', sans-serif; 
                        font-weight: 500; font-size: 1rem; margin: 1rem 0;">
                Please upload PNG or JPEG files only
            </div>
            """, unsafe_allow_html=True)

# --- Object Detection and Recommendations --- #
def main():
    object = upload_image()

    if object:
        prediction = False
        image_obj = Image.open(object)
        
        # Display uploaded image with elegant styling
        col1, col2, col3 = st.columns([1, 2, 1])
        with col2:
            st.markdown('<div class="image-container">', unsafe_allow_html=True)
            st.image(image_obj, caption="Your Fashion Selection", use_column_width=True)
            st.markdown('</div>', unsafe_allow_html=True)
        
        # Elegant analysis button
        st.markdown("<br>", unsafe_allow_html=True)
        col1, col2, col3 = st.columns([1, 1, 1])
        with col2:
            button = st.button('✨ Discover Recommendations', use_container_width=True)
        
        if button:
            with st.spinner("🔍 Analyzing your style with AI intelligence..."):
                image_array = np.array(image_obj)
                cropped_objs = yolo.crop_objects(image_array)
                if cropped_objs is not None:
                    prediction = True
                else:
                    st.markdown("""
                    <div class="elegant-card" style="text-align: center;">
                        <p style="font-family: 'Inter', sans-serif; color: #8b7355; font-weight: 500; font-size: 1.1rem;">
                        No fashion objects detected. Try an image with clearer fashion items.
                        </p>
                    </div>
                    """, unsafe_allow_html=True)

        if prediction:
            # Filter valid cropped objects
            valid_objs = []
            for obj in cropped_objs:
                if obj is not None and obj.size > 0:
                    # Check if object has valid dimensions
                    if len(obj.shape) == 3 and obj.shape[0] > 10 and obj.shape[1] > 10:
                        valid_objs.append(obj)

            if not valid_objs:
                st.markdown("""
                <div class="elegant-card" style="text-align: center;">
                    <p style="font-family: 'Inter', sans-serif; color: #dfd4cc;">
                    No valid fashion objects detected. Please try with a clearer image.
                    </p>
                </div>
                """, unsafe_allow_html=True)
            else:
                st.markdown(f"""
                <div style="background: linear-gradient(135deg, #c2ada1, #dfd4cc); 
                            color: white; padding: 1rem 2rem; border-radius: 25px; 
                            text-align: center; font-family: 'Inter', sans-serif; 
                            margin: 1.5rem 0;">
                    ✨ Successfully detected {len(valid_objs)} fashion object(s)!
                </div>
                """, unsafe_allow_html=True)

            # Optional: Show detected objects with elegant styling
            if st.checkbox("👁️ Show detected objects", value=False):
                st.markdown("""
                <h4 style="font-family: 'Playfair Display', serif; color: #c2ada1; 
                           text-align: center; margin: 2rem 0 1rem 0;">Detected Fashion Elements</h4>
                """, unsafe_allow_html=True)
                
                if len(valid_objs) == 1:
                    col1, col2, col3 = st.columns([1, 2, 1])
                    with col2:
                        st.markdown('<div class="image-container">', unsafe_allow_html=True)
                        st.image(valid_objs[0])
                        st.markdown('</div>', unsafe_allow_html=True)
                else:
                    # Display multiple objects in elegant grid
                    cols = st.columns(min(len(valid_objs), 4))
                    for i, obj in enumerate(valid_objs[:4]):
                        with cols[i % 4]:
                            st.markdown('<div class="image-container">', unsafe_allow_html=True)
                            st.image(obj, caption=f"Element {i+1}")
                            st.markdown('</div>', unsafe_allow_html=True)

            # Find recommendations with elegant loading
            with st.spinner("🎨 Curating your personalized recommendations..."):
                boards = []
                for i, obj in enumerate(valid_objs):
                    try:
                        embedding = extract_img(obj, transformations)
                        if embedding is not None:
                            selected_neighbor_paths = similar_img_search(embedding, loaded_idx)
                            if selected_neighbor_paths:
                                boards.extend(selected_neighbor_paths)
                    except Exception as e:
                        st.warning(f"Could not process object {i+1}: {e}")

                if boards:
                    # Remove duplicates while preserving order
                    unique_boards = []
                    seen = set()
                    for path in boards:
                        if path not in seen:
                            unique_boards.append(path)
                            seen.add(path)
                    
                    # Limit to reasonable number for display
                    display_boards = unique_boards[:20]  # Show max 20 for cleaner display
                    
                    # Elegant recommendations header
                    st.markdown(f"""
                    <div class="recommendations-header">
                        Your Curated Collection
                    </div>
                    <div style="text-align: center; font-family: 'Inter', sans-serif; 
                               color: #6b5b4a; font-weight: 500; font-size: 1.1rem; margin-bottom: 2rem;">
                        {len(display_boards)} handpicked recommendations by AI curation
                    </div>
                    """, unsafe_allow_html=True)
                    
                    try:
                        rec_fig = visualize_outfits(display_boards)
                        
                        # Style the matplotlib figure to match our aesthetic
                        rec_fig.patch.set_facecolor('#fdf5f4')
                        
                        st.pyplot(rec_fig, use_container_width=True)
                        
                    except Exception as e:
                        st.markdown(f"""
                        <div class="elegant-card" style="text-align: center;">
                            <p style="font-family: 'Inter', sans-serif; color: #dfd4cc;">
                            Error displaying recommendations: {e}
                            </p>
                        </div>
                        """, unsafe_allow_html=True)
                else:
                    st.markdown("""
                    <div class="elegant-card" style="text-align: center;">
                        <p style="font-family: 'Inter', sans-serif; color: #dfd4cc;">
                        No similar items found. Please try with a different fashion image.
                        </p>
                    </div>
                    """, unsafe_allow_html=True)

if __name__ == "__main__":
    main()