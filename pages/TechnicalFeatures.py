import streamlit as st

st.set_page_config(
    page_title="Technical Details - Élégant Fashion AI",
    page_icon="⚙️",
    layout="wide"
)

# Apply the same elegant styling
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Playfair+Display:wght@300;400;500;600;700&family=Inter:wght@200;300;400;500&display=swap');
    
    .stApp {
        background: linear-gradient(135deg, #fdf5f4 0%, #fff1ef 100%);
        font-family: 'Inter', sans-serif;
    }
    
    .tech-title {
        font-family: 'Playfair Display', serif;
        font-size: 3rem;
        font-weight: 600;
        color: #6b5b4a;
        text-align: center;
        margin-bottom: 0.5rem;
        letter-spacing: 2px;
        text-shadow: 1px 1px 2px rgba(0,0,0,0.1);
    }
    
    .tech-subtitle {
        font-family: 'Inter', sans-serif;
        font-size: 1.2rem;
        font-weight: 500;
        color: #8b7355;
        text-align: center;
        margin-bottom: 2rem;
        letter-spacing: 1px;
    }
    
    .elegant-card {
        background: rgba(255, 241, 239, 0.7);
        backdrop-filter: blur(10px);
        border-radius: 20px;
        border: 1px solid rgba(239, 226, 224, 0.3);
        padding: 2rem;
        margin: 1rem 0;
        box-shadow: 0 8px 32px rgba(194, 173, 161, 0.1);
    }
    
    .tech-section {
        font-family: 'Playfair Display', serif;
        font-size: 1.6rem;
        font-weight: 600;
        color: #6b5b4a;
        margin: 2rem 0 1rem 0;
        letter-spacing: 1px;
        text-shadow: 1px 1px 2px rgba(0,0,0,0.1);
    }
    
    .tech-feature {
        font-family: 'Inter', sans-serif;
        font-size: 1.1rem;
        color: #6b5b4a;
        font-weight: 500;
        line-height: 1.8;
        margin: 0.5rem 0;
    }
    
    .metric-card {
        background: linear-gradient(135deg, #c2ada1, #dfd4cc);
        color: white;
        padding: 1.5rem;
        border-radius: 20px;
        text-align: center;
        margin: 1rem 0;
    }
    
    .image-container {
        border-radius: 16px;
        overflow: hidden;
        box-shadow: 0 8px 32px rgba(194, 173, 161, 0.15);
        margin: 2rem 0;
    }
</style>
""", unsafe_allow_html=True)

st.markdown("""
<div style="text-align: center; padding: 2rem 0;">
    <h1 class="tech-title">Technical Architecture</h1>
    <p class="tech-subtitle">Behind the scenes of intelligent fashion curation</p>
</div>
""", unsafe_allow_html=True)

st.markdown("""
<div class="elegant-card">
    <p style="font-family: 'Inter', sans-serif; color: #6b5b4a; font-weight: 500; line-height: 1.8; font-size: 1.2rem;">
    Our system employs cutting-edge computer vision technology to analyze fashion images and deliver personalized style recommendations. 
    The Joan Kusuma +Layer CNN architecture achieves 66.7% mAP@5 performance across 21 fashion categories.
    </p>
</div>
""", unsafe_allow_html=True)

# System overview with elegant styling
st.markdown('<h2 class="tech-section">🔄 System Architecture</h2>', unsafe_allow_html=True)
st.markdown('<div class="image-container">', unsafe_allow_html=True)
st.image('images/flowcharts/serving_stg.png', use_column_width=True)
st.markdown('</div>', unsafe_allow_html=True)
st.markdown("""
<div class="elegant-card">
    <p class="tech-feature">
    When you upload an image, our YOLOv5 model detects individual fashion objects, 
    extracts 512-dimensional embeddings using Joan Kusuma's architecture, and performs 
    similarity search across 6,778 curated fashion items.
    </p>
</div>
""", unsafe_allow_html=True)

# Vector index with elegant styling
st.markdown('<h2 class="tech-section">🧠 Intelligence Engine</h2>', unsafe_allow_html=True)
st.markdown('<div class="image-container">', unsafe_allow_html=True)
st.image('images/flowcharts/vector_index.png', use_column_width=True)
st.markdown('</div>', unsafe_allow_html=True)
st.markdown("""
<div class="elegant-card">
    <p class="tech-feature">
    Our FAISS vector index stores high-dimensional embeddings from every catalog item, 
    enabling lightning-fast similarity searches using L2 distance metrics for precise style matching.
    </p>
</div>
""", unsafe_allow_html=True)

# Performance metrics
st.markdown('<h2 class="tech-section">📊 Performance Excellence</h2>', unsafe_allow_html=True)
col1, col2, col3 = st.columns(3)

with col1:
    st.markdown("""
    <div class="metric-card">
        <h3 style="font-family: 'Playfair Display', serif; margin-bottom: 0.5rem;">66.7%</h3>
        <p style="font-family: 'Inter', sans-serif; font-size: 0.9rem; margin: 0;">mAP@5 Accuracy</p>
    </div>
    """, unsafe_allow_html=True)

with col2:
    st.markdown("""
    <div class="metric-card">
        <h3 style="font-family: 'Playfair Display', serif; margin-bottom: 0.5rem;">6,778</h3>
        <p style="font-family: 'Inter', sans-serif; font-size: 0.9rem; margin: 0;">Fashion Items</p>
    </div>
    """, unsafe_allow_html=True)

with col3:
    st.markdown("""
    <div class="metric-card">
        <h3 style="font-family: 'Playfair Display', serif; margin-bottom: 0.5rem;">21</h3>
        <p style="font-family: 'Inter', sans-serif; font-size: 0.9rem; margin: 0;">Categories</p>
    </div>
    """, unsafe_allow_html=True)

# Technical features with elegant styling
st.markdown('<h2 class="tech-section">⚙️ Core Technologies</h2>', unsafe_allow_html=True)

st.markdown("""
<div class="elegant-card">
    <div class="tech-feature">
        <strong style="color: #c2ada1;">🎯 Joan Kusuma +Layer CNN:</strong> 
        Advanced autoencoder with 8 convolutional layers and 4 max-pooling operations, 
        generating 512-dimensional embeddings that exceed research benchmarks by 25.9%.
    </div>
    <div class="tech-feature">
        <strong style="color: #c2ada1;">👁️ YOLOv5 Object Detection:</strong> 
        State-of-the-art model trained specifically on fashion imagery, 
        achieving precise detection and cropping of fashion objects.
    </div>
    <div class="tech-feature">
        <strong style="color: #c2ada1;">⚡ FAISS Vector Search:</strong> 
        Facebook's optimized similarity search library enabling sub-millisecond 
        retrieval across thousands of high-dimensional fashion embeddings.
    </div>
    <div class="tech-feature">
        <strong style="color: #c2ada1;">🎨 Smart Curation:</strong> 
        Category-aware recommendation system with 2.7 average same-category matches 
        and intelligent diversity balancing across 21 fashion types.
    </div>
</div>
""", unsafe_allow_html=True)
