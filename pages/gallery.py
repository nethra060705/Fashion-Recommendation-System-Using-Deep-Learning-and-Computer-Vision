import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg

from src.utilities import viz_thumbnail

st.set_page_config(
    page_title="Gallery - Élégant Fashion AI",
    page_icon="🖼️",
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
    
    .gallery-title {
        font-family: 'Playfair Display', serif;
        font-size: 3rem;
        font-weight: 600;
        color: #6b5b4a;
        text-align: center;
        margin-bottom: 0.5rem;
        letter-spacing: 2px;
        text-shadow: 1px 1px 2px rgba(0,0,0,0.1);
    }
    
    .gallery-subtitle {
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
    
    .image-container {
        border-radius: 16px;
        overflow: hidden;
        box-shadow: 0 8px 32px rgba(194, 173, 161, 0.15);
        margin: 1rem 0;
    }
    
    .section-header {
        font-family: 'Playfair Display', serif;
        font-size: 1.4rem;
        font-weight: 600;
        color: #6b5b4a;
        margin: 1rem 0;
        letter-spacing: 1px;
        text-shadow: 1px 1px 2px rgba(0,0,0,0.1);
    }
</style>
""", unsafe_allow_html=True)

st.markdown("""
<div style="text-align: center; padding: 2rem 0;">
    <h1 class="gallery-title">Style Gallery</h1>
    <p class="gallery-subtitle">Explore curated fashion recommendations in action</p>
</div>
""", unsafe_allow_html=True)

# Load your images
# pink-white
pw_path = 'gallery/sample_query/pink-white/pw_1.jpg'

# black-coat
bc_path = 'gallery/sample_query/black-coat/bc_1.jpg'

# sweater-skirt
ss_path = 'gallery/sample_query/sweater-skirt/ss_1.jpg'

# black-jacket
bk_path = 'gallery/sample_query/black-jacket/bk_1.jpg'

#Set the size for thumbnail
thumbnail_size = (10, 10)

tab1, tab2, tab3, tab4 = st.tabs(['✨ Pink & White', '🖤 Black Coat', '🧥 Sweater Look', '🧥 Black Jacket'])

with tab1:
    st.markdown('<h4 class="section-header">Original Style</h4>', unsafe_allow_html=True)
    col1, col2, col3 = st.columns([1, 2, 1])
    with col2:
        st.markdown('<div class="image-container">', unsafe_allow_html=True)
        fig1 = viz_thumbnail(pw_path, thumbnail_size)
        fig1.patch.set_facecolor('#fdf5f4')
        st.pyplot(fig1)
        st.markdown('</div>', unsafe_allow_html=True)
    
    st.markdown('<h4 class="section-header">AI Curated Matches</h4>', unsafe_allow_html=True)
    st.markdown('<div class="image-container">', unsafe_allow_html=True)
    st.image('gallery/sample_results/pink-white/pw_im1.png')
    st.markdown('</div>', unsafe_allow_html=True)
    st.markdown('<div class="image-container">', unsafe_allow_html=True)
    st.image('gallery/sample_results/pink-white/pw_im2.png')
    st.markdown('</div>', unsafe_allow_html=True)

with tab2:
    st.markdown('<h4 class="section-header">Original Style</h4>', unsafe_allow_html=True)
    col1, col2, col3 = st.columns([1, 2, 1])
    with col2:
        st.markdown('<div class="image-container">', unsafe_allow_html=True)
        fig2 = viz_thumbnail(bc_path, thumbnail_size)
        fig2.patch.set_facecolor('#fdf5f4')
        st.pyplot(fig2)
        st.markdown('</div>', unsafe_allow_html=True)
    
    st.markdown('<h4 class="section-header">AI Curated Matches</h4>', unsafe_allow_html=True)
    st.markdown('<div class="image-container">', unsafe_allow_html=True)
    st.image('gallery/sample_results/black-coat/bc_im1.png')
    st.markdown('</div>', unsafe_allow_html=True)
    st.markdown('<div class="image-container">', unsafe_allow_html=True)
    st.image('gallery/sample_results/black-coat/bc_im2.png')
    st.markdown('</div>', unsafe_allow_html=True)
    
with tab3:
    st.markdown("#### :rainbow[Query Image:]")
    fig3 = viz_thumbnail(ss_path, thumbnail_size)
    st.pyplot(fig3)
    st.divider()
    st.markdown("#### :rainbow[Recommendations:]")
    st.image('gallery/sample_results/sweater/ss_im1.png')
    st.image('gallery/sample_results/sweater/ss_im2.png')

with tab4:
    st.markdown("#### :rainbow[Query Image:]")
    fig4 = viz_thumbnail(bk_path, thumbnail_size)
    st.pyplot(fig4)
    st.divider()
    st.markdown("#### :rainbow[Recommendations:]")
    st.image('gallery/sample_results/black-jacket/bk_im1.png')
    st.image('gallery/sample_results/black-jacket/bk_im2.png')