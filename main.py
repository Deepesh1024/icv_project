import streamlit as st
import numpy as np
from tensorflow.keras.preprocessing.image import img_to_array, load_img
from tensorflow.keras.models import load_model
from tensorflow.keras.optimizers import Adam
import matplotlib.pyplot as plt
from PIL import Image
import io
import os

# Set page configuration
st.set_page_config(
    page_title="AI Image Segmentation",
    page_icon="🎯",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# Enhanced Custom CSS for modern UI
st.markdown("""
    <style>
        /* Main app styling */
        .stApp {
            background: linear-gradient(to bottom right, #1a1a1a, #2d2d2d);
        }
        
        /* Header styling */
        .main-header {
            background: linear-gradient(90deg, #FF4B2B 0%, #FF416C 100%);
            padding: 2rem;
            border-radius: 15px;
            box-shadow: 0 8px 32px 0 rgba(31, 38, 135, 0.37);
            backdrop-filter: blur(4px);
            -webkit-backdrop-filter: blur(4px);
            border: 1px solid rgba(255, 255, 255, 0.18);
            margin-bottom: 2rem;
            text-align: center;
        }
        
        .main-header h1 {
            color: white;
            font-size: 3rem;
            font-weight: 700;
            margin: 0;
            text-shadow: 2px 2px 4px rgba(0,0,0,0.3);
        }
        
        /* Card styling */
        .css-12oz5g7 {
            padding: 2rem;
            border-radius: 15px;
            background: rgba(255, 255, 255, 0.05);
            box-shadow: 0 8px 32px 0 rgba(31, 38, 135, 0.37);
            backdrop-filter: blur(4px);
            -webkit-backdrop-filter: blur(4px);
            border: 1px solid rgba(255, 255, 255, 0.18);
            margin-bottom: 1rem;
        }
        
        /* Upload area styling */
        .uploadedFile {
            border: 2px dashed #FF416C;
            padding: 2rem;
            border-radius: 15px;
            background: rgba(255, 255, 255, 0.05);
            text-align: center;
            transition: all 0.3s ease;
        }
        
        .uploadedFile:hover {
            border-color: #FF4B2B;
            transform: translateY(-5px);
        }
        
        /* Button styling */
        .stButton>button {
            background: linear-gradient(90deg, #FF416C 0%, #FF4B2B 100%);
            color: white;
            padding: 0.8rem 2rem;
            border-radius: 10px;
            border: none;
            width: 100%;
            font-weight: 600;
            transition: all 0.3s ease;
            text-transform: uppercase;
            letter-spacing: 1px;
        }
        
        .stButton>button:hover {
            transform: translateY(-2px);
            box-shadow: 0 5px 15px rgba(255, 65, 108, 0.4);
        }
        
        /* Image container styling */
        .image-container {
            background: rgba(255, 255, 255, 0.05);
            padding: 1rem;
            border-radius: 15px;
            margin: 1rem 0;
        }
        
        /* Results section styling */
        .results-header {
            background: linear-gradient(90deg, #FF416C 0%, #FF4B2B 100%);
            color: white;
            padding: 1rem;
            border-radius: 10px;
            margin-bottom: 1rem;
            text-align: center;
        }
        
        /* Progress bar styling */
        .stProgress > div > div > div {
            background-color: #FF416C;
        }
        
        /* Text styling */
        h1, h2, h3 {
            color: white !important;
        }
        
        p {
            color: #e0e0e0 !important;
        }
        
        /* Expander styling */
        .streamlit-expanderHeader {
            background: rgba(255, 255, 255, 0.05) !important;
            border-radius: 10px !important;
        }
        
        /* Download button styling */
        .stDownloadButton>button {
            background: linear-gradient(90deg, #4B4EE7 0%, #2B86C5 100%);
            color: white;
            padding: 0.8rem 2rem;
            border-radius: 10px;
            border: none;
            width: 100%;
            font-weight: 600;
            transition: all 0.3s ease;
        }
        
        .stDownloadButton>button:hover {
            transform: translateY(-2px);
            box-shadow: 0 5px 15px rgba(43, 134, 197, 0.4);
        }
    </style>
    """, unsafe_allow_html=True)

@st.cache_resource
def load_model_cache():
    try:
        model = load_model("unet_resnet50_model.keras")
    except:
        try:
            model = load_model("unet_resnet50_model.h5")
        except:
            model = TFSMLayer(
                "unet_resnet50_model",
                call_endpoint='serving_default'
            )
    model.compile(
        optimizer=Adam(learning_rate=0.001),
        loss='binary_crossentropy',
        metrics=['accuracy']
    )
    return model

def process_image(image):
    if image.mode != 'RGB':
        image = image.convert('RGB')
    image = image.resize((256, 256))
    image_array = img_to_array(image) / 255.0
    return image_array

def generate_segmentation(model, image_array):
    input_array_expanded = np.expand_dims(image_array, axis=0)
    predictions = model.predict(input_array_expanded, verbose=0)
    output_mask = predictions[0, :, :, 0] > 0.5
    overlay = image_array.copy()
    overlay[..., 0] = np.where(output_mask == 1, 1, overlay[..., 0])
    overlay[..., 1:] = np.where(output_mask[..., None] == 1, 0, overlay[..., 1:])
    return output_mask, overlay

def create_figure(original, mask, overlay):
    plt.style.use('dark_background')
    fig = plt.figure(figsize=(15, 5))
    
    # Plot original image
    plt.subplot(131)
    plt.title('Original Image', pad=20, color='white', fontsize=12)
    plt.imshow(original)
    plt.axis('off')
    
    # Plot binary mask
    plt.subplot(132)
    plt.title('Segmentation Mask', pad=20, color='white', fontsize=12)
    plt.imshow(mask, cmap='magma')
    plt.axis('off')
    
    # Plot overlay
    plt.subplot(133)
    plt.title('Overlay Visualization', pad=20, color='white', fontsize=12)
    plt.imshow(overlay)
    plt.axis('off')
    
    plt.tight_layout()
    buf = io.BytesIO()
    plt.savefig(buf, format='png', bbox_inches='tight', dpi=300, pad_inches=0.1,
                facecolor='#1a1a1a', edgecolor='none')
    plt.close()
    buf.seek(0)
    return buf

def main():
    # Header
    st.markdown("""
        <div class="main-header">
            <h1>🎯 AI Image Segmentation</h1>
        </div>
    """, unsafe_allow_html=True)
    
    # Load model
    with st.spinner('Initializing AI Model...'):
        model = load_model_cache()
    
    # Create two columns for layout
    col1, col2 = st.columns([1, 1])
    
    with col1:
        st.markdown("""
            <div style='background: rgba(255, 255, 255, 0.05); padding: 2rem; border-radius: 15px;'>
                <h3 style='color: white; text-align: center;'>Upload Your Image</h3>
            </div>
        """, unsafe_allow_html=True)
        
        uploaded_file = st.file_uploader("", type=['png', 'jpg', 'jpeg', 'webp'])
        
        if uploaded_file is not None:
            image = Image.open(uploaded_file)
            st.image(image, use_column_width=True, caption="Original Image")
            
            if st.button("✨ Generate Segmentation"):
                with st.spinner('AI is processing your image...'):
                    # Add a progress bar
                    progress_bar = st.progress(0)
                    for i in range(100):
                        progress_bar.progress(i + 1)
                    
                    image_array = process_image(image)
                    mask, overlay = generate_segmentation(model, image_array)
                    result_buf = create_figure(image_array, mask, overlay)
                    
                    with col2:
                        st.markdown("""
                            <div class='results-header'>
                                <h3 style='margin: 0;'>AI Segmentation Results</h3>
                            </div>
                        """, unsafe_allow_html=True)
                        st.image(result_buf, use_column_width=True)
                        
                        # Download section with custom styling
                        st.markdown("""
                            <div style='background: rgba(255, 255, 255, 0.05); padding: 1rem; border-radius: 15px; margin-top: 1rem;'>
                                <h4 style='color: white; text-align: center; margin-bottom: 1rem;'>Save Your Results</h4>
                            </div>
                        """, unsafe_allow_html=True)
                        
                        st.download_button(
                            label="📥 Download Results",
                            data=result_buf,
                            file_name="ai_segmentation_result.png",
                            mime="image/png"
                        )
    
    # Information section
    st.markdown("<br>", unsafe_allow_html=True)
    with st.expander("ℹ️ About This AI Tool"):
        st.markdown("""
            <div style='background: rgba(255, 255, 255, 0.05); padding: 2rem; border-radius: 15px;'>
                <h3 style='color: white;'>How to Use This AI Tool</h3>
                <p style='color: #e0e0e0;'>
                    1. Upload your image using the upload area above<br>
                    2. Click the "Generate Segmentation" button<br>
                    3. View your results in real-time<br>
                    4. Download the processed images if desired
                </p>
                <h3 style='color: white;'>Technical Details</h3>
                <p style='color: #e0e0e0;'>
                    This application uses state-of-the-art AI technology with a U-Net architecture 
                    and ResNet50 backbone to perform advanced image segmentation.
                </p>
            </div>
        """, unsafe_allow_html=True)

if __name__ == "__main__":
    main()