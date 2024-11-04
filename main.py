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
    page_title="Brain Tumor Segmentation",
    page_icon="🧠",
    layout="wide"
)

# Custom CSS for styling
st.markdown("""
    <style>
        .stApp {
            max-width: 1200px;
            margin: 0 auto;
        }
        .uploadedFile {
            border: 2px dashed #4CAF50;
            padding: 20px;
            border-radius: 10px;
        }
        .stButton>button {
            background-color: #4CAF50;
            color: white;
            padding: 10px 20px;
            border-radius: 5px;
            border: none;
            width: 200px;
        }
        .title {
            text-align: center;
            color: #2C3E50;
            padding: 20px;
        }
    </style>
""", unsafe_allow_html=True)


@st.cache_resource
def load_model_cache():
    try:
    # For .keras format
        model = load_model("unet_resnet50_model.keras")
    except:
        try:
        # For .h5 format
            model = load_model("unet_resnet50_model.h5")
        except:
        # Option 2: If you have a SavedModel format
            model = TFSMLayer(
                "unet_resnet50_model",
                call_endpoint='serving_default'  # You might need to adjust this endpoint name
            )
    model.compile(
        optimizer=Adam(learning_rate=0.001),
        loss='binary_crossentropy',
        metrics=['accuracy']
    )
    return model

def process_image(image):
    # Convert to RGB if necessary
    if image.mode != 'RGB':
        image = image.convert('RGB')
    
    # Resize image
    image = image.resize((256, 256))
    
    # Convert to array and normalize
    image_array = img_to_array(image) / 255.0
    return image_array

def generate_segmentation(model, image_array):
    # Expand dimensions
    input_array_expanded = np.expand_dims(image_array, axis=0)
    
    # Make prediction
    predictions = model.predict(input_array_expanded, verbose=0)
    output_mask = predictions[0, :, :, 0] > 0.5
    
    # Create overlay
    overlay = image_array.copy()
    overlay[..., 0] = np.where(output_mask == 1, 1, overlay[..., 0])
    overlay[..., 1:] = np.where(output_mask[..., None] == 1, 0, overlay[..., 1:])
    
    return output_mask, overlay

def create_figure(original, mask, overlay):
    fig = plt.figure(figsize=(15, 5))
    
    # Plot original image
    plt.subplot(131)
    plt.title('Original Image', pad=20)
    plt.imshow(original)
    plt.axis('off')
    
    # Plot binary mask
    plt.subplot(132)
    plt.title('Tumor Segmentation Mask', pad=20)
    plt.imshow(mask, cmap='gray')
    plt.axis('off')
    
    # Plot overlay
    plt.subplot(133)
    plt.title('Overlay Visualization', pad=20)
    plt.imshow(overlay)
    plt.axis('off')
    
    # Convert plot to image
    buf = io.BytesIO()
    plt.savefig(buf, format='png', bbox_inches='tight', dpi=300, pad_inches=0.1)
    plt.close()
    buf.seek(0)
    return buf

def main():
    # Header
    st.markdown("<h1 class='title'>🧠 Brain Tumor Segmentation</h1>", unsafe_allow_html=True)
    
    # Load model
    with st.spinner('Loading model...'):
        model = load_model_cache()
    
    # File uploader
    st.markdown("### Upload CT Scan Image")
    uploaded_file = st.file_uploader("Choose a brain CT scan image...", type=['png', 'jpg', 'jpeg', 'webp'])
    
    if uploaded_file is not None:
        # Display original image
        image = Image.open(uploaded_file)
        
        # Create columns for layout
        col1, col2 = st.columns([1, 2])
        
        with col1:
            st.markdown("### Original Image")
            st.image(image, use_column_width=True)
        
        # Process button
        if st.button("Generate Segmentation"):
            with st.spinner('Processing image...'):
                # Process image
                image_array = process_image(image)
                mask, overlay = generate_segmentation(model, image_array)
                
                # Create and display figure
                result_buf = create_figure(image_array, mask, overlay)
                
                with col2:
                    st.markdown("### Segmentation Results")
                    st.image(result_buf, use_column_width=True)
                
                # Add download button
                st.download_button(
                    label="Download Results",
                    data=result_buf,
                    file_name="segmentation_results.png",
                    mime="image/png"
                )
    
    # Add information section
    with st.expander("ℹ️ About this app"):
        st.markdown("""
        This application uses a U-Net architecture with ResNet50 backbone to perform brain tumor segmentation on CT scan images.
        
        **How to use:**
        1. Upload a brain CT scan image
        2. Click 'Generate Segmentation'
        3. View the results showing:
           - Original image
           - Segmentation mask
           - Overlay visualization
        4. Download the results if needed
        
        **Note:** For best results, use clear CT scan images in common formats (PNG, JPG, JPEG, WEBP).
        """)

if __name__ == "__main__":
    main()