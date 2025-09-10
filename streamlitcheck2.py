import streamlit as st
import cv2
import numpy as np
from skimage.metrics import structural_similarity as ssim
from skimage.feature import graycomatrix, graycoprops
from PIL import Image
import os

def compute_laplacian_variance(image):
    laplacian = cv2.Laplacian(image, cv2.CV_64F)
    return laplacian.var()

#def compute_ssim(image1, image2):
    #return ssim(image1, image2)

def compute_contrast(image):
    min_intensity, max_intensity = np.min(image), np.max(image)
    contrast_value = (max_intensity - min_intensity) / (max_intensity + min_intensity) if max_intensity + min_intensity != 0 else 0
    return contrast_value

def compute_segmented_brightness(image):
    return np.mean(image)


def compute_glcm_features(image):
    glcm = graycomatrix(image, distances=[1], angles=[0], levels=256, symmetric=True, normed=True)
    contrast = graycoprops(glcm, 'contrast')[0, 0]
    homogeneity = graycoprops(glcm, 'homogeneity')[0, 0]
    return contrast, homogeneity

def compute_sobel_sharpness(image):
    sobelX = cv2.Sobel(image, cv2.CV_64F, 1, 0, 3)
    sobelY = cv2.Sobel(image, cv2.CV_64F, 0, 1, 3)
    return np.mean(np.sqrt(sobelX**2 + sobelY**2))

def analyze_xray(image, mode):
    if image is None:
        return "Error: Unable to load image", "neutral"
    
    # Check bit depth
    if image.dtype == np.uint16:
        return "Error: Image is 16-bit, not 24-bit", "neutral"
    elif len(image.shape) == 3 and image.shape[2] == 3:
        bit_depth = 24
    else:
        bit_depth = 8
    
    if bit_depth != 24:
        return "Error: Image is not 24-bit", "neutral"
    
    image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    exi_value = compute_segmented_brightness(image)
    blurriness = compute_laplacian_variance(image)
    ContrastMicleson = compute_contrast(image)
    contrastGLCM, homogenity = compute_glcm_features(image)
    
    # Load reference image (relative path)
    reference_image_path = os.path.join(os.path.dirname(__file__), "ref_img", "GoodImageSpine.jpg")
    print(f"DEBUG: Current working directory: {os.getcwd()}")
    print(f"DEBUG: Reference image path: {reference_image_path}")
    print(f"DEBUG: Does reference image exist? {os.path.exists(reference_image_path)}")
    
    if not os.path.exists(reference_image_path):
        return "Error: Reference image not found", "neutral"
    
    reference_image = cv2.imread(reference_image_path, cv2.IMREAD_UNCHANGED)
    if reference_image is None:
        return "Error: Reference image cannot be loaded", "neutral"
    
    image = image.astype(np.float64) / 255.0
    reference_image = reference_image.astype(np.float64) / 255.0
    reference_image = cv2.resize(reference_image, (image.shape[1], image.shape[0]))
    
   # ssim_value = ssim(image, reference_image, data_range=1.0)
    sharpness = compute_sobel_sharpness(image)
    
    # Define intensity thresholds based on mode
    handselection = False
    lateralSpine = False
    
    if mode == "Spine":
        min_intensity, max_intensity = 120, 190
    elif mode == "Lateral Spine":
        lateralSpine = True
        min_intensity, max_intensity = 80, 125
    else:
        handselection= True
        print("hand selection true")
        min_intensity, max_intensity = 120, 210
    
    # Intensity Analysis
    if min_intensity <= exi_value <= max_intensity:
        highlight = "✅ Intensity is optimal"
    else:
        print("exival is ",exi_value)
        highlight = "❌ Intensity is not optimal"
        
    
    # Blurriness & Analysis
    
    
    if (lateralSpine == True):
        print("inside lateral spine checks ")
        if 200 <= blurriness < 280 and ContrastMicleson <= 1 and homogenity > 0.30 and sharpness < 20:
            quality = "✅ Great Image Quality (High Structural Similarity)"
            badge = "great"
        elif 280 <= blurriness < 400 and 20< sharpness <40:
            quality = "🟡 Good Image Quality"
            badge = "good"
        elif 400 <= blurriness < 700 and ContrastMicleson > 1 and sharpness > 40:
            quality = "🔴 Average Image Quality"
            badge = "average"
        else:
            quality = "⚫ Poor Image Quality"
            badge = "bad"
    elif(handselection== False and lateralSpine == False):
        print("inside spine selection")
        if 240 <= blurriness < 280 and ContrastMicleson <= 1 and homogenity > 0.40 and sharpness < 20:
            quality = "✅ Great Image Quality (High Structural Similarity)"
            badge = "great"
        elif 280 <= blurriness < 400 and 20< sharpness <40:
            quality = "🟡 Good Image Quality"
            badge = "good"
        elif 400 <= blurriness < 700 and ContrastMicleson > 1 and sharpness > 40:
            quality = "🔴 Average Image Quality"
            badge = "average"
        else:
            quality = "⚫ Poor Image Quality"
            badge = "bad"
    else:
        print("inside hand selection")
        if 245 <= blurriness <= 375 and ContrastMicleson <= 1.0 and homogenity > 0.25 and sharpness < 20:
            quality = "✅ Great Image Quality (High Structural Similarity)"
            badge = "great"
        elif 376 <= blurriness < 400 and 20 < sharpness < 30:
            quality = "🟡 Good Image Quality"
            badge = "good"
        elif 401 <= blurriness < 700 and ContrastMicleson > 1 and  31 < sharpness <40:
            quality = "🔴 Average Image Quality"
            badge = "average"
        else:
            quality = "⚫ Poor Image Quality"
            badge = "bad" 
            
    print(f"bluriness {blurriness} contrasmicleson{ContrastMicleson} homoginity {homogenity} sharpenss{sharpness}")
    result_text = f"{highlight}\n{quality}\n\n**Blurriness**: {blurriness:.2f}\n**Contrast Micleson**: {ContrastMicleson:.2f}\n**Homogeneity**: {homogenity:.2f}\n **sharpness**{sharpness:.2f}"
    return result_text, badge

# Streamlit UI
st.set_page_config(page_title="SharpScan", layout="wide")
st.title("SharpScan: X-ray Image Analysis")
st.markdown("Upload an X-ray image to analyze its quality with advanced metrics.")

# Custom CSS for fancy badges
st.markdown("""
    <style>
    .badge-great {
        background-color: #28a745;
        color: white;
        padding: 5px 10px;
        border-radius: 5px;
        font-weight: bold;
    }
    .badge-good {
        background-color: #ffc107;
        color: black;
            padding: 5px 10px;
            border-radius: 5px;
            font-weight: bold;
        }
        .badge-average {
            background-color: #dc3545;
            color: white;
            padding: 5px 10px;
            border-radius: 5px;
            font-weight: bold;
        }
        .badge-bad {
            background-color: #343a40;
            color: white;
            padding: 5px 10px;
            border-radius: 5px;
            font-weight: bold;
        }
        .badge-neutral {
            background-color: #6c757d;
            color: white;
            padding: 5px 10px;
            border-radius: 5px;
            font-weight: bold;
        }
        .result-box {
        background-color: #f8f9fa;
        padding: 20px;
        border-radius: 10px;
        box-shadow: 0 4px 8px rgba(0,0,0,0.1);
    }
    </style>
""", unsafe_allow_html=True)

# Create two columns: left for controls/results, right for image
col1, col2 = st.columns([2, 1])

with col1:
    # Mode selection
    mode = st.selectbox("Select X-ray Type:", ["Spine", "Lateral Spine", "Hand"], index=0)
    
    # Weight selection
    weight = st.selectbox("Select Patient Weight:", ["Below 90", "Above 90"], index=0)
    
    # Image upload
    uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "png", "jpeg", "bmp"])

with col2:
    if uploaded_file is not None:
        # Read and display image
        file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
        image = cv2.imdecode(file_bytes, cv2.IMREAD_UNCHANGED)
        st.image(image, caption="Uploaded X-ray Image", use_container_width=True)

# Analyze and display results
if uploaded_file is not None:
    with col1:
        result_text, badge = analyze_xray(image, mode)
        
        # Display results in a fancy box
        st.markdown("### Analysis Results")
        st.markdown('<div class="result-box">', unsafe_allow_html=True)
        st.markdown(result_text, unsafe_allow_html=True)
        
        # Display badge based on quality
        if badge == "great":
            st.markdown('<span class="badge-great">Great Quality</span>', unsafe_allow_html=True)
        elif badge == "good":
            st.markdown('<span class="badge-good">Good Quality</span>', unsafe_allow_html=True)
        elif badge == "average":
            st.markdown('<span class="badge-average">Average Quality</span>', unsafe_allow_html=True)
        elif badge == "bad":
            st.markdown('<span class="badge-bad">Poor Quality</span>', unsafe_allow_html=True)
        else:
            st.markdown('<span class="badge-neutral">Neutral</span>', unsafe_allow_html=True)
        
        st.markdown('</div>', unsafe_allow_html=True)
else:
    with col1:
        st.info("Please upload an image to start the analysis.")