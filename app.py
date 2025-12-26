import streamlit as st # pyright: ignore[reportMissingImports]
import cv2 # pyright: ignore[reportMissingImports]
import numpy as np # pyright: ignore[reportMissingImports]
import matplotlib.pyplot as plt # pyright: ignore[reportMissingModuleSource]
from PIL import Image # pyright: ignore[reportMissingImports]

# --- BACKEND FUNCTIONS ---

def adjust_gamma(image, gamma=1.0):
    """
    Applies Gamma Correction to an image.
    Formula: O = ((I / 255) ^ gamma) * 255
    """
    # Build a lookup table mapping the pixel values [0, 255] to their new adjusted values
    invGamma = 1.0 / gamma
    table = np.array([((i / 255.0) ** invGamma) * 255
                      for i in np.arange(0, 256)]).astype("uint8")

    # Apply gamma correction using the lookup table
    return cv2.LUT(image, table)

def plot_histogram(image, title):
    """
    Generates a histogram for the grayscale image.
    """
    fig, ax = plt.subplots()
    ax.hist(image.ravel(), 256, [0, 256], color='gray')
    ax.set_title(title)
    ax.set_xlim([0, 256])
    ax.axis('off')  # Turn off axis for cleaner look
    return fig

# --- FRONTEND INTERFACE (Streamlit) ---

st.set_page_config(page_title="MedGamm - Medical Enhancement", layout="wide")

st.title("🏥 MedGamm: Medical Imaging Diagnostic Tool")
st.markdown("""
**Project 14 Implementation: Gamma Correction in Medical Analysis**
This tool simulates how the human eye perceives contrast versus digital sensors. 
It allows radiologists to adjust the **Gamma value** to reveal hidden details in X-rays or MRI scans.
""")

# Sidebar for controls
st.sidebar.header("Control Panel")
uploaded_file = st.sidebar.file_uploader("Upload Medical Image (X-ray/MRI)", type=["jpg", "png", "jpeg", "tif"])

if uploaded_file is not None:
    # Convert the file to an opencv image
    image = Image.open(uploaded_file).convert('L') # Convert to grayscale
    img_array = np.array(image)

    # 1. Slider for Gamma (Questions 1 & 3 of Project 14)
    gamma_val = st.sidebar.slider("Gamma Correction Value (γ)", min_value=0.1, max_value=3.0, value=1.0, step=0.1)

    # 2. Apply Gamma (Backend processing)
    processed_img = adjust_gamma(img_array, gamma=gamma_val)

    # 3. Display Images (Comparison)
    col1, col2 = st.columns(2)

    with col1:
        st.subheader("Original X-Ray/Scan")
        st.image(img_array, caption="Linear Response (Sensor)", use_container_width=True)
        # 4. Plot Histogram (Question 2 of Project 14)
        st.pyplot(plot_histogram(img_array, "Original Histogram"))

    with col2:
        st.subheader(f"Enhanced Diagnosis (γ={gamma_val})")
        st.image(processed_img, caption="Logarithmic Correction (Perceptual)", use_container_width=True)
        # Plot Histogram
        st.pyplot(plot_histogram(processed_img, "Corrected Histogram"))

    # 5. Analysis / Medical Insight (Question 4 & 5 of Project 14)
    st.divider()
    st.subheader("🩺 Diagnostic Analysis")
    
    if gamma_val < 1.0:
        st.info(f"**Gamma < 1 ({gamma_val}):** This effectively stretches the dark intervals. Use this to detect **fractures in shadowed areas** or details in underexposed X-rays.")
    elif gamma_val > 1.0:
        st.warning(f"**Gamma > 1 ({gamma_val}):** This stretches the bright intervals. Use this if the scan is washed out (overexposed) to see **tissue density** better.")
    else:
        st.success("**Gamma = 1:** No correction applied. This is the raw sensor data.")

    # Data for the "Report" requirement
    st.markdown("---")
    st.caption(f"Image Statistics | Min Intensity: {np.min(processed_img)} | Max Intensity: {np.max(processed_img)}")

else:
    st.info("Please upload a medical image (X-ray, CT, MRI) to begin the analysis.")
    st.markdown("### Why Gamma Correction?")
    st.write("The human eye follows a logarithmic response, while medical sensors are linear. Gamma correction bridges this gap, ensuring that what a doctor sees on screen matches the physical reality of the tissue density.")