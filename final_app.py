import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from PIL import Image

import tensorflow as tf
from tensorflow.keras.preprocessing import image
from tensorflow.keras.models import load_model

import streamlit as st

# ================================================================================================================================================ #

# Set default layout to wide mode
st.set_page_config(layout="wide")

# Load models
@st.cache_resource

def load_models():
    model_classification_path = "classification_model.keras"
    model_segmentation_path = "segmentation_model.keras"
    model_classification = load_model(model_classification_path)
    model_segmentation = load_model(model_segmentation_path)
    return model_classification, model_segmentation

model_classification, model_segmentation = load_models()

# ================================================================================================================================================ #

# List of breed names from the index row
breed_names = [
    "Abyssinian", "Bengal", "Birman", "Bombay", "British_Shorthair", 
    "Egyptian_Mau", "Maine_Coon", "Persian", "Ragdoll", "Russian_Blue", 
    "Siamese", "Sphynx", "american_bulldog", "american_pit_bull_terrier", 
    "basset_hound", "beagle", "boxer", "chihuahua", "english_cocker_spaniel", 
    "english_setter", "german_shorthaired", "great_pyrenees", "havanese", 
    "japanese_chin", "keeshond", "leonberger", "miniature_pinscher", 
    "newfoundland", "pomeranian", "pug", "saint_bernard", "samoyed", 
    "scottish_terrier", "shiba_inu", "staffordshire_bull_terrier", 
    "wheaten_terrier", "yorkshire_terrier"
]

# Create a dictionary with breed names as keys and indices as values
breed_index_dict = {breed: index for index, breed in enumerate(breed_names)}

# ================================================================================================================================================ #

def preprocess_image_classification(img):
    
    # Ensure the image is in RGB mode
    img = img.convert("RGB")
    # Resize the image to match the model's expected input size
    img = img.resize((384, 384))
    # Convert the image to an array
    img_array = np.array(img)
    # Rescale the image (normalization)
    img_array = img_array / 255.0
    # Expand dimensions to match the model's input shape (1, 384, 384, 3)
    img_array = np.expand_dims(img_array, axis=0)

    return img_array

def get_key_by_value(dict, value):
    for key, val in dict.items():
        if val == value:
            return key
    return None  # If the value is not found

def predict_image_classification(model, img_path):
    
    # Preprocess the image
    preprocessed_img = preprocess_image_classification(img_path)
    # Get prediction possibilities matrix
    prediction_prob = model.predict(preprocessed_img)
    breed_names = list(breed_index_dict.keys())
    prediction_prob_df = pd.DataFrame(data=prediction_prob.T, index=breed_names, columns=['Probability']) #.rename_axis('Breed')
    # You can get the predicted class INDEX like this
    predicted_class_index = np.argmax(prediction_prob, axis=-1)
    # You can get the predicted class LABEL like this
    predicted_class_label = get_key_by_value(breed_index_dict, predicted_class_index)
    
    return predicted_class_index, predicted_class_label, prediction_prob, prediction_prob_df

# ================================================================================================================================================ #

def preprocess_image_segmentation(img):
    
    # Ensure the image is in RGB mode
    img = img.convert("RGB")
    # Resize the image to match the model's expected input size
    img = img.resize((224, 224))
    # Convert the image to an array
    img_array = np.array(img)
    # Rescale the image (normalization)
    img_array = img_array / 255.0
    # Expand dimensions to match the model's input shape (1, 224, 224, 3)
    img_array = np.expand_dims(img_array, axis=0)

    return img_array

def display_prediction(image, pred_mask):
    fig, axes = plt.subplots(1, 2, figsize=(10, 5))

    # Remove the batch dimension from the image and pred_mask
    image = np.squeeze(image, axis=0)
    pred_mask = np.squeeze(pred_mask, axis=0)

    axes[0].imshow(image)
    axes[0].set_title('Original Image (224 x 224)')
    axes[0].axis('off')

    # Convert one-hot encoded masks to class labels if necessary
    if pred_mask.ndim == 3:  # pred_mask has shape (224, 224, 3)
        pred_mask = np.argmax(pred_mask, axis=-1)

    axes[1].imshow(pred_mask, cmap='gray')
    axes[1].set_title('Pred Mask (224 x 224)')
    axes[1].axis('off')

    plt.tight_layout()
    return fig

def predict_image_segmentation(model, img_path):
    preprocessed_img = preprocess_image_segmentation(img)
    y_pred = model.predict(preprocessed_img)
    fig = display_prediction(preprocessed_img, y_pred)
    st.pyplot(fig)

# ================================================================================================================================================ #

# Streamlit Title
st.title("Cat / Dog Breed Classification & Image Segmentation")

# Image Upload Seection
uploaded_image = st.file_uploader("Upload an image...", type=["jpg", "jpeg", "png"])

if uploaded_image is not None:
    # Open and display the image
    img = Image.open(uploaded_image)
    st.image(img, caption='User Uploaded Image', width=400) # use_column_width=True
    
    # Extract and display additional information
    file_name = uploaded_image.name
    file_size = uploaded_image.size  # Size in bytes
    width, height = img.size  # Get image dimensions

    # Create columns
    col1, col2 = st.columns([1, 2.5])

    with col1:
        # Image Data
        st.markdown("<h2 style='font-size: 20px;'>Image Metadata:</h2>", unsafe_allow_html=True)
        st.write(f"**File Name:** {file_name}")
        st.write(f"**File Size:** {file_size / 1024:.2f} KB")  # Convert size to KB
        st.write(f"**Dimensions:** {width} x {height}")  # Display dimensions in width x height
        
        # Classification
        class_index, class_label, class_prob, class_prob_df = predict_image_classification(model_classification, img)

        class_prob_df['Probability'] = class_prob_df['Probability'].apply(lambda x: f"{x:.15f}")
        label_category = "Cat" if class_label[0].isupper() else "Dog"

        st.markdown("<h2 style='font-size: 20px;'>Classification:</h2>", unsafe_allow_html=True)
        st.write(f"**Predicted Class Index:** {class_index}")
        st.write(f"**Predicted Class Type:** {label_category}")
        st.write(f"**Predicted Class Label:** {class_label}")
        st.write("**Prediction Probabilities:**")
        st.dataframe(class_prob_df, use_container_width=False, width=800)

    with col2:
        # Segmentation
        st.markdown("<h2 style='font-size: 20px;'>Segmentation:</h2>", unsafe_allow_html=True)
        predict_image_segmentation(model_segmentation, img)

    # Set custom font size for "End of Page."
    st.write("")
    st.markdown(
        "<h2 style='text-align: center; font-size: 24px;'>End of Page.</h2>",
        unsafe_allow_html=True
    )
else:
    st.write("Please upload an image!")
