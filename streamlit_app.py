import streamlit as st
import tensorflow as tf
import pandas as pd
import numpy as np
from PIL import Image
import gdown
from tensorflow.keras import layers, Model, Input, applications

# Download the model weights
url1 = 'https://drive.google.com/uc?id=1bAdiAlpfv5oG3V9Nfp7wjoHjclwSaGWN'
output1 = 'xception_model_weights.h5'

gdown.download(url1, output1, quiet=False)

import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)

st.title("Pneumonia Detection Image Classifier")
st.text("Upload a Chest X-ray Image for Pneumonia Detection")

def load_model_xception():
    # Define the model architecture
    xcep_base = applications.Xception(weights='imagenet', include_top=False, input_shape=(150, 150, 3))
    xcep_base.trainable = False

    inputs = Input(shape=(150, 150, 3))
    x = xcep_base(inputs, training=False)
    x = layers.GlobalAveragePooling2D()(x)
    x = layers.Dense(1024, activation='relu')(x)
    x = layers.Dropout(0.5)(x)
    outputs = layers.Dense(2, activation='sigmoid')(x)
    xcep_model = Model(inputs, outputs)

    # Compile the model with the same configuration
    xcep_model.compile(
        optimizer=tf.keras.optimizers.Adam(),
        loss=tf.keras.losses.BinaryCrossentropy(from_logits=True),
        metrics=[tf.keras.metrics.CategoricalAccuracy()],
    )
    # Load the model weights
    xcep_model.load_weights(output1)
    return xcep_model

with st.spinner('Loading Model Into Memory....'):
    model_xception = load_model_xception()

st.write("Model Loaded Successfully!")

classes = ['Normal', 'Pneumonia']

def decode_img(image):
    img = tf.image.decode_jpeg(image, channels=3)
    img = tf.image.resize(img, [150, 150])
    img = img / 255.0  # Normalizing the image
    return np.expand_dims(img, axis=0)

uploaded_file = st.file_uploader("Choose a Chest X-ray Image...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    try:
        image = Image.open(uploaded_file)
        st.image(image, caption='Uploaded Image', use_column_width=True)

        st.write("Predicted Class and Probabilities:")
        with st.spinner('Classifying...'):
            image_array = decode_img(uploaded_file.getvalue())
            prediction = model_xception.predict(image_array)

            mean_probabilities = prediction[0]
            label = np.argmax(mean_probabilities)

        st.write(f"Predicted: {classes[label]}")
        
        
        # Display probabilities for each class
        probabilities_df = pd.DataFrame({
            'Class': classes,
            'Probability': mean_probabilities
        })
        st.write("Class Probabilities:")
        st.table(probabilities_df)

        
    except Exception as e:
        st.error(f"Error: {e}")

st.text("This project is developed by Swayam Mohanty")
