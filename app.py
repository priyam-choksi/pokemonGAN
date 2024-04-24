import streamlit as st
import pickle
import json
import numpy as np
from PIL import Image

# Step 1: Load the Model
@st.cache(allow_output_mutation=True)
def load_model(model_path):
    with open(model_path, 'rb') as f:
        model = pickle.load(f)
    return model

# Step 2: Preprocess the Configuration
def preprocess_config(config_path):
    with open(config_path, 'r') as f:
        config = json.load(f)
    # Add any preprocessing steps if needed
    return config

# Step 3: Create a Streamlit App
def main():
    st.title("Random Pokémon Generator")

    # Load the model
    model_path = 'pokemon_generator_model.pkl'
    model = load_model(model_path)

    # Preprocess the configuration
    config_path = 'model_config.json'
    model_config = preprocess_config(config_path)

    # Generate Pokémon Images
    if st.button("Generate Random Pokémon"):
        latent_vector = np.random.randn(model_config['latent_dim'])
        generated_image = model.generate(latent_vector)  # You need to adjust this based on how your model generates images
        generated_image = np.clip(generated_image, 0, 255).astype(np.uint8)
        st.image(Image.fromarray(generated_image), caption='Random Pokémon')

if __name__ == "__main__":
    main()
