import streamlit as st
import onnxruntime as ort
import numpy as np
from PIL import Image
import torchvision.transforms as transforms

# Load the ONNX model
onnx_session = ort.InferenceSession("student_model_pruned_quantized.onnx")
input_name = onnx_session.get_inputs()[0].name

# Define the transformation for the image
transform = transforms.Compose([
    transforms.Resize((224, 224)),  # Resize image for the model
    transforms.ToTensor(),  # Convert image to tensor
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),  # Normalize
])

# Streamlit Web Interface
st.title("Plant Disease Detection Web App")

# Upload image button
uploaded_file = st.file_uploader("Upload a plant leaf image", type="jpg")

if uploaded_file is not None:
    # Open the image using PIL
    img = Image.open(uploaded_file)
    st.image(img, caption="Uploaded Image", use_column_width=True)

    # Preprocess the image
    img_tensor = transform(img).unsqueeze(0).numpy()

    # Run inference on the uploaded image
    output = onnx_session.run(None, {input_name: img_tensor.astype(np.float32)})
    
    # Get the prediction (index of the predicted class)
    prediction = np.argmax(output[0], axis=1)[0]

    # Display prediction result
    st.write(f"Predicted Disease Class: {prediction}")
