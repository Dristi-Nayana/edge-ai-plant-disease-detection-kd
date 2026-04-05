import cv2
import numpy as np
import onnxruntime
import time

# Load the ONNX model
session = onnxruntime.InferenceSession("student_model_pruned_quantized.onnx")
input_name = session.get_inputs()[0].name
input_shape = session.get_inputs()[0].shape  # (1, 3, H, W)

# Open DroidCam video stream (Wi-Fi)
cap = cv2.VideoCapture("http://192.168.31.68:4747/video")

if not cap.isOpened():
    print("❌ Could not open video stream.")
    exit()

print("✅ Starting video stream...")

# List of class labels
labels = [
    'Raspberry___healthy', 'Tomato___Late_blight', 'Strawberry___Leaf_scorch', 
    'Pepper,_bell___healthy', 'Pepper,_bell___Bacterial_spot', 
    'Grape___Esca_(Black_Measles)', 'Strawberry___healthy', 'Apple___healthy',
    'Grape___healthy', 'Blueberry___healthy', 'Apple___Black_rot', 
    'Tomato___Spider_mites Two-spotted_spider_mite',
    'Tomato___Tomato_Yellow_Leaf_Curl_Virus', 'Corn___Cercospora_leaf_spot Gray_leaf_spot',
    'Tomato___Early_blight', 'Cherry___Powdery_mildew', 'Soybean___healthy',
    'Tomato___healthy', 'Corn___Northern_Leaf_Blight', 'Corn___Common_rust',
    'Peach___Bacterial_spot', 'Tomato___Target_Spot', 'Background_without_leaves',
    'Grape___Leaf_blight_(Isariopsis_Leaf_Spot)', 'Potato___Late_blight',
    'Tomato___Tomato_mosaic_virus', 'Potato___healthy', 'Potato___Early_blight',
    'Tomato___Leaf_Mold', 'Grape___Black_rot', 'Cherry___healthy', 
    'Tomato___Septoria_leaf_spot', 'Peach___healthy', 'Apple___Apple_scab',
    'Corn___healthy', 'Tomato___Bacterial_spot',
    'Orange___Haunglongbing_(Citrus_greening)', 'Apple___Cedar_apple_rust',
    'Squash___Powdery_mildew'
]

# Run inference loop
while True:
    start = time.time()

    ret, frame = cap.read()
    if not ret:
        print("❌ Failed to read frame.")
        break

    # Resize & normalize image
    img = cv2.resize(frame, (input_shape[2], input_shape[3]))
    img = img.astype(np.float32) / 255.0  # Normalize to 0–1
    img = np.transpose(img, (2, 0, 1))     # Channels first
    img = np.expand_dims(img, axis=0)     # Add batch dimension

    # Run model
    outputs = session.run(None, {input_name: img})
    pred_index = int(np.argmax(outputs[0]))
    label = labels[pred_index] if pred_index < len(labels) else "Unknown"

    end = time.time()
    elapsed = end - start
    fps = 1 / elapsed if elapsed > 0 else 0

    # Display prediction and FPS
    cv2.putText(frame, f"{label} ({fps:.2f} FPS)", (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 255), 2, cv2.LINE_AA)  # Yellow text

    # Show the frame
    cv2.imshow("Real-Time Prediction", frame)

    # Exit loop on 'q' key press
    if cv2.waitKey(1) == ord('q'):
        break

# Clean up
cap.release()
cv2.destroyAllWindows()
