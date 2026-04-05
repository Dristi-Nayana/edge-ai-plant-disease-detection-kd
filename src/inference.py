import cv2
import numpy as np
import onnxruntime
import time

# Load the ONNX model
session = onnxruntime.InferenceSession("student_model_pruned_quantized.onnx")
input_name = session.get_inputs()[0].name
input_shape = session.get_inputs()[0].shape  # (1, 3, H, W)

# Open DroidCam video stream
cap = cv2.VideoCapture("http://192.168.31.68:4747/video")
if not cap.isOpened():
    print("❌ Could not open video stream.")
    exit()

print("✅ Starting video stream from DroidCam...")

# Class labels
labels = ['Raspberry___healthy', 'Tomato___Late_blight', 'Strawberry___Leaf_scorch', 'Pepper,_bell___healthy', 
          'Pepper,_bell___Bacterial_spot', 'Grape___Esca_(Black_Measles)', 'Strawberry___healthy', 'Apple___healthy',
          'Grape___healthy', 'Blueberry___healthy', 'Apple___Black_rot', 'Tomato___Spider_mites Two-spotted_spider_mite',
          'Tomato___Tomato_Yellow_Leaf_Curl_Virus', 'Corn___Cercospora_leaf_spot Gray_leaf_spot', 'Tomato___Early_blight',
          'Cherry___Powdery_mildew', 'Soybean___healthy', 'Tomato___healthy', 'Corn___Northern_Leaf_Blight',
          'Corn___Common_rust', 'Peach___Bacterial_spot', 'Tomato___Target_Spot', 'Background_without_leaves',
          'Grape___Leaf_blight_(Isariopsis_Leaf_Spot)', 'Potato___Late_blight', 'Tomato___Tomato_mosaic_virus',
          'Potato___healthy', 'Potato___Early_blight', 'Tomato___Leaf_Mold', 'Grape___Black_rot', 'Cherry___healthy',
          'Tomato___Septoria_leaf_spot', 'Peach___healthy', 'Apple___Apple_scab', 'Corn___healthy',
          'Tomato___Bacterial_spot', 'Orange___Haunglongbing_(Citrus_greening)', 'Apple___Cedar_apple_rust',
          'Squash___Powdery_mildew']

while True:
    ret, frame = cap.read()
    if not ret:
        print("❌ Failed to read frame.")
        break

    # Resize and normalize input
    input_img = cv2.resize(frame, (input_shape[2], input_shape[3]))
    input_tensor = input_img.astype(np.float32) / 255.0
    input_tensor = np.transpose(input_tensor, (2, 0, 1))  # (3, H, W)
    input_tensor = np.expand_dims(input_tensor, axis=0)   # (1, 3, H, W)

    # Measure inference time
    start = time.time()
    outputs = session.run(None, {input_name: input_tensor})
    end = time.time()
    inference_time = (end - start) * 1000  # in ms

    # Prediction
    prediction = np.argmax(outputs[0])
    predicted_label = labels[prediction] if prediction < len(labels) else str(prediction)

    # Display prediction
    display_text = f"{predicted_label} ({inference_time:.2f} ms)"
    cv2.putText(frame, display_text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX,
                1, (0, 255, 255), 2, cv2.LINE_AA)  # Yellow color

    cv2.imshow("Prediction", frame)

    if cv2.waitKey(1) == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
