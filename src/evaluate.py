from PIL import Image
from glob import glob
import numpy as np
import onnxruntime
import os

# Load ONNX model
session = onnxruntime.InferenceSession("student_model_pruned_quantized.onnx", providers=["CPUExecutionProvider"])
input_name = session.get_inputs()[0].name
input_shape = session.get_inputs()[0].shape

image_paths = glob("data/test/*.jpg")
correct = 0
total = len(image_paths)

for path in image_paths:
    img = Image.open(path).convert("RGB")
    img = img.resize((input_shape[2], input_shape[3]))
    img = np.array(img).astype(np.float32) / 255.0
    img = np.transpose(img, (2, 0, 1))
    img = np.expand_dims(img, axis=0)

    outputs = session.run(None, {input_name: img})
    prediction = np.argmax(outputs[0])

    print(f"{os.path.basename(path)} → Predicted: {prediction}")

    # If you have labels, compare here:
    # if prediction == true_label: correct += 1

# print(f"UAV Accuracy: {correct/total:.2f}")
