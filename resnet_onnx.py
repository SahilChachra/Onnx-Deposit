import onnx
import onnxruntime as ort
import numpy as np
import cv2
from imagenet_classlist import get_class

model_path = 'resnet18.onnx'
model = onnx.load(model_path)
image_path = "tiger.jpg"

try:
    print("Checking model...")
    onnx.checker.check_model(model)
    onnx.helper.printable_graph(model.graph)
    print("Model checked...")
    
    print("Running inference...")
    
    ort_session = ort.InferenceSession(model_path)

    image = cv2.imread(image_path, cv2.IMREAD_COLOR)
    image = cv2.resize(image, ((224, 224)))
    image = np.moveaxis(image, -1, 0)
    
    outputs = ort_session.run(None, {"input":np.array([image/255.0]).astype(np.float32)})
    
    print("Index :",np.argmax(outputs[0]))
    print("Class is :", get_class(np.argmax(outputs[0])))

except Exception as e:
    print("Exception occured : ", e)