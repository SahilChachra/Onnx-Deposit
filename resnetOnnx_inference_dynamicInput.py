'''
This inference script takes in images of dynamic size
Runs inference in batch

** In this images have been resized but not need for this script
'''

import onnx
import onnxruntime as ort
import numpy as np
import cv2
from imagenet_classlist import get_class
import os

model_path = 'resnet18.onnx'
model = onnx.load(model_path)
image_path = "./images"

try:
    print("Checking model...")
    onnx.checker.check_model(model)
    onnx.helper.printable_graph(model.graph)
    print("Model checked...")
    
    print("Running inference...")
    
    ort_session = ort.InferenceSession(model_path)

    img_list = []
    for image in os.listdir(image_path):
        img = cv2.imread(os.path.join(image_path, image), cv2.IMREAD_COLOR)
        img = cv2.resize(img, ((224, 224)))
        img = np.moveaxis(img, -1, 0) # (Batch_size, channels, width, heigth)
        img_list.append(img/255.0) # Normalize the image

    outputs = ort_session.run(None, {"input":img_list})
    out = np.array(outputs)

    for image_num, image_name in zip(range(out.shape[1]), os.listdir(image_path)):
        index = out[0][image_num]
        print("Image : {0}, Class : {1}".format(image_name, get_class(np.argmax(index))))

except Exception as e:
    print("Exception occured : ", e)