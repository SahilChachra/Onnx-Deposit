'''
This inference script takes in images of size 256x256 and batch size 1
The images for UNET are not uploaded as its under NDA.

Plugin your own converted model and images :)
'''

import onnx
import onnxruntime as ort
import numpy as np
import cv2

model_path = 'unet_binary.onnx'
model = onnx.load(model_path)
image_path = 'unet_images/0.png'

try:
    print("Checking model...")
    onnx.checker.check_model(model)
    onnx.helper.printable_graph(model.graph)
    print("Model checked...")
    
    print("Running inference...")
    
    ort_session = ort.InferenceSession(model_path)

    image = cv2.imread(image_path, cv2.IMREAD_COLOR)
    image = cv2.resize(image, ((256, 256)))
    print("image shape :", image.shape)
    
    outputs = ort_session.run(None, {"img":np.array([image]).astype(np.float32)})  # img here is the name of input layer.
                                                                                   # Use netron to check the name of layers
    predict = outputs[0][0]

    for i in range(predict.shape[0]):
        for j in range(predict.shape[1]):
            if(predict[i][j]>=0.5):
                predict[i][j]=1
            else:
                predict[i][j]=0
    
    predict=np.uint8(predict)
    predict=predict*255

    cv2.imshow('output_mask',predict)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

except Exception as e:
    print("Exception occured : ", e)