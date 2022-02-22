'''
This py file runs inference on YoloV5 Onnx model.
To convert your model to Onnx, refer Ultralytics/YoloV5 repo
Checkout the arguements you can pass
Helping functions are borrowed from YoloV5 official repository by Ultralytics
'''

import onnx
import onnxruntime as ort
import numpy as np
import cv2
import argparse
import torch
import torchvision
import time
import math
import os

import warnings
warnings.filterwarnings('ignore')

def xywh2xyxy(x):
    # Convert nx4 boxes from [x, y, w, h] to [x1, y1, x2, y2] where xy1=top-left, xy2=bottom-right
    y = x.clone() if isinstance(x, torch.Tensor) else np.copy(x)
    y[:, 0] = x[:, 0] - x[:, 2] / 2  # top left x
    y[:, 1] = x[:, 1] - x[:, 3] / 2  # top left y
    y[:, 2] = x[:, 0] + x[:, 2] / 2  # bottom right x
    y[:, 3] = x[:, 1] + x[:, 3] / 2  # bottom right y
    return y

def non_max_suppression(prediction, conf_thres=0.25, iou_thres=0.45, agnostic=False, multi_label=False,
                        max_det=300):
    """Runs Non-Maximum Suppression (NMS) on inference results

    Returns:
         list of detections, on (n,6) tensor per image [xyxy, conf, cls]
    """
    prediction = torch.tensor(prediction)
    nc = prediction.shape[2] - 5  # number of classes
    xc = prediction[..., 4] > conf_thres  # create a mask tensor -> set true for object with conf > thresh else false
    
    # Checks
    assert 0 <= conf_thres <= 1, f'Invalid Confidence threshold {conf_thres}, valid values are between 0.0 and 1.0'
    assert 0 <= iou_thres <= 1, f'Invalid IoU {iou_thres}, valid values are between 0.0 and 1.0'

    # Settings
    min_wh, max_wh = 2, 7680  # (pixels) minimum and maximum box width and height
    max_nms = 30000  # maximum number of boxes into torchvision.ops.nms()
    time_limit = 10.0  # seconds to quit after
    redundant = True  # require redundant detections
    multi_label &= nc > 1  # multiple labels per box (adds 0.5ms/img)
    merge = False  # use merge-NMS

    t = time.time()
    output = [torch.zeros((0, 6))] * prediction.shape[0]
    for xi, x in enumerate(prediction):  # image index, image inference
        # Apply constraints
        x[((x[..., 2:4] < min_wh) | (x[..., 2:4] > max_wh)).any(1), 4] = 0  # width-height
        x = x[xc[xi]]  # confidence 
        
        # If none remain process next image
        if not x.shape[0]:
            continue

        # Compute conf
        x[:, 5:] *= x[:, 4:5]  # conf = obj_conf * cls_conf
        
        # Box (center x, center y, width, height) to (x1, y1, x2, y2)
        box = xywh2xyxy(x[:, :4])

        # Detections matrix nx6 (xyxy, conf, cls)
        if multi_label:
            i, j = (x[:, 5:] > conf_thres).nonzero(as_tuple=False).T
            x = torch.cat((box[i], x[i, j + 5, None], j[:, None].float()), 1)
        else:  # best class only
            conf, j = torch.tensor(x[:, 5:]).max(1, keepdim=True)
            
            x = torch.cat((box, conf, j.float()), 1)[conf.view(-1) > conf_thres]

        # Check shape
        n = x.shape[0]  # number of boxes
        if not n:  # no boxes
            continue
        elif n > max_nms:  # excess boxes
            x = x[x[:, 4].argsort(descending=True)[:max_nms]]  # sort by confidence

        # Batched NMS
        c = x[:, 5:6] * (0 if agnostic else max_wh)  # classes
        boxes, scores = x[:, :4] + c, x[:, 4]  # boxes (offset by class), scores
        i = torchvision.ops.nms(boxes, scores, iou_thres)  # NMS
        if i.shape[0] > max_det:  # limit detections
            i = i[:max_det]

        output[xi] = x[i]
        if (time.time() - t) > time_limit:
            LOGGER.warning(f'WARNING: NMS time limit {time_limit}s exceeded')
            break  # time limit exceeded

    return output

def main(model_path, image_path, conf_thresh, iou_thresh, img_width, img_height, labels_path):
    
    model = onnx.load(model_path)
    print("Checking model...")
    onnx.checker.check_model(model)
    onnx.helper.printable_graph(model.graph)
    print("Model checked...")

    try:
        f = open(labels_path, 'r')
        class_list = f.readlines()
    except Exception as e:
        print("Exception while reading labels :", e)
    
    image_list = []
    img_list = []
    for image_name in os.listdir(image_path):
        image = cv2.imread(os.path.join(image_path, image_name), cv2.IMREAD_COLOR)
        image = cv2.resize(image, (img_width, img_height), cv2.INTER_AREA)
        img = image.copy()
        image = np.moveaxis(image, -1, 0)
        image = image/255.0
        image = np.array(image).astype(np.float32)

        image_list.append(image)
        img_list.append(img)

    print("Running inference...")
    
    cuda=False # TO ADD CUDA SUPPORT LATER
    providers = ['CUDAExecutionProvider', 'CPUExecutionProvider'] if cuda else ['CPUExecutionProvider']
    session = ort.InferenceSession(model_path, providers=providers)

    for i in range(len(image_list)):
        image = image_list[i]
        img = img_list[i]
        y = session.run([session.get_outputs()[0].name], {session.get_inputs()[0].name: [image]})[0]
        pred = non_max_suppression(y)
        # print(pred[0].numpy())
        outputs = pred[0].numpy()
        for out in outputs:
            # print((out[0], out[1]))
            x1, y1, x2, y2, class_id = int(out[0]), int(out[1]), int(out[2]), int(out[3]), int(out[5])
            cv2.rectangle(img, (x1, y1), (x2, y2), (255, 0, 0))
            cv2.putText(img, class_list[class_id].rstrip(), (x1-5, y1-5), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0))
        cv2.imshow('output_mask',img)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

def arguementParser():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_path', type=str, default='yolov5s.onnx',help="model path")
    parser.add_argument('--image_path', type=str, default='../sample_images', help='path to input images folder')
    parser.add_argument('--conf_thresh', type=float, default=0.25, help='class confidence threshold')
    parser.add_argument('--iou_thresh', type=float, default=0.45, help='bbox iou threshold')
    parser.add_argument('--img_width', type=int, default=640, help='img width')
    parser.add_argument('--img_height', type=int, default=640, help='img height')
    parser.add_argument('--labels_path', type=str, default='coco_labels.txt', help='text file having labels')
    
    opt = parser.parse_args()
    return opt

if __name__=='__main__':
    args = arguementParser()
    main(**vars(args))