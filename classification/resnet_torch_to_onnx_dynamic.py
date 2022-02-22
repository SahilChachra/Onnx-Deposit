'''
Convert ResNet torch model to Onnx with dynamic batch size and width, height
'''

import torch
import torchvision

dummy_input = torch.randn(1, 3, 224, 224, requires_grad=True).cuda()
model = torchvision.models.resnet18(pretrained=True).cuda()
model.eval()
# print(model)
torch.onnx.export(model, dummy_input, "resnet18.onnx", input_names=['input'], output_names=['output'],
    dynamic_axes= {'input':{0:'batch_size' , 2:'width', 3:'height'}, 
        'output':{0:'batch_size' , 2:'width', 3:'height'}})