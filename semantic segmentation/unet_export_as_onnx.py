# *************************** TENSORFLOW - TESTED *********************************
# When using Tenforflow, after training, save your model using the follow
# Make a note of Input image size

tf.saved_model.save(model, "./output")

# After saving the model, install the following :-
# pip install git+https://github.com/onnx/tensorflow-onnx
# Assuming tensorflow, onnx and onnxruntime is installed
python -m tf2onnx.convert --saved-model /content/output --output unet.onnx

# Refer tf2onnx/convert.py for checking out more supported formats

# *** Log should look like following (Environment - Google Colab)
'''
/usr/lib/python3.7/runpy.py:125: RuntimeWarning: 'tf2onnx.convert' found in sys.modules after import of package 'tf2onnx', but prior to execution of 'tf2onnx.convert'; this may result in unpredictable behaviour
  warn(RuntimeWarning(msg))
2022-02-19 11:36:36,700 - WARNING - '--tag' not specified for saved_model. Using --tag serve
2022-02-19 11:36:39,017 - INFO - Signatures found in model: [serving_default].
2022-02-19 11:36:39,017 - WARNING - '--signature_def' not specified, using first signature: serving_default
2022-02-19 11:36:39,017 - INFO - Output names: ['conv2d_75']
WARNING:tensorflow:From /usr/local/lib/python3.7/dist-packages/tf2onnx/tf_loader.py:706: extract_sub_graph (from tensorflow.python.framework.graph_util_impl) is deprecated and will be removed in a future version.
Instructions for updating:
Use `tf.compat.v1.graph_util.extract_sub_graph`
2022-02-19 11:36:39,482 - WARNING - From /usr/local/lib/python3.7/dist-packages/tf2onnx/tf_loader.py:706: extract_sub_graph (from tensorflow.python.framework.graph_util_impl) is deprecated and will be removed in a future version.
Instructions for updating:
Use `tf.compat.v1.graph_util.extract_sub_graph`
2022-02-19 11:36:39,645 - INFO - Using tensorflow=2.8.0, onnx=1.11.0, tf2onnx=1.10.0/22db2a
2022-02-19 11:36:39,645 - INFO - Using opset <onnx, 9>
2022-02-19 11:36:40,041 - INFO - Computed 0 values for constant folding
2022-02-19 11:36:40,397 - INFO - Optimizing ONNX model
2022-02-19 11:36:40,677 - INFO - After optimization: BatchNormalization -9 (9->0), Cast -4 (4->0), Concat -4 (8->4), Const -63 (92->29), Identity -13 (13->0), Reshape +1 (0->1), Shape -4 (4->0), Slice -4 (4->0), Squeeze -4 (4->0), Transpose -53 (54->1), Unsqueeze -16 (16->0)
2022-02-19 11:36:40,687 - INFO - 
2022-02-19 11:36:40,687 - INFO - Successfully converted TensorFlow model /content/output to ONNX
2022-02-19 11:36:40,687 - INFO - Model inputs: ['img']
2022-02-19 11:36:40,687 - INFO - Model outputs: ['conv2d_75']
2022-02-19 11:36:40,687 - INFO - ONNX model is saved at unet.onnx
'''
