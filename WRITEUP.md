# Project Write-Up

You can use this document as a template for providing your project write-up. However, if you
have a different format you prefer, feel free to use it as long as you answer all required
questions.

First of all, I have uploaded the following model: 
https://arxiv.org/abs/1704.04861
To do so, I executed the following commands in the terminal: 
- I've downloaded the model:
wget http://download.tensorflow.org/models/object_detection/ssd_mobilenet_v2_coco_2018_03_29.tar.gz

- I've extracted the model:
tar -xvf ssd_mobilenet_v2_coco_2018_03_29.tar.gz 

- I've navigated to the folder and converted the model:
cd ssd_mobilenet_v2_coco_2018_03_29
python /opt/intel/openvino/deployment_tools/model_optimizer/mo.py --input_model frozen_inference_graph.pb --tensorflow_object_detection_api_pipeline_config pipeline.config --reverse_input_channels --tensorflow_use_custom_operations_config /opt/intel/openvino/deployment_tools/model_optimizer/extensions/front/tf/ssd_v2_support.json

After changing the inference.py, I've used this command to check whether it compiles:
python inference.py /home/workspace/ssd_mobilenet_v2_coco_2018_03_29/frozen_inference_graph.xml 

IMPORTANT: execution has to be performed in the first Terminal, after ENV button is clicked. Otherwise it will show an error "No module named openvino"

- To start main.py program, I used this command:
python main.py -m /home/workspace/ssd_mobilenet_v2_coco_2018_03_29/frozen_inference_graph.xml -i resources/Pedestrian_Detect_2_1_1.mp4 -l /opt/intel/openvino/deployment_tools/inference_engine/lib/intel64/libcpu_extension_sse4.so -d CPU -pt 0.6 | ffmpeg -v warning -f rawvideo -pixel_format bgr24 -video_size 768x432 -framerate 24 -i - http://0.0.0.0:3004/fac.ffm


## Explaining Custom Layers

As per OpenVino documentation: "Custom layers are layers that are not included into a list of known layers. If your topology contains any layers that are not in the list of known layers, the Model Optimizer classifies them as custom."
(see https://docs.openvinotoolkit.org/latest/_docs_HOWTO_Custom_Layers_Guide.html)

This means that when we take to convert some existing TensorFlow, Caffe, MXNet, Kaldi or ONYX model, some of the layers will be classsified as custom.

These layers are not supported, so when converted, they won't be there. This drives to the reduction of accuracy. Though accuracy is reduced, the memory space needed is reduced, and speed remains high.

In this work I've converted the TensorFlow model. As mentioned in the lesson, to do so, the following steps must be done:
- Configure the Model Optimizer for TensorFlow (classroom had this already done)
- Freeze the TF model if your model is not already frozen
- Or use the separate instructions to convert a non-frozen model
- Convert the TF model with the Model Optimizer to an optimized IR
- Test the model in the IR format using the Inference Engine. 

## Comparing Model Performance

I've selected the SSD Mobilenet v2  Model, since I read an article about its performance (https://arxiv.org/pdf/1704.04861.pdf), and in comparison to some other models its accuracy was higher. This seemed to work out for the given task. 

## Assess Model Use Cases

Some of the potential use cases of the people counter app are:
- A queue in the shop - understand how many people per day come to the shop. This could be a good KPI for a business. 
- Office check - checking how many people come to the office, at which time. 
- Transport - seeing the peak times, and adapting transport schedule accordingly. 
- Monitoring security system - see if someone is trying to enter the house. 

Each of these use cases would be useful because:
- It can help out businesses
- Enhance planning
- Guarantee security

## Assess Effects on End User Needs

It is important to find an optimal position for the camera. If it's too dark or too light, the people might not be distinguished from the environment. Therefore, it is important to place the camera so, that there would be no direct lights at it.
If camera is too close or too far, the size of perceived image will also be hard to analyze. This will influence the accuracy. If camera is located in some place, which doesn't capture people when they are moving, this would be also a problem. 
The current model accuracy sometimes loses person from the view, and then counts it multiple times, but this lies on the model accuracy only. Quality of this video seems to be sufficient, but in some real life applications a low camera quality can play a big role. 