# Road Network Extraction from Satellite Images using Geometric Deep Learning
(Work in Progress)

## Objectives :
The [MIT RoadTracer](https://roadmaps.csail.mit.edu/roadtracer.pdf) paper from 2018 used an iterative search process guided by a CNN-based decision function to derive the
road network graph directly from the output of the CNN. Their method requires human to identify two initial points which are present on the road. After that, roadtracer model takes a 256 by 256 image patch and outputs the next one node in the road network. 

Here, using the dataset from RoadTracer, a geometric deep learning approach for solving road network extraction is being carried out. The goal is to develop a model which takes a 256 by 256 image - same as RoadTracer - and outputs the entire road network graph present in that image patch. In order to achieve this, I am looking into spatial graph convolution layer ([GraphSAGE](https://arxiv.org/pdf/1706.02216.pdf)) and dense graph pooling layer ([DIFFPOOL](https://arxiv.org/pdf/1806.08804.pdf)) to convert feature maps of images into graph networks. The network comprises of 4-5 convolution + pooling layers followed by a combination of GraphSAGE and DIFFPOOL layer to convert the extracted feature maps into graphs. 

## Libraries : 
PyTorch, Pytorch Geometric, Tensorflow, Networkx, OpenCV, scikit-learn
