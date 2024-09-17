import matplotlib.pyplot as plt
import cv2
import numpy as np
import Conv2d
import canvas
import Library
from Library import Dense,Relu,CategoricalCrossEntropyLoss,NeuralNetwork
nn=NeuralNetwork.load_model('model.pkl')

array = cv2.imread('canvas_image.png',cv2.IMREAD_GRAYSCALE)
conv=Conv2d.Convolution((3,3))
array=np.expand_dims(array,axis=0)
array=np.expand_dims(array,axis=3)
array=conv.forward(array)
nn.predict(array)