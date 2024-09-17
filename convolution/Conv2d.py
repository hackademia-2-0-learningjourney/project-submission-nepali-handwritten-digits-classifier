import numpy as np
from numpy.lib.stride_tricks import as_strided
class Convolution:
    def __init__(self, filter_size, padding=1, stride=1):
        self.filter_height, self.filter_width = filter_size
        self.padding = padding
        self.stride = stride

    def get_patches(self, image_array, pool=False):
        self.image_array = image_array
        self.batch_size, self.image_height, self.image_width, self.channel = self.image_array.shape
        if pool:
            self.stride = 2
            self.filter_height = 2
            self.filter_width = 2
        else:
            self.stride = 1
            self.filter_height = 3
            self.filter_width = 3

        self.new_height = (self.image_height - self.filter_height) // self.stride + 1
        self.new_width = (self.image_width - self.filter_width) // self.stride + 1
        self.new_shape = (self.batch_size, self.new_height, self.new_width, self.filter_height, self.filter_width, self.channel)
        self.mem_location = (
            self.image_array.strides[0],
            self.image_array.strides[1] * self.stride,
            self.image_array.strides[2] * self.stride,
            self.image_array.strides[1],
            self.image_array.strides[2],
            self.image_array.strides[3]
        )
        self.patches = as_strided(self.image_array, self.new_shape, self.mem_location)
        return self.patches

    def forward(self, image_array):

        self.image_array_padded = np.pad(
            image_array,
            ((0, 0), (self.padding, self.padding), (self.padding, self.padding), (0, 0)),
            mode='constant'
        )


        self.patches = self.get_patches(self.image_array_padded)
        self.patches = self.patches.reshape(
            self.patches.shape[0],
            self.patches.shape[1],
            self.patches.shape[2],
            self.filter_height * self.filter_width * self.channel
        )


        self.filter = np.array([
            [ 0.2663854 , -0.15837954, -0.5408459 ,  0.32252902, -0.8184993 ,
              0.39549002,  0.24141438,  0.3327212 ],
            [ 0.42389   ,  0.46823323, -1.5491629 , -0.21998599, -1.233032  ,
             -0.6546184 , -0.24297695,  0.41794822],
            [-0.16864721,  0.6169712 , -0.27409363,  0.21056598, -0.6376102 ,
              0.11626843, -0.32959324, -0.85535324],
            [-0.04536731,  0.44682398,  0.856612  ,  0.3269437 , -0.5095954 ,
              0.58412886,  0.6624227 ,  0.48114   ],
            [ 0.06798663,  0.57804066,  0.2823445 , -0.11366118,  0.4670202 ,
             -0.6701556 ,  0.23715912,  0.6309564 ],
            [ 0.07941878, -0.10522328, -1.4801729 ,  0.5282848 ,  0.41923472,
             -0.12389828, -0.65917736, -0.9575157 ],
            [-0.19738226, -0.702964  ,  0.6215948 ,  0.2402507 ,  0.43504846,
              0.44762683, -0.28033203,  0.3938811 ],
            [ 0.41120657, -1.0216138 ,  0.70563865,  0.01174228,  0.8256505 ,
             -0.15622851,  0.0332014 , -0.05408265],
            [ 0.22918034, -0.69328874, -0.37063798, -0.01847612,  0.22378902,
             -0.12718084,  0.7328887 , -1.2189778 ]
        ])


        self.edge = np.tensordot(self.patches, self.filter, axes=([3], [0]))


        self.patches2 = self.get_patches(self.edge, pool=True)
        self.output = np.max(self.patches2, axis=(3, 4))
        self.output = np.maximum(0, self.output)
        print(self.output.shape)
        
        return self.output.reshape(self.output.shape[0], -1)
