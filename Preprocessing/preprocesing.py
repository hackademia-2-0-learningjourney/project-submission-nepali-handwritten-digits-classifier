import Conv2d
import numpy as np
conv = Conv2d((3,3))
train = np.load('arr_0.npy')
X_train=conv.forward(train)
np.save('train.npy')

test = np.load('arr_2.npy')
X_test=conv.forward(test)
np.save(test.npy)