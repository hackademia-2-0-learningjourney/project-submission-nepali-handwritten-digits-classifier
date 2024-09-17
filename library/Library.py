import numpy as np
import pickle
from tensorflow.keras.utils import to_categorical
class Dense():
    def __init__(self, ninputs, nnodes,activation=None ):

        self.weight = np.random.randn(ninputs, nnodes) * np.sqrt(2. / ninputs) #xaiver initialization
        self.bias = np.random.rand(nnodes) * 0.01
        self.sdw = np.zeros((ninputs, nnodes))
        self.sdb = np.zeros(nnodes)
        self.vdw = np.zeros((ninputs, nnodes))
        self.vdb = np.zeros(nnodes)
        self.t = 0
        self.activation=activation

    def forward(self, inputs):
        self.input = inputs
        self.output = np.dot(inputs, self.weight) + self.bias
        if self.activation:
          self.output=self.activation.forward(self.output)
        return self.output

    def backward(self, gradient):
        if self.activation:
          gradient=self.activation.backward(gradient)
        self.gradient_weight = np.dot(self.input.T, gradient)
        self.gradient_bias = np.sum(gradient, axis=0)
        self.gradient_input = np.dot(gradient, self.weight.T)


        return self.gradient_input

    def calculate(self, optimizer):
        if optimizer == 'adam':
            self.t += 1
            beta1, beta2 = 0.9, 0.999
            epsilon = 1e-8

            self.sdw = beta2 * self.sdw + (1 - beta2) * (self.gradient_weight ** 2)
            self.sdb = beta2 * self.sdb + (1 - beta2) * (self.gradient_bias ** 2)

            self.vdw = beta1 * self.vdw + (1 - beta1) * self.gradient_weight
            self.vdb = beta1 * self.vdb + (1 - beta1) * self.gradient_bias

            # Bias correction for adam optimizer for the starting difference while using exponantially weighted average
            sdw_corrected = self.sdw / (1 - beta2 ** self.t)
            sdb_corrected = self.sdb / (1 - beta2 ** self.t)
            vdw_corrected = self.vdw / (1 - beta1 ** self.t)
            vdb_corrected = self.vdb / (1 - beta1 ** self.t)

            self.sdw_corrected = sdw_corrected
            self.sdb_corrected = sdb_corrected
            self.vdw_corrected = vdw_corrected
            self.vdb_corrected = vdb_corrected

    def update(self, learning_rate, optimizer):
        if optimizer == 'adam':
            self.weight -= learning_rate * self.vdw_corrected / (np.sqrt(self.sdw_corrected) + 1e-8)
            self.bias -= learning_rate * self.vdb_corrected / (np.sqrt(self.sdb_corrected) + 1e-8)
        else:
            self.weight -= learning_rate * self.gradient_weight
            self.bias -= learning_rate * self.gradient_bias

    def l2(self):
        return np.sum(self.weight ** 2)

class Relu():
    def forward(self, inputs):
        self.input = inputs
        self.output = np.maximum(0, inputs)
        return self.output

    def backward(self, gradients):
        self.gradient = gradients * (self.input > 0) #why not self.output>>>because we need a boolean return
        return self.gradient

class Sigmoid():
    def forward(self, inputs):
        self.input = inputs
        self.output = 1 / (1 + np.exp(-inputs))
        return self.output

    def backward(self, dvalues):
        self.dinputs = dvalues * (1 - self.output) * self.output
        return self.dinputs

class Softmax():
    def __init__(self,final=False):
        self.final = final
    def forward(self, inputs):
        self.input = inputs
        exp = np.exp(inputs - np.max(inputs, axis=1, keepdims=True))
        probabilities = exp / np.sum(exp, axis=1, keepdims=True)
        self.output = probabilities
        return self.output

    def backward(self, gradient):
      if self.final == True:
        return gradient
      else:
        self.dinputs = gradient * self.output * (1 - self.output)  # Derivative of softmax
        return self.dinputs

class CategoricalCrossEntropyLoss():
    def forward(self, probs, true_outputs,lamda=0.01,layers=None):
        clipped_probs = np.clip(probs, 1e-7, 1 - 1e-7)
        loss_data = -np.sum(true_outputs * np.log(clipped_probs)) / (len(true_outputs) + 1e-8)
        if layers:
          l2_terms = [lamda * np.sum(layer.l2()) for layer in layers]
          loss_weight = 0.5 * np.sum(l2_terms) / (len(true_outputs) +  1e-8)
        else:
          loss_weight=0
        return loss_data+loss_weight

    def accuracy(self, probs, true_outputs):

        prediction=np.argmax(probs, axis=1)
        true_label=np.argmax(true_outputs, axis=1)
        accuracy=np.mean(prediction == true_label)
        return accuracy

    def backward(self, probs, true_outputs):
        samples = len(true_outputs)

        self.dinputs = (probs - true_outputs) / samples
        return self.dinputs

class BinaryCrossEntropyLoss():
    def forward(self, y_pred, y_true, layers):
        y_pred = np.clip(y_pred, 1e-7, 1 - 1e-7)
        loss_data = -np.mean(y_true * np.log(y_pred) + (1 - y_true) * np.log(1 - y_pred))
        return loss_data

    def backward(self, dvalues, y_true):
        dvalues = np.clip(dvalues, 1e-7, 1 - 1e-7)
        self.dinputs = (dvalues - y_true) / len(y_true)
        return self.dinputs




class NeuralNetwork():
  def __init__(self,loss_function='CategoricalCrossEntropyLoss()',optimizer='adams',learning_rate=0.001):
     self.layers=[]
     self.loss_function = loss_function
     self.learning_rate = learning_rate
     self.optimizer = optimizer


  def add(self,layer):
    self.layers.append(layer)



  def fit(self, X_train, y_train, batch_size,epochs=10):
      self.epochs=epochs
      for epoch in range(self.epochs):
          epoch_loss = 0
          accuracy=0
          epoch_loss_val = 0
          for i in range(0, len(X_train), batch_size):
              batch_inputs = X_train[i:i + batch_size]
              # batch_validate=X_test[i:i + batch_size]
              batch_true_outputs = y_train[i:i + batch_size]
              # batch_validate_outputs = y_test[i:i + batch_size]

              x = batch_inputs
              #print(f'x ko shape{x.shape}')
              for layer in self.layers:
                  x = layer.forward(x)
                  #print(x.shape)


              loss = self.loss_function.forward(x, batch_true_outputs)
              epoch_loss += loss  # Accumulate batch loss
              accuracy+=self.loss_function.accuracy(x, batch_true_outputs)
              gradient = self.loss_function.backward(x, batch_true_outputs)
              for layer in reversed(self.layers):
                  # print(f'gradient is {gradient.shape}')
                  gradient = layer.backward(gradient)
                  # print(f'gradient is {gradient.shape}')

              for layer in self.layers:
                  layer.calculate(self.optimizer)

              for layer in self.layers:
                  layer.update(self.learning_rate, self.optimizer)


          print(f"Epoch {epoch + 1}, Loss: {epoch_loss / len(X_train) * batch_size}, accuracy={accuracy/ len(X_train) * batch_size}")  # Print average loss for the epoch
          epoch_accuracy = 0
          epoch_loss_val = 0
          # for i in range(0,len(X_test),batch_size):
          #     batch_validate = X_test[i:i + batch_size]
          #     batch_validate_true = y_test[i:i + batch_size]
  def eval(self,X_test,y_test):
    x2=X_test
    for layer in self.layers:
        x2=layer.forward(x2)
        print(x2.shape)
    loss_validate = self.loss_function.forward(x2, y_test, self.layers)
    accurate=self.loss_function.accuracy(x2, y_test)
    # epoch_loss_val += loss_validate
    # epoch_accuracy+=accurate
    print(f"val_Loss: {loss_validate},val_accuracy:{accurate}")

  def predict(self,X):
    x=X
    for layer in self.layers:
      x=layer.forward(x)
    print(np.argmax(x,axis=1))



  def save_model(self, filename):
     
      with open(filename, 'wb') as file:
          pickle.dump(self, file)
      print(f"Model saved to {filename}")

  @staticmethod
  def load_model(filename):
        
        with open(filename, 'rb') as file:
            model = pickle.load(file)
        print(f"Model loaded from {filename}")
        return model




