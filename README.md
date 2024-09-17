# Nepali Handwritten Digits Classifier

## Group Name
ScratchDevlopers

## Group Members
- Pranav Subedi
- Anil Tiwari
- Rahul Rawal
- Saroj Maharjan

## Project Description

### Project Idea and Approach
This project aims to develop a custom classifier for Nepali handwritten digits using a convolutional neural network (CNN), built from the ground up using NumPy. The classifier converts user input images into inverted grayscale and performs classification, focusing on manual implementation of core components like convolution, pooling, and backpropagation.

### Approach Story
We approached this project with a clear focus on building a custom neural network, leveraging low-level operations to fully understand the mechanics of CNNs. The core development was centered around building a library entirely from scratch, without relying on machine learning frameworks like PyTorch or TensorFlow.

**Pranav Subedi** took charge of designing and implementing the core library, which handled operations like convolutions, pooling, and backpropagation. He played a key role in translating complex mathematical operations into efficient NumPy-based implementations. Pranav also worked on creating an interactive canvas that converts user input images into inverted grayscale, allowing the system to preprocess the input in a form suitable for classification.

**Anil Tiwari** was responsible for data preprocessing. He worked on resizing, normalizing, and batching the Nepali handwritten digits dataset, ensuring the data was ready for training and testing. His preprocessing work allowed the model to handle input in a consistent and optimized manner.

**Saroj Maharjan** led the model development efforts, training the custom-built layers and fine-tuning the network to achieve optimal accuracy. Rahul's deep understanding of neural networks helped the team navigate the challenges of training a custom implementation.

**Rahul Rawal** handled model testing and evaluation, ensuring that the model’s performance on unseen data was robust. He conducted thorough testing to validate the classifier’s accuracy and identified areas for improvement.

Together, the team created a system that not only trains a classifier on handwritten digits but also allows users to interact with the model through a canvas-based interface.

### Tech Stack
- **Programming Language(s):** Python
- **Libraries and Packages:**
  - Custom neural network library implemented using NumPy
  - Filter values for convolutional layers initialized using a PyTorch pre-trained model
- **Modules/Functionalities:**
  - Data preprocessing: Image resizing, normalization, and batching
  - Custom convolutional layer implementation (using manually set filters)
  - Pooling and fully connected layers built from scratch
  - Backpropagation implemented using matrix multiplication
  - Model evaluation metrics: accuracy, loss tracking, and confusion matrix generation
  - Prediction pipeline for classifying new handwritten digit images
  - Canvas that converts input into inverted grayscale for further processing
- **Tools and Platforms:** Colab, GitHub


