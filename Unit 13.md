# 1. Explain the architecture of Artificial Neural Networks.

### 1. Architecture of Artificial Neural Networks (ANN)

Artificial Neural Networks (ANN) are inspired by the biological neural networks of the human brain. The architecture of an ANN is composed of a series of layers, each consisting of interconnected nodes or neurons. Here's a breakdown:

- **Input Layer:** The first layer that receives the input data. Each neuron in this layer represents a feature of the input data.
- **Hidden Layers:** One or more layers that perform computations through neurons. These layers transform the input into something the output layer can use.
- **Output Layer:** The final layer that provides the output of the network. The number of neurons here typically corresponds to the number of output classes or the desired output format.
- **Neurons:** The basic units of an ANN, they receive inputs and pass on their signal to the next layer after applying an activation function.
- **Weights and Biases:** Each connection between neurons has an associated weight and each neuron has a bias. These are adjusted during training to minimize the error in predictions.
- **Activation Functions:** Functions applied at each neuron to introduce non-linearities, allowing the network to learn complex patterns.

# 2. List the various tools used to implement ANN.

### 2. Tools for Implementing ANN

Various software tools and libraries are available for implementing ANNs, including:

- **TensorFlow:** An open-source library developed by Google, popular for deep learning applications.
- **Keras:** A high-level neural networks API, capable of running on top of TensorFlow, CNTK, or Theano.
- **PyTorch:** An open-source machine learning library developed by Facebook, known for its flexibility and dynamic computational graph.
- **Scikit-Learn:** A Python library, though more limited for deep learning, it's useful for basic ANN models.
- **MATLAB Neural Network Toolbox:** Offers algorithms, pretrained models, and apps for designing and simulating neural networks.

# 3. What are all the activation functions used for training ANN?

### 3. Activation Functions in ANN

Several activation functions are used in ANNs, including:

- **Sigmoid or Logistic:** Converts input into a value between 0 and 1, useful for binary classification.
- **ReLU (Rectified Linear Unit):** Allows only positive values to pass through, and it's computationally efficient.
- **Tanh (Hyperbolic Tangent):** Similar to sigmoid but outputs values between -1 and 1.
- **Softmax:** Used in the output layer of multi-class classification problems, turning logits into probabilities.
- **Leaky ReLU:** A variant of ReLU that allows small negative values when the input is less than zero, helping to fix the dying ReLU problem.

# 4. Givean example how the weights are adjusted.

### 4. Example of Weight Adjustment

Weight adjustment in ANNs typically occurs through a process known as backpropagation combined with an optimization algorithm like Gradient Descent. Here's a simplified example:

- **Forward Pass:** Input data is passed through the network, and an initial output is produced.
- **Calculate Error:** The difference between the predicted output and the actual output (error) is calculated.
- **Backward Pass:** The error is propagated back through the network, and the gradient of the error with respect to each weight is calculated.
- **Update Weights:** The weights are adjusted in the opposite direction of the gradient, typically using a learning rate to control the step size.

# 5. Differentiate biological neuron and artificial neuron.

### 5. Biological Neuron vs. Artificial Neuron

- **Biological Neuron:**
  - Found in the nervous system, it's a cell that processes and transmits information through electrical and chemical signals.
  - Has dendrites for receiving signals, a cell body (soma) that processes these signals, and an axon for sending out signals.
  - The transmission of information is based on complex biological processes.

- **Artificial Neuron:**
  - A mathematical function designed to model the operation of biological neurons.
  - Receives input signals (data points), processes them using weights and activation functions, and produces an output.
  - Works based on algorithms and computational processes rather than biological mechanisms.

In summary, while artificial neurons are inspired by their biological counterparts, they function based on mathematical and computational principles and are part of a broader and more simplified network structure compared to the complex and diverse functionality of biological neural networks.