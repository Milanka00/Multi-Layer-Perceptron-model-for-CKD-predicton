# Multi-Layer-Perceptron-model-for-CKD-predicton

Developed a MLP which is a configurable neural network designed for binary classification tasks. It features a flexible architecture with a user-defined number of hidden layers and neurons. Weights are initialized randomly, and biases are set to zero. The network uses the sigmoid activation function for non-linearity and uses forward propagation to compute outputs layer by layer. Training is performed using backpropagation with gradient descent, updating weights and biases based on mean squared error (MSE). The model supports batch gradient accumulation and allows tuning of hyperparameters like learning rate, epochs, and layer sizes. Predictions are generated through a forward pass, making it suitable for applications requiring binary classification.

```bash
class NeuralNetwork:
    def __init__(self, n_inputs, hidden_layer_sizes=[3]):

        #Initializes the feedforward neural network.

        self.input_size = n_inputs
        self.output_size = 1  # binary classification
        self.num_hidden_layers = len(hidden_layer_sizes)
        self.layer_sizes = [self.input_size] + hidden_layer_sizes + [self.output_size]

        # Initialize weights (W) and biases (B) for all layers
        self.weights = {}
        self.biases = {}
        for i in range(self.num_hidden_layers + 1):
            self.weights[i + 1] = np.random.randn(self.layer_sizes[i], self.layer_sizes[i + 1])
            self.biases[i + 1] = np.zeros((1, self.layer_sizes[i + 1]))

    def sigmoid(self, z):

       # Activation function: Sigmoid.

        return 1.0 / (1.0 + np.exp(-z))

    def forward_pass(self, inputs):

        self.activations = {}
        self.layer_outputs = {}
        self.layer_outputs[0] = inputs.reshape(1, -1)

        for i in range(self.num_hidden_layers + 1):
            # Compute linear combination for current layer
            self.activations[i + 1] = np.matmul(self.layer_outputs[i], self.weights[i + 1]) + self.biases[i + 1]
            # Apply sigmoid activation
            self.layer_outputs[i + 1] = self.sigmoid(self.activations[i + 1])

        return self.layer_outputs[self.num_hidden_layers + 1]

    def sigmoid_gradient(self, output):

        return output * (1 - output)

    def compute_gradients(self, inputs, true_output):

        self.forward_pass(inputs)

        self.grad_weights = {}
        self.grad_biases = {}
        self.grad_activations = {}
        self.grad_layer_outputs = {}

        # Compute gradient for the last layer (output layer)
        last_layer = self.num_hidden_layers + 1
        self.grad_activations[last_layer] = self.layer_outputs[last_layer] - true_output

        # Backpropagate through layers
        for layer in range(last_layer, 0, -1):
            # Gradients for weights and biases
            self.grad_weights[layer] = np.matmul(self.layer_outputs[layer - 1].T, self.grad_activations[layer])
            self.grad_biases[layer] = self.grad_activations[layer]
            # Propagate error to the previous layer
            if layer > 1:
                self.grad_layer_outputs[layer - 1] = np.matmul(self.grad_activations[layer], self.weights[layer].T)
                self.grad_activations[layer - 1] = np.multiply(self.grad_layer_outputs[layer - 1],
                                                               self.sigmoid_gradient(self.layer_outputs[layer - 1]))

    def fit(self, X, Y, epochs=1, learning_rate=1, initialize=True, display_loss=False):

        # Reinitialize weights and biases if required
        if initialize:
            for i in range(self.num_hidden_layers + 1):
                self.weights[i + 1] = np.random.randn(self.layer_sizes[i], self.layer_sizes[i + 1])
                self.biases[i + 1] = np.zeros((1, self.layer_sizes[i + 1]))

        if display_loss:
            loss = {}

        for epoch in tqdm_notebook(range(epochs), total=epochs, unit="epoch"):
            # Initialize cumulative gradients
            cumulative_grad_weights = {i + 1: np.zeros_like(self.weights[i + 1]) for i in range(self.num_hidden_layers + 1)}
            cumulative_grad_biases = {i + 1: np.zeros_like(self.biases[i + 1]) for i in range(self.num_hidden_layers + 1)}

            for x, y in zip(X, Y):
                self.compute_gradients(x, y)

                for i in range(self.num_hidden_layers + 1):
                    cumulative_grad_weights[i + 1] += self.grad_weights[i + 1]
                    cumulative_grad_biases[i + 1] += self.grad_biases[i + 1]

            # Update weights and biases using gradient descent
            m = X.shape[0]  # Number of training samples
            for i in range(self.num_hidden_layers + 1):
                self.weights[i + 1] -= learning_rate * cumulative_grad_weights[i + 1] / m
                self.biases[i + 1] -= learning_rate * cumulative_grad_biases[i + 1] / m

            # Compute loss
            if display_loss:
                Y_pred = self.predict(X)
                loss[epoch] = mean_squared_error(Y_pred, Y)

        if display_loss:
            plt.plot(list(loss.values()))
            plt.xlabel('Epochs')
            plt.ylabel('Mean Squared Error')
            plt.show()

    def predict(self, X):

        predictions = []
        for x in X:
            predictions.append(self.forward_pass(x))
        return np.array(predictions).squeeze()
```
Function for evaluating the accuracy of training and testing

```bash
def evaluate_model(model, X_train, Y_train, X_test, Y_test, threshold=0.5):

    # Predict on training data
    Y_pred_train = model.predict(X_train)
    Y_pred_binarised_train = (Y_pred_train >= threshold).astype("int").ravel()

    # Predict on testing data
    Y_pred_test = model.predict(X_test)
    Y_pred_binarised_test = (Y_pred_test >= threshold).astype("int").ravel()

    # Calculate accuracies
    accuracy_train = accuracy_score(Y_train, Y_pred_binarised_train)
    accuracy_test = accuracy_score(Y_test, Y_pred_binarised_test)

    # Return results
    return {
        "training_accuracy": accuracy_train,
        "testing_accuracy": accuracy_test,
        "train_predictions": Y_pred_binarised_train,
        "testing_predictions": Y_pred_binarised_test,
    }
```
Example: 
```bash
# Instantiate the model with 24 inputs and 1 hidden layer with 14 neurons
ffsnn = NeuralNetwork(24, [14])

# Train the model
ffsnn.fit(X_train, Y_train, epochs=2500, learning_rate=0.01, display_loss=True)

results = evaluate_model(ffsnn, X_train, Y_train, X_test, Y_test)
print(f"Training Accuracy: {results['training_accuracy']}")
print(f"Testing Accuracy: {results['testing_accuracy']}")
```
Results:

![results](./results.png)
