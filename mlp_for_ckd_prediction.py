# -*- coding: utf-8 -*-
"""MLP for CKD prediction.ipynb

Automatically generated by Colab.

Original file is located at
    https://colab.research.google.com/drive/1KnUwh4SxuewyODB2ssH85gUqOQ5J104R

# 1. Installing necessary libariries and fetching the data set:
"""

pip install ucimlrepo

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import math
import warnings
warnings.filterwarnings(action='ignore')
from ucimlrepo import fetch_ucirepo
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, mean_squared_error, log_loss, confusion_matrix
from google.colab import files

# Fetch the dataset
chronic_kidney_disease = fetch_ucirepo(id=336)

# Load features and target into a single DataFrame
data = pd.DataFrame(chronic_kidney_disease.data.features, columns=chronic_kidney_disease.feature_names)
data['target'] = chronic_kidney_disease.data.targets

data.head(10)

"""# 2. Data preprocessing"""

data.info()

"""**2.1 Data Cleaning**"""

# exploring unique values in the data set
for i in data.columns:
    print('unique values in "{}":\n'.format(i),data[i].unique())

"""In categorical features there are some misstyped data other the null values"""

# fixing misstyped data
for i in range(data.shape[0]):
    if data.iloc[i,24]=='ckd\t':
        data.iloc[i,24]='ckd'
    if data.iloc[i,19] in [' yes','\tyes']:
        data.iloc[i,19]='yes'
    if data.iloc[i,19]=='\tno':
        data.iloc[i,19]='no'
    if data.iloc[i,15]=='\t?':
        data.iloc[i,15]=np.nan
    if data.iloc[i,16]=='\t?':
        data.iloc[i,16]=np.nan
    if data.iloc[i,17]=='\t?':
        data.iloc[i,17]=np.nan
    if data.iloc[i,24]=='ckd':
        data.iloc[i,24]='yes'
    if data.iloc[i,24]=='notckd':
        data.iloc[i,24]='no'

#giving meaningfull feature names for columns
feature_names=['Age (yrs)','Blood Pressure (mm/Hg)','Specific Gravity','Albumin','Sugar','Red Blood Cells',
               'Pus Cells','Pus Cell Clumps','Bacteria','Blood Glucose Random (mgs/dL)','Blood Urea (mgs/dL)',
               'Serum Creatinine (mgs/dL)','Sodium (mEq/L)','Potassium (mEq/L)','Hemoglobin (gms)','Packed Cell Volume',
               'White Blood Cells (cells/cmm)','Red Blood Cells (millions/cmm)','Hypertension','Diabetes Mellitus',
               'Coronary Artery Disease','Appetite','Pedal Edema','Anemia','Chronic Kidney Disease']
data.columns=feature_names

data.head()

#separate numeric and categorical columns
numeric_cols = data.select_dtypes(include=[np.number]).columns
categorical_cols = data.select_dtypes(exclude=[np.number]).columns

# Plot numeric columns
if len(numeric_cols) > 0:
    rows = (len(numeric_cols) + 2) // 3
    plt.figure(figsize=(15, 5 * rows))
    for i, col in enumerate(numeric_cols, 1):
        plt.subplot(rows, 3, i)
        sns.histplot(data[col], kde=True, bins=20, color='blue')
        plt.title(f'Distribution of {col}')
        plt.xlabel('')
        plt.ylabel('')
    plt.tight_layout()
    # Save the figure
    plt.savefig("numeric_columns_distribution.png", dpi=300, bbox_inches='tight')
    plt.show()

    # Download the file
    files.download("numeric_columns_distribution.png")

# Plot categorical columns
if len(categorical_cols) > 0:
    rows = (len(categorical_cols) + 2) // 3
    plt.figure(figsize=(15, 5 * rows))
    for i, col in enumerate(categorical_cols, 1):
        plt.subplot(rows, 3, i)
        sns.countplot(data[col], palette='Set2', order=data[col].value_counts().index)
        plt.title(f'Distribution of {col}')
        plt.xlabel('')
        plt.ylabel('')
        plt.xticks(rotation=45)
    plt.tight_layout()
    # Save the figure
    plt.savefig("categoric_columns_distribution.png", dpi=300, bbox_inches='tight')
    plt.show()

    # Download the file
    files.download("categoric_columns_distribution.png")

"""**2.2 Handling missing values**"""

print(data.isnull().sum())

# Impute numeric columns with the mean
imputer_numeric = SimpleImputer(strategy='mean')
data[numeric_cols] = imputer_numeric.fit_transform(data[numeric_cols])

# Impute categorical/boolean columns with the most frequent value
imputer_categorical = SimpleImputer(strategy='most_frequent')
data[categorical_cols] = imputer_categorical.fit_transform(data[categorical_cols])

print(data.isnull().sum())

"""**2.3 Handling outliers**"""

# box plot visulization
plt.figure(figsize=(12, 6))

num_plots = len(numeric_cols)
num_cols = 4
num_rows = (num_plots // num_cols) + (1 if num_plots % num_cols != 0 else 0)

for i, col in enumerate(numeric_cols, 1):
    plt.subplot(num_rows, num_cols, i)
    sns.boxplot(x=data[col])
    plt.title(f"Boxplot of {col}")

plt.tight_layout()
# Save the figure
plt.savefig("outliers.png", dpi=300, bbox_inches='tight')
plt.show()

# Download the file
# files.download("outliers.png")

# Detecting outliers using IQR (Interquartile Range)
Q1 = data[numeric_cols].quantile(0.25)
Q3 = data[numeric_cols].quantile(0.75)
IQR = Q3 - Q1

lower_bound = Q1 - 1.5 * IQR
upper_bound = Q3 + 1.5 * IQR

outliers_condition = (data[numeric_cols] < lower_bound) | (data[numeric_cols] > upper_bound)

# Count outliers per feature
outliers_count = outliers_condition.sum()
print("Outliers count per feature:")
print(outliers_count)

# Replace outliers with the lower or upper bound values
for col in numeric_cols:
    data[col] = np.where(data[col] < lower_bound[col], lower_bound[col], data[col])
    data[col] = np.where(data[col] > upper_bound[col], upper_bound[col], data[col])

# Visualize the distributions of numeric columns again
plt.figure(figsize=(12, 6))
for i, col in enumerate(numeric_cols, 1):
    plt.subplot(num_rows, num_cols, i)
    sns.boxplot(x=data[col])
    plt.title(f"Boxplot of {col}")

plt.tight_layout()
plt.show()

"""**2.4 Encoding categorical data**"""

data.info()

binary_cols = [
    'Red Blood Cells', 'Pus Cells', 'Pus Cell Clumps', 'Bacteria',
    'Hypertension', 'Diabetes Mellitus', 'Coronary Artery Disease',
    'Appetite', 'Pedal Edema', 'Anemia', 'Chronic Kidney Disease'
]

# Apply label encoding to binary columns in the original data
label_encoder = LabelEncoder()
for col in binary_cols:
    data[col] = label_encoder.fit_transform(data[col])

# Check the transformed DataFrame
print(data.head())

data.info()

"""**Separating the features and target**"""

target_column = 'Chronic Kidney Disease'

# Separate features (X) and target (y)
X = data.drop(columns=[target_column])  # Features (all columns except the target)
Y = data[target_column]

# reshaped for future use
X_array = X.to_numpy()
X = X_array.reshape(400, 24)

"""**Scaling the features**"""

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

"""**train, test split**"""

# Convert target variable to a numpy array
Y = np.array(Y).reshape(-1, 1)

# Split data into training and testing sets (80% training, 20% testing)
X_train, X_test, Y_train, Y_test = train_test_split(X_scaled, Y, test_size=0.3,stratify=Y, random_state=42)

"""# Generic class for MLP model"""

from sklearn.metrics import mean_squared_error
from tqdm import tqdm_notebook

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

from sklearn.metrics import accuracy_score

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

# Instantiate the model with 24 inputs and 1 hidden layer with 14 neurons
ffsnn = NeuralNetwork(24, [14])

# Train the model
ffsnn.fit(X_train, Y_train, epochs=2500, learning_rate=0.01, display_loss=True)

results = evaluate_model(ffsnn, X_train, Y_train, X_test, Y_test)
print(f"Training Accuracy: {results['training_accuracy']}")
print(f"Testing Accuracy: {results['testing_accuracy']}")

# Calculate confusion matrices
conf_matrix_train = confusion_matrix(Y_train, results['train_predictions'])
conf_matrix_test = confusion_matrix(Y_test, results['testing_predictions'])

# Plot confusion matrices
def plot_confusion_matrix(conf_matrix, title):
    plt.figure(figsize=(6, 4))
    sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues', cbar=False)
    plt.title(title)
    plt.xlabel("Predicted Labels")
    plt.ylabel("True Labels")
    plt.show()

# Plot for training and validation
print("Confusion Matrix (Training):")
plot_confusion_matrix(conf_matrix_train, "Training Confusion Matrix")

print("Confusion Matrix (Testing):")
plot_confusion_matrix(conf_matrix_test, "Testing Confusion Matrix")

"""# Variations

**Affect of number of hidden layers**
"""

# Instantiate the model with 24 inputs and 2 hidden layer with 14,12 neurons
ffsnn1 = NeuralNetwork(24, [14,12])

# Train the model
ffsnn1.fit(X_train, Y_train, epochs=2500, learning_rate=0.01, display_loss=True)

results = evaluate_model(ffsnn1, X_train, Y_train, X_test, Y_test)
print(f"Training Accuracy: {results['training_accuracy']}")
print(f"Testing Accuracy: {results['testing_accuracy']}")

# Instantiate the model with 24 inputs and 3 hidden layer with 14,12,10 neurons
ffsnn2 = NeuralNetwork(24, [14,12,10])

# Train the model
ffsnn2.fit(X_train, Y_train, epochs=2500, learning_rate=0.01, display_loss=True)

results = evaluate_model(ffsnn2, X_train, Y_train, X_test, Y_test)
print(f"Training Accuracy: {results['training_accuracy']}")
print(f"Testing Accuracy: {results['testing_accuracy']}")

"""**Affect of standardizing the input attributes**"""

#when X is not scaled
# Split data into training and testing sets (80% training, 20% testing)
X_train1, X_test1, Y_train1, Y_test1 = train_test_split(X, Y, test_size=0.3,stratify=Y, random_state=42)

# Instantiate the model with 24 inputs and 1 hidden layer with 12 neurons
ffsnnNotScaled = NeuralNetwork(24, [12])

# Train the model
ffsnnNotScaled.fit(X_train1, Y_train1, epochs=2500, learning_rate=0.01, display_loss=True)

results = evaluate_model(ffsnnNotScaled, X_train1, Y_train1, X_test1, Y_test1)
print(f"Training Accuracy: {results['training_accuracy']}")
print(f"Testing Accuracy: {results['testing_accuracy']}")