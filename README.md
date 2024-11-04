# Data-Science-and-ML-Breast-Cancer-Dataset
Breast Cancer Dataset model in python by Waako Lucius and Ademun Sarah

Breast Cancer Prediction using Neural Networks
Dataset:

This project utilizes the Breast Cancer Wisconsin (Diagnostic) Dataset. The dataset contains features computed from a digitized image of a fine needle aspirate (FNA) of a breast mass. The features describe characteristics of the cell nuclei present in the image.   

Data Preparation and Preprocessing:

Load Data:

The dataset is loaded into a Pandas DataFrame.
Handle Missing Values:

The dataset is checked for missing values. In this case, there are no missing values.
Data Splitting:

The dataset is split into training and testing sets using a stratified split to ensure that the proportion of classes in each set is similar to the original dataset.
Feature Scaling:

Feature scaling is applied to normalize the features, which can improve the performance of the neural network.
Model Building and Evaluation:

Neural Network Architecture:

A feedforward neural network with multiple hidden layers is used.
The hidden layers have 512 and 16 neurons, respectively.
The output layer has a single neuron with a sigmoid activation function to predict the probability of malignancy.
Model Compilation:

The model is compiled using the Adam optimizer, binary cross-entropy loss function, and accuracy metric.
Model Training:

The model is trained on the training set for a specified number of epochs.
L1/L2 model tuning is implemented to prevent overfitting.
Model Evaluation:

The trained model is evaluated on the testing set.
Performance metrics such as accuracy, precision, recall, and F1-score are calculated.
A confusion matrix is generated to visualize the model's predictions.
Additional Considerations:

Hyperparameter Tuning: Experiment with different hyperparameters (e.g., learning rate, batch size, number of layers, number of neurons) to optimize the model's performance.
Regularization: L1/L2 regularization are used to prevent overfitting.
