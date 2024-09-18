# ECG Anomaly Detection using Autoencoder in PyTorch

## Introduction

This project is inspired by the Kaggle notebook ["Detecting Anomalies using Autoencoder"](https://www.kaggle.com/code/ohseokkim/dectecting-anomaly-using-autoencoder/notebook) by OH SEOK KIM, where an autoencoder model is used to detect anomalies in ECG signals. The ECG signals can also be found on the kaggle link.
While the original notebook employs TensorFlow, this implementation is done using the PyTorch library. The objective is to replicate the anomaly detection approach with PyTorch and analyze ECG signals effectively. This has enabled me to improve my mastery of the PyTorch library.
The use of the PyTorch library mostly changed the way the data is handled compared to how Tensorflow does it, aswell as how the auto encoder model is created and trained.

## Project Steps

1. **Data Loading and Quick EDA:**
   - First the ECG signal dataset is loaded and we perform a short initial exploratory data analysis (EDA) to understand the structure and characteristics of the data.

2. **Label Distribution:**
   - Then we analyze the distribution of the labels in the dataset to understand the balance between normal and abnormal signals.

3. **Visualization of Examples:**
   - We then visualize examples of normal and abnormal ECG signals to get insights into their patterns and characteristics.

4. **Model Architecture and Training:**
   - We can then define and train an autoencoder model. The architecture includes an encoder and a decoder to learn how to reconstruct ECG signals.

5. **Visualization of Reconstructed Signals:**
   - Once the model trained we can plot normal and abnormal ECG signals along with their reconstructed signals from the autoencoder to evaluate how well the model performs in reconstructing normal versus abnormal signals.

6. **Defining Thresholds:**
   - Finally we can find a threshold that separates normal and abnormal signals based on the reconstruction loss.

7. **Prediction and Performance Evaluation:**
   - We then use this threshold to predict with out model whether a signal is normal or abnormal using the defined threshold. An evaluation of the performance of the model is then made by computing different metrics such as the accuracy, precision and recall, and plotting a confusion matrices to visualize clearly the performances of the model.

## Required Libraries  
This project requires the following Python libraries: 
- `numpy`
- `pandas`
- `matplotlib`
- `seaborn`
- `torch`
- `scikit-learn`
- `plotly`
- `umap`
