# Rock, Paper, Scissors CNN Classifier

This project implements a Convolutional Neural Network (CNN) to classify hand gesture images of Rock, Paper, and Scissors. The model is built using PyTorch and trained on the Rock Paper Scissors dataset from Kaggle. All the implementation details and analysis are provided in the `PES2UG23CS363_Lab-14.ipynb` Jupyter Notebook.

## Table of Contents
- [Objective](#objective)
- [Dataset](#dataset)
- [Methodology](#methodology)
  - [1. Setup and Data Download](#1-setup-and-data-download)
  - [2. Imports and Device Setup](#2-imports-and-device-setup)
  - [3. Data Loading and Preprocessing](#3-data-loading-and-preprocessing)
  - [4. CNN Model Architecture](#4-cnn-model-architecture)
  - [5. Model Training](#5-model-training)
  - [6. Model Evaluation](#6-model-evaluation)
  - [7. Single Image Testing](#7-single-image-testing)
  - [8. Interactive Game](#8-interactive-game)
- [Results](#results)
- [How to Run](#how-to-run)
- [Dependencies](#dependencies)
- [File Structure](#file-structure)

## Objective

The main objective of this lab is to build, train, and test a Convolutional Neural Network (CNN) that can accurately classify images of hands showing Rock, Paper, or Scissors gestures. This project demonstrates the application of deep learning for image classification tasks and provides hands-on experience with:

- Building CNN architectures using PyTorch
- Implementing image preprocessing and data augmentation
- Training deep learning models on image datasets
- Evaluating model performance on test data
- Deploying the model for real-world predictions

## Dataset

The dataset used is the **Rock Paper Scissors** dataset from Kaggle (`drgfreeman/rockpaperscissors`). It contains images of hands displaying three different gestures:

- **Rock**: Closed fist gesture
- **Paper**: Open palm gesture
- **Scissors**: Two fingers extended gesture

The dataset includes approximately 2,188 images distributed across the three classes. Images are in various sizes and are preprocessed to a standard 128x128 pixel format for training.

**Dataset Split:**
- Training set: 80% of the data (1,750 images)
- Test set: 20% of the data (438 images)

## Methodology

### 1. Setup and Data Download

The project begins by downloading the Rock Paper Scissors dataset from Kaggle using the `kagglehub` library. The dataset is then organized into a local directory structure with separate folders for each class (rock, paper, scissors).

### 2. Imports and Device Setup

The necessary libraries are imported, including:
- **PyTorch**: For building and training the neural network
- **torchvision**: For image transformations and dataset handling
- **PIL**: For image processing
- **NumPy**: For numerical operations

The code automatically detects and configures the device (CPU or GPU) for optimal performance.

### 3. Data Loading and Preprocessing

Images undergo several preprocessing transformations:
- **Resizing**: All images are resized to 128x128 pixels for consistency
- **Tensor Conversion**: Images are converted to PyTorch tensors
- **Normalization**: Pixel values are normalized with mean=0.5 and std=0.5 for each RGB channel

The dataset is loaded using `ImageFolder` from torchvision and split into training (80%) and testing (20%) sets. DataLoaders are created with appropriate batch sizes for efficient training and evaluation.

### 4. CNN Model Architecture

The CNN architecture consists of multiple layers designed to extract hierarchical features from the images:

**Convolutional Layers:**
- Multiple convolutional layers with increasing filter sizes to capture features at different scales
- ReLU activation functions for non-linearity
- Max pooling layers for spatial dimension reduction

**Fully Connected Layers:**
- Flatten layer to convert 2D feature maps to 1D vectors
- Dense layers for final classification
- Output layer with 3 neurons (one for each class)

The model uses cross-entropy loss and an optimizer (typically Adam or SGD) for training.

### 5. Model Training

The training process involves:
- **Forward Pass**: Input images are passed through the network to generate predictions
- **Loss Calculation**: Cross-entropy loss is computed between predictions and true labels
- **Backward Pass**: Gradients are computed using backpropagation
- **Parameter Update**: Model weights are updated using the optimizer

The model is trained for multiple epochs, with training loss monitored at each step to track learning progress.

### 6. Model Evaluation

After training, the model is evaluated on the test set to assess its performance:
- **Accuracy Calculation**: The percentage of correctly classified images
- **Loss Computation**: The average loss on the test set
- **Confusion Analysis**: Understanding which classes are most often confused

The evaluation provides insights into the model's generalization capability on unseen data.

### 7. Single Image Testing

The notebook includes functionality to test the trained model on individual images:
- Load a single image from the dataset
- Apply the same preprocessing transformations
- Pass it through the trained model
- Display the predicted class along with the actual class

This demonstrates how the model can be used for inference on new data.

### 8. Interactive Game

An interactive game feature is implemented where:
- Users can provide input images (Rock, Paper, or Scissors)
- The model predicts the user's gesture
- The computer makes a random choice
- The game logic determines the winner based on traditional Rock, Paper, Scissors rules

This makes the project engaging and demonstrates a practical application of the trained model.

## Results

The trained CNN model achieves high accuracy in classifying Rock, Paper, and Scissors gestures. Key performance metrics include:

- **Training Accuracy**: Typically >90% after sufficient training epochs
- **Test Accuracy**: Demonstrates good generalization on unseen data
- **Classification Performance**: The model successfully distinguishes between the three hand gestures with minimal confusion

The model performs well across all three classes, showing that the CNN architecture is effective for this image classification task.

## How to Run

### Prerequisites
1. Ensure you have Python 3.7+ installed
2. Install Jupyter Notebook or JupyterLab
3. Set up a Kaggle account and API credentials (for dataset download)

### Steps
1. Clone this repository to your local machine
2. Navigate to the `ML_Lab_week-14` directory
3. Install the required dependencies (see below)
4. Open the Jupyter Notebook:
   ```bash
   jupyter notebook PES2UG23CS363_Lab-14.ipynb
   ```
5. Run all cells sequentially to:
   - Download and prepare the dataset
   - Build and train the CNN model
   - Evaluate the model performance
   - Test on individual images
   - Play the interactive game

### Running in Google Colab
Alternatively, you can run this notebook in Google Colab for access to free GPU resources:
1. Upload the notebook to Google Colab
2. Enable GPU runtime (Runtime → Change runtime type → GPU)
3. Run all cells

## Dependencies

The following Python libraries are required to run the notebook:

- `torch` (PyTorch): Deep learning framework
- `torchvision`: Computer vision utilities for PyTorch
- `kagglehub`: For downloading datasets from Kaggle
- `numpy`: Numerical computing
- `PIL` (Pillow): Image processing
- `matplotlib`: Visualization (if used for plotting)

### Installation

You can install the required dependencies using pip:

```bash
pip install torch torchvision kagglehub numpy pillow matplotlib
```

For GPU support with PyTorch, refer to the [PyTorch installation guide](https://pytorch.org/get-started/locally/) to install the appropriate version for your system.

## File Structure

```
ML_Lab_week-14/
├── PES2UG23CS363_Lab-14.ipynb      # Main Jupyter notebook with implementation
├── PES2UG23CS363_Lab-14_Report.pdf  # Detailed lab report
└── README.md                        # This file
```

---

**Author**: Nandan D (PES2UG23CS363)  
**Course**: Machine Learning Lab  
**Week**: 14 - Convolutional Neural Networks
