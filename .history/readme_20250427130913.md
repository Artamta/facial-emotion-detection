# Emotion Recognition Using Deep Learning

This project implements a deep learning pipeline for emotion recognition using the CK+ dataset. The model is based on the VGG19 architecture and is trained using PyTorch. The project includes data preprocessing, model training, evaluation, and inference, with support for 5-fold cross-validation.

---

## **Project Proposal**

### **Problem Description**
Facial expressions are a key aspect of human communication, conveying emotions and intentions. Automating the recognition of emotions from facial expressions has applications in mental health, human-computer interaction, and security. This project aims to classify facial expressions into seven emotion categories (Angry, Disgust, Fear, Happy, Sadness, Surprise, Contempt) using deep learning.

### **Input-Output Statement**
- **Input**: A grayscale image of a face (48x48 pixels) in `.jpg` format.
- **Output**: A predicted emotion category (one of the seven classes).

### **Data Source**
The CK+ dataset, sourced from [Kaggle](https://www.kaggle.com/datasets/shuvoalok/ck-dataset/data), is used for training and evaluation. It contains labeled facial expression images for seven emotion categories.

### **Model Architecture**
The model is based on the VGG19 architecture, modified to accept grayscale images. Dropout layers and weight decay are used for regularization to prevent overfitting. The choice of VGG19 is motivated by:
- Its proven performance in image classification tasks.
- Its ability to extract hierarchical features from images.
- Its simplicity and adaptability for grayscale inputs.

### **Why Not Adam Optimizer?**
Adam optimizer, while effective for faster convergence, can lead to overfitting on small datasets like CK+. Instead, SGD with momentum is used for better generalization and stability.

### **Expected Outcome**
The model will achieve an average test accuracy of ~65% across 5-fold cross-validation, demonstrating its ability to generalize to unseen data.

---

## **What I Have Done**
1. **Dataset Preparation**:
   - Downloaded the CK+ dataset from [Kaggle](https://www.kaggle.com/datasets/shuvoalok/ck-dataset/data).
   - Preprocessed the dataset by resizing images to **48x48 pixels** and converting them to grayscale.
   - Applied data augmentation techniques such as random horizontal flip, rotation, affine transformations, and more to improve generalization.

2. **Model Architecture**:
   - Used the **VGG19** architecture as the base model.
   - Modified the first convolutional layer to accept grayscale images (1 channel).
   - Added dropout layers and weight decay to prevent overfitting.

3. **Training**:
   - Implemented a training pipeline with **5-fold cross-validation** to evaluate the model's generalization ability.
   - Used **Stochastic Gradient Descent (SGD)** with momentum as the optimizer to avoid overfitting, instead of Adam.
   - Integrated a learning rate scheduler to reduce the learning rate during training for better convergence.

4. **Evaluation**:
   - Evaluated the model on the test set of each fold using accuracy and loss as metrics.
   - Saved the best model weights for each fold.

5. **Inference**:
   - Developed a script (`predict.py`) to classify new images into one of the seven emotion categories.

---

## **Directory Structure**
The project follows the required directory structure:

```
project_ayush_raj/
|
|_ checkpoints/
|   |_ final_weights.pth_fold1.pth
|   |_ final_weights.pth_fold2.pth
|   |_ ...
|
|_ data/
|   |_ CK_data.h5
|   |_ img01.jpg
|   |_ img02.jpg
|   |_ ...
|
|_ dataset.py
|_ model.py
|_ train.py
|_ predict.py
|_ config.py
|_ interface.py
|_ readme.md
```

### **Key Files**
- **`dataset.py`**: Defines the custom dataset and dataloader.
- **`model.py`**: Contains the VGG-based model architecture.
- **`train.py`**: Implements the training loop with 5-fold cross-validation.
- **`predict.py`**: Provides inference functionality for emotion classification.
- **`config.py`**: Stores hyperparameters and configuration details.
- **`interface.py`**: Standardizes function and class names for grading.
- **`readme.md`**: Documentation for the project.

---

## **How to Run the Project**

### **Step 1: Clone the Repository**
Clone the repository to your local machine:
```bash
git clone https://github.com/your_username/project_ayush_raj.git
cd project_ayush_raj
```

### **Step 2: Install Dependencies**
Install the required Python packages:
```bash
pip install -r requirements.txt
```

### **Step 3: Prepare the Dataset**
Ensure the CK+ dataset is downloaded and preprocessed. Place the preprocessed dataset (`CK_data.h5`) in the `data/` directory.

### **Step 4: Train the Model**
Run the training script to train the model with 5-fold cross-validation:
```bash
python train.py
```
This will save the best model weights for each fold in the `checkpoints/` directory.

### **Step 5: Perform Inference**
Use the `predict.py` script to classify new images:
```bash
python predict.py --image_path data/img01.jpg
```

---

## **Dataset**
The CK+ dataset is used for this project. It contains labeled facial expression images for seven emotion categories. The dataset is preprocessed and stored in an HDF5 file (`CK_data.h5`).

### **Preprocessing**
- Images are resized to **48x48 pixels**.
- Grayscale images are used as input to the model.
- Data augmentation techniques include:
  - Random horizontal flip
  - Random rotation
  - Random affine transformations
  - Color jitter
  - Random cropping
  - Random perspective transformations
  - Random erasing

---

## **Model Architecture**
The model is based on the **VGG19** architecture, with the following modifications:
- The first convolutional layer is adjusted to accept grayscale images (1 channel).
- Dropout layers are added to the fully connected layers for regularization.
- The final fully connected layer outputs predictions for 7 emotion classes.

### **Regularization Techniques**
- **Dropout**: Applied with a probability of 0.5 to prevent overfitting.
- **Weight Decay**: Set to `1e-4` in the optimizer to penalize large weights.

---

## **Training**
The training process includes:
- **5-Fold Cross-Validation**: The dataset is split into 5 folds, and the model is trained and evaluated on each fold.
- **Optimizer**: Stochastic Gradient Descent (SGD) with momentum.
- **Loss Function**: Cross-Entropy Loss.
- **Learning Rate**: Set to `0.01` with a step decay schedule.

### **Training Pipeline**
1. Load the dataset using `dataset.py`.
2. Apply data augmentation and preprocessing.
3. Train the model for 60 epochs per fold.
4. Save the best model weights for each fold in the `checkpoints/` directory.

---

## **Evaluation**
The model is evaluated on the test set of each fold. Metrics include:
- **Accuracy**: Percentage of correctly classified samples.
- **Loss**: Cross-entropy loss on the test set.

The final performance is reported as the **average accuracy across all 5 folds**.

---

## **Inference**
The `predict.py` script provides functionality for emotion classification on new images. It accepts a list of image paths and outputs the predicted emotion for each image.

### **Usage**
```python
from predict import classify_images

# List of image paths
image_paths = ["data/img01.jpg", "data/img02.jpg"]

# Get predictions
predictions = classify_images(image_paths)
print(predictions)
```

---

## **Results**
### **Training Performance**
- **Training Accuracy**: ~92% (average across folds)
- **Test Accuracy**: ~64% (average across folds)

### **Cross-Validation Results**
| Fold | Test Accuracy |
|------|---------------|
| 1    | 64.65%        |
| 2    | 67.68%        |
| 3    | TBD           |
| 4    | TBD           |
| 5    | TBD           |

**Average Test Accuracy**: **TBD**

---

## **Future Work**
- **Hyperparameter Tuning**: Experiment with different learning rates, batch sizes, and dropout rates.
- **Advanced Architectures**: Explore more advanced models like ResNet or EfficientNet.
- **Data Augmentation**: Add more augmentation techniques to improve generalization.
- **Transfer Learning**: Use pre-trained models for better performance on small datasets.

---

## **Acknowledgments**
- **CK+ Dataset**: Used for training and evaluation.
- **PyTorch**: Framework for building and training the model.
- **Instructors and Peers**: For guidance and support throughout the project.