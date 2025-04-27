
### **Key Files**
- **`dataset.py`**: Defines the custom dataset and dataloader.
- **`model.py`**: Contains the VGG-based model architecture.
- **`train.py`**: Implements the training loop with 5-fold cross-validation.
- **`predict.py`**: Provides inference functionality for emotion classification.
- **`config.py`**: Stores hyperparameters and configuration details.
- **`interface.py`**: Standardizes function and class names for grading.

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