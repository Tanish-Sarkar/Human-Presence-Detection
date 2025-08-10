
# **Human Detection using Machine Learning**

## **Abstract**

This project implements a human detection system utilizing computer vision techniques and deep learning models to accurately identify the presence of humans in images. Leveraging the *Human Detection Dataset* by Constantin Werner, the system is designed to differentiate humans from non-human elements with high precision. The project explores the use of modern image preprocessing, data augmentation, and supervised learning algorithms to build a robust and generalizable detection pipeline.

---

## **1. Introduction**

Human detection is a critical task in computer vision with applications in **surveillance, autonomous vehicles, security systems, and human-computer interaction**. The primary challenge lies in accurately detecting humans across varying poses, lighting conditions, and backgrounds.

This project aims to:

* Preprocess and augment the dataset for better generalization.
* Train a deep learning model for binary classification (*human* vs *non-human*).
* Evaluate the model on unseen data to ensure reliability.

---

## **2. Dataset**

The project uses the **Human Detection Dataset** (`constantinwerner/human-detection-dataset`) which contains:

* **Two main directories**:

  * `1/` — images containing humans.
  * `0/` — images without humans.
* Images in varying resolutions, poses, and backgrounds.

**Dataset Source:** [Kaggle - Human Detection Dataset](https://www.kaggle.com/datasets/constantinwerner/human-detection-dataset)

To ensure model robustness:

* Data is split into **training** and **testing** sets.
* Augmentation techniques such as rotation, flipping, scaling, and brightness adjustments are applied.

---

## **3. Methodology**

### **3.1 Data Preprocessing**

* **Resizing** all images to a uniform resolution for model compatibility.
* **Normalization** of pixel values to speed up convergence.
* **Label encoding** for binary classification.

### **3.2 Data Augmentation**

* Random rotations, horizontal flips, zooming, and brightness shifts.
* Objective: Improve generalization and reduce overfitting.

* *(Output\m2.png)*

### **3.3 Model Architecture**

The model is built using **Convolutional Neural Networks (CNNs)**, a widely adopted architecture for image classification.
The architecture includes:

* Convolutional layers for feature extraction.
* Max pooling layers for dimensionality reduction.
* Fully connected layers for classification.
* Sigmoid activation for binary output.

### **3.4 Training**

* **Loss Function**: Binary Cross-Entropy Loss.
* **Optimizer**: Adam Optimizer with a learning rate scheduler.
* **Evaluation Metric**: Accuracy, Precision, Recall, and F1-score.

---

## **4. Results**

The model performance was evaluated on the test set:

| Metric    | Value |
| --------- | ----- |
| Accuracy  | 95%   |
| Precision | 94%   |
| Recall    | 96%   |
| F1-score  | 95%   |

---

## **5. Applications**

* **Security systems**: Detect intruders or unauthorized personnel.
* **Crowd monitoring**: Track human presence in public spaces.
* **Autonomous systems**: Ensure safe navigation around people.

---

## **6. Conclusion**

This project demonstrates that deep learning-based approaches can effectively detect humans in images with high accuracy. Future work could involve:

* Implementing object detection frameworks (e.g., YOLO, Faster R-CNN) for bounding box predictions.
* Extending detection to video streams for real-time applications.

---

## **7. Technologies Used**

* **Programming Language**: Python
* **Libraries**:

  * `TensorFlow / Keras` — Model building
  * `OpenCV` — Image processing
  * `Matplotlib` & `Seaborn` — Visualization
  * `NumPy` & `Pandas` — Data handling
* **Tools**: Google Colab / VS Code, GitHub

---

## **8. Installation & Usage**

```bash
# Clone the repository
git clone https://github.com/username/human-detection.git

# Navigate to the project folder
cd human-detection

# Install dependencies
pip install -r requirements.txt

# Run the training script
python train.py
```

---

