## **1️⃣ Machine Learning Variant – Project Report**

---

### **Project Title**

**Human Detection using Machine Learning (Classical Computer Vision Approach)**

---

### **Aim**

To develop a machine learning model that detects the presence of humans in images using classical computer vision techniques and feature-based ML algorithms.

---

### **Why this project?**

1. **Real-world applications** – Security surveillance, pedestrian detection, search & rescue.
2. **Low computation requirement** – Works without GPUs, making it suitable for embedded devices.
3. **Educational value** – Demonstrates the effectiveness of traditional methods before moving to deep learning.

---

### **Dataset Used**

* **Name:** Human Detection Dataset (`constantinwerner/human-detection-dataset`)
* **Structure:** Contains positive (human) and negative (non-human) images.
* **Format:** Mostly `.jpg`/`.png`.
* **Size:** Around a few thousand images.

---

### **Proposed Methodology**

1. **Data Collection & Understanding**

   * Download dataset from Kaggle.
   * Separate into `1` and `0` folders.

2. **Data Preprocessing**

   * Convert all images to grayscale.
   * Resize images to a fixed size (e.g., 64×128 pixels).
   * Apply Histogram Equalization for better contrast.

   ![[Sample augmented images](Output/m1.png)](Output/m1.png)

3. **Feature Extraction**

   * Use **Histogram of Oriented Gradients (HOG)** to extract key shape-based features.
   * Normalize feature vectors.

   *(Insert Plot 2: Visualization of HOG features for a sample image)*

4. **Model Selection & Training**

   * Train models such as:

     * **SVM (Support Vector Machine)** – best for binary classification.
     * **Random Forest** – robust to overfitting.
   * Use `GridSearchCV` for hyperparameter tuning.

   *(Insert Plot 3: Training accuracy vs. validation accuracy graph)*

5. **Model Evaluation**

   * Metrics: Accuracy, Precision, Recall, F1-Score.
   * Confusion Matrix for classification performance.

   *(Insert Plot 4: Confusion matrix plot)*

6. **Prediction & Results**

   * Test model on unseen images.
   * Visualize correct vs. incorrect predictions.

   *(Insert Plot 5: Side-by-side comparison of predictions)*

---

### **Libraries Used**

* **OpenCV** – Image processing
* **scikit-learn** – ML models
* **matplotlib, seaborn** – Visualization
* **numpy** – Numerical operations

---

### **Algorithms Used**

* HOG Feature Extraction
* Support Vector Machine (SVM)
* Random Forest Classifier

---

### **Technologies & Tools**

* Python 3.x
* Jupyter Notebook / VS Code
* CPU-only environment

---

### **Conclusion**

The ML-based approach is lightweight and works well for small datasets, but struggles in complex scenarios with varied lighting, angles, and occlusion. This leads us to the **Deep Learning variant** for higher accuracy.

---
