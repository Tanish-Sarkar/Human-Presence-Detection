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

   *![[Plot 1](../Output/m1.png)](../Output/m1.png)*

3. **Feature Extraction**

   * Use **Histogram of Oriented Gradients (HOG)** to extract key shape-based features.
   * Normalize feature vectors.

   *![[Plot 2](../Output/m2.png)](../Output/m2.png)*

4. **Model Selection & Training**

   * Train models such as:

     * **SVM (Support Vector Machine)** – best for binary classification.
     * **Random Forest** – robust to overfitting.
   * The Comparision for the accuracy of the models are as follow  

   *![[Plot 3](../Output/m3.png)](../Output/m3.png)*

5. **Model Evaluation**

   * Metrics: Accuracy, Precision, Recall, F1-Score.
   * Confusion Matrix for classification performance.

   *![[Plot 4](../Output/m4.png)](../Output/m4.png)*

6. **The Pricipal Component Analysis Implementation**
    *![[Plot 5](../Output/m5.png)](../Output/m5.png)*

7. **Prediction & Results**

   * Test model on unseen images.
   * Visualize correct vs. incorrect predictions.

   *![[Plot 6](../Output/m6.png)](../Output/m6.png)*
   *![[Plot 7](../Output/m7.png)](../Output/m7.png)*

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
* VS Code
* CPU-only environment

---

### **Conclusion**

**The ML-based approach is lightweight and works well for small datasets, but struggles in complex scenarios with varied lighting, angles, and occlusion. This leads us to the Deep Learning variant for higher accuracy**.
---
