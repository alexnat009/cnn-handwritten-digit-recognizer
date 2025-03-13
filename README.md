# **Handwritten Digit Recognition Using CNN**

## **Project Overview**

This project implements a **Convolutional Neural Network (CNN)** for recognizing handwritten digits using the **MNIST dataset**. The model is trained using **K-Fold Cross-Validation** and evaluates performance through accuracy metrics and visualization plots. The project includes functions for **loading, normalizing, training, and testing the model**, as well as predicting digit classes from images.

## **Model Architecture**

The CNN model consists of:

- **2 Convolutional Layers** (ReLU activation)
- **1 Max-Pooling Layer**
- **Dropout Layer** (prevents overfitting)
- **Flatten Layer**
- **Fully Connected Dense Layer** (128 neurons, ReLU activation)
- **Output Layer** (10 neurons, Softmax activation for digit classification)

The model is compiled using the **Adam optimizer** with **categorical cross-entropy loss**.

## **Dataset**

The **MNIST dataset** consists of **28x28 grayscale images** of handwritten digits (0-9). It is split into training and testing sets:

- **Training Set**: 60,000 images
- **Testing Set**: 10,000 images

The dataset is automatically loaded using TensorFlow.

## **Installation & Setup**

1)**Clone the repository:**

```bash
git clone https://github.com/alexnat009/cnn-handwritten-digit-recognizer.git
cd cnn-handwritten-digit-recognizer
```

2)**Install dependencies:**

```bash
pip install numpy pandas matplotlib seaborn tensorflow
```

3)**Run the model training:**

```bash
python main.py
```

## **Model Training & Evaluation**

- The model is trained for **10 epochs**
- **K-Fold Cross-Validation** is applied (default: **5 folds**)
- Training and validation accuracy/loss are plotted for analysis

### **Visualization**

The script generates **accuracy & loss curves** for performance monitoring:  
![Model Accuracy & Loss](./assets/training_plot.png)

## **Digit Prediction**

The model is used to predict handwritten digits from the **MNIST test set**:

```python
img = load_image(X_test[0])
predict_value = model.predict(img)
digit = np.argmax(predict_value)
print(f'Predicted Digit: {digit}')
```

## **License**

This project is **open-source** under the [MIT License](LICENSE).
