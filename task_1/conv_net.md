# Convolutional Neural Networks (CNNs)

## 1. Introduction

A **Convolutional Neural Network (CNN)** is a class of deep learning models specifically designed to process structured grid-like data, such as images or reshaped multidimensional feature matrices. CNNs automatically learn hierarchical spatial features through convolution operations, making them highly effective for pattern recognition tasks.

CNNs gained prominence with **Yann LeCun**'s LeNet-5 (1998) for handwritten digit recognition and became mainstream after **AlexNet** won the ImageNet competition in 2012 by a large margin.

Key advantages over fully connected networks include:

- **Local connectivity** — neurons respond only to a small spatial region
- **Weight sharing** — same filter applied across the input
- **Spatial hierarchies** — low-level features (edges) to high-level (objects)
- **Pooling** for dimensionality reduction and translation invariance

These reduce parameters dramatically and improve generalization.

![Convolution Example](https://miro.medium.com/1*tNQvssqUaiYteDpREHQyFw.png)  
_Figure 1: Kernel sliding over input during convolution (source: Medium article on convolutions)._

![Another Convolution Diagram](https://media.licdn.com/dms/image/v2/D4D12AQHMz5kTcCa7eQ/article-cover_image-shrink_720_1280/article-cover_image-shrink_720_1280/0/1673940856506?e=2147483647&v=beta&t=0rcjdI1uJ1Jq5n_OIVyMqeMS_Q9K8Ll3FEEkbB5sAkk)  
_Figure 2: 3D visualization of convolution operation._

## 2. Main Components of a CNN

### 2.1 Convolution Layer

The core operation: a learnable kernel slides over the input to produce **feature maps**.

Mathematical representation (simplified 2D):

**Output(i, j) = Σ (Input(i+m, j+n) × Kernel(m, n)) + bias**

Hyperparameters:

- Kernel size (e.g., 3×3, 5×5)
- Stride
- Padding (valid / same)

This detects local patterns like edges, textures, etc.

### 2.2 Activation Function

Most common: **ReLU** — _ReLU(x) = max(0, x)_

Introduces non-linearity, speeds up training, and mitigates vanishing gradients.

### 2.3 Pooling Layer

Downsamples feature maps while preserving key information.  
Common type: **Max Pooling** (takes the maximum in each window).

Example (2×2 max pooling):

Input region:
1 3
5 6

→ Output: 6

Benefits:

- Reduces spatial size → fewer parameters
- Controls overfitting
- Adds translation invariance

![Typical CNN Architecture](https://media.geeksforgeeks.org/wp-content/uploads/20240524180818/lenet-min.PNG)  
_Figure 3: LeNet-5 architecture overview (classic CNN example)._

![AlexNet Architecture](https://miro.medium.com/v2/resize:fit:1400/1*B_ZaaaBg2njhp8SThjCufA.png)  
_Figure 4: AlexNet — deeper CNN that popularized modern deep learning._

### 2.4 Fully Connected Layers

At the end: flatten the feature maps → feed into dense layers → classification/regression output.

For binary tasks: final layer uses **sigmoid** activation.

## 3. Training Process

Standard deep learning workflow:

1. Forward pass → compute predictions
2. Calculate loss (e.g., Binary Cross-Entropy for classification)
3. Backpropagation → compute gradients
4. Update weights (optimizer: Adam, SGD, etc.)

Convolution filters are learned just like other weights.

## 4. Practical Example in Cybersecurity: Network Intrusion Detection

CNNs are widely used in **Intrusion Detection Systems (IDS)** to classify network traffic as normal or malicious (attack).

A popular dataset for this is **NSL-KDD** (improved version of KDD Cup 1999), publicly available for download

Dataset: NSL-KDD — download KDDTrain+.txt from https://www.unb.ca/cic/datasets/nsl.html

It includes ~41 network traffic features per connection, such as:

- duration
- protocol_type
- service
- flag
- src_bytes
- dst_bytes
- ...

Labels: normal or various attack types. For simplicity, we use **binary classification**: 0 = normal, 1 = attack.

Features are normalized, categorical columns encoded, padded slightly, and reshaped into a small 2D matrix (e.g., 7×6) so a 2D CNN can capture local correlations between related features.

### Illustrative Data Example (single row snippet)

duration, protocol_type, service, flag, src_bytes, dst_bytes, ..., label
0, tcp, http, SF, 181, 5450, ..., normal

## 5. Python Code Example (Keras/TensorFlow)

```python
import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt

column_names = [f"f{i}" for i in range(41)] + ["label", "difficulty"]
data = pd.read_csv("KDDTrain+.txt", names=column_names)
data = data.drop("difficulty", axis=1)

data["label"] = data["label"].apply(lambda x: 0 if x == "normal" else 1)

for col in data.select_dtypes(include=["object"]).columns:
    if col != "label":
        data[col] = LabelEncoder().fit_transform(data[col])

X = data.drop("label", axis=1)
y = data["label"]

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

X_padded = np.pad(X_scaled, ((0,0),(0,1)), mode='constant')
X_reshaped = X_padded.reshape(-1,7,6,1)

X_train, X_test, y_train, y_test = train_test_split(
    X_reshaped, y, test_size=0.2, random_state=42
)

model = models.Sequential([
    layers.Conv2D(32, (3,3), activation='relu', input_shape=(7,6,1)),
    layers.MaxPooling2D((2,2)),
    layers.Conv2D(64, (2,2), activation='relu'),
    layers.Flatten(),
    layers.Dense(64, activation='relu'),
    layers.Dropout(0.5),
    layers.Dense(1, activation='sigmoid')
])

model.compile(
    optimizer='adam',
    loss='binary_crossentropy',
    metrics=['accuracy']
)

history = model.fit(
    X_train, y_train,
    epochs=10,
    batch_size=64,
    validation_data=(X_test, y_test)
)

test_loss, test_acc = model.evaluate(X_test, y_test)
print("Test Accuracy:", test_acc)

plt.plot(history.history['accuracy'])
plt.plot(history.history['val_accuracy'])
plt.title("Model Accuracy")
plt.xlabel("Epoch")
plt.ylabel("Accuracy")
plt.legend(["Train", "Validation"])
plt.show()


6. Conclusion
Convolutional Neural Networks are powerful tools for extracting hierarchical patterns from grid-structured data. Originally developed for images, their flexibility allows effective application to cybersecurity tasks like network intrusion detection.
By reshaping tabular network features into 2D matrices and applying a CNN (as shown with the NSL-KDD dataset), we can achieve strong classification performance on normal vs. malicious traffic. This demonstrates CNNs' broad utility beyond traditional computer vision.
```
