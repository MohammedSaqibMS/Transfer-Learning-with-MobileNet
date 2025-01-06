# 🚀 Transfer Learning with MobileNetV2

This repository demonstrates the implementation of transfer learning using the MobileNetV2 model. It includes key steps such as dataset preprocessing, data augmentation, feature extraction, and model fine-tuning. The code leverages TensorFlow and Keras to showcase the power of transfer learning for image classification tasks.

---

## 📂 Table of Contents

1. [Introduction](#introduction)
2. [Dataset Preparation](#dataset-preparation)
3. [Data Augmentation](#data-augmentation)
4. [Transfer Learning with MobileNetV2](#transfer-learning-with-mobilenetv2)
5. [Model Summary](#model-summary)
6. [Predictions](#predictions)
7. [Credits](#credits)

---

## 🌟 Introduction

Transfer learning enables leveraging pre-trained models like MobileNetV2 to solve new tasks efficiently. This repository focuses on:
- Loading and preprocessing datasets 📊
- Using data augmentation to boost model generalization 🌈
- Fine-tuning the MobileNetV2 architecture for image classification tasks 🖼️

---

## 📦 Dataset Preparation

The dataset is loaded and split into training and validation subsets with an 80-20 split. Images are resized to 160x160 pixels to match the input dimensions expected by MobileNetV2.

```python
BATCH_SIZE = 32
IMG_SIZE = (160, 160)
directory = "dataset/"

train_dataset = image_dataset_from_directory(
    directory,
    shuffle=True,
    batch_size=BATCH_SIZE,
    image_size=IMG_SIZE,
    validation_split=0.2,
    subset='training',
    seed=42
)

validation_dataset = image_dataset_from_directory(
    directory,
    shuffle=True,
    batch_size=BATCH_SIZE,
    image_size=IMG_SIZE,
    validation_split=0.2,
    subset='validation',
    seed=42
)
```

---

## 🎨 Data Augmentation

To enhance model robustness, data augmentation techniques such as random horizontal flipping and rotation are applied.

```python
def data_augmenter():
    return tf.keras.Sequential([
        RandomFlip("horizontal"),
        RandomRotation(0.2)
    ])
```

---

## 🧠 Transfer Learning with MobileNetV2

MobileNetV2 is utilized with pre-trained weights from ImageNet. Key steps include freezing the base model's layers and attaching a custom classifier head for the target task.

```python
base_model = tf.keras.applications.MobileNetV2(
    input_shape=(160, 160, 3),
    include_top=False,
    weights='imagenet'
)
base_model.trainable = False
```

---

## 📝 Model Summary

The model's architecture is summarized below:

```
Model: "mobilenetv2_1.00_160"
Total params: 3,538,984
Trainable params: 3,504,872
Non-trainable params: 34,112
```

---

## 🔍 Predictions

Sample predictions on the test set include:

```
Image 1: [('n04589890', 'window_screen', 42.58%), ('n02708093', 'analog_clock', 9.27%)]
Image 2: [('n02883205', 'bow_tie', 35.23%), ('n02808440', 'bathtub', 8.19%)]
...
```

---

## 🙌 Credits

This implementation is inspired by the **Deep Learning Specialization** by [DeepLearning.AI](https://www.deeplearning.ai/courses/deep-learning-specialization/). 🌟 Special thanks to their comprehensive and insightful course materials!

---

## 📧 Contact

For any queries or collaboration opportunities, feel free to reach out!
