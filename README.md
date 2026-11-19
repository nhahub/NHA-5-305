# Land Type Classification Using Deep Learning and Satellite Imagery

**Graduation Project | Digital Egypt Pioneers Initiative (DEPI)**

## Team Members

**Project Leader**
- Mohamed Fathy Elsayed Youssef

**Team Members**
- Mohamed Masoud Saleh Masoud
- Mohamed Nasser Elden Abdelfattah Ahmed Alam
- Mohamed Elshahat Rizk Jafar
- Menna Allah Mahmoud Ali Abdelrazek
- Ahmed Hossam Mohamed AlAnany

---

## Project Overview

This project focuses on land use and land cover classification using Deep Learning and Sentinel-2 satellite imagery.

The main objective is to develop an image classification system capable of identifying 10 different land cover categories, including agricultural areas, forests, residential regions, industrial zones, roads, and water bodies.

The project uses the EuroSAT dataset to explore satellite image analysis, image preprocessing, deep learning model development, and classification performance evaluation.

The system is also designed to provide an interactive interface for uploading satellite images and displaying classification results.

## Dataset

The project is based on the **EuroSAT (European Satellite) Dataset**, developed by Patrick Helber and colleagues using imagery from the Sentinel-2 satellite mission.

EuroSAT is widely used for land use and land cover classification research.

### Dataset Information

| Property | Description |
|---|---|
| Dataset Name | EuroSAT |
| Satellite | Sentinel-2 |
| Total Images | 27,000 |
| Number of Classes | 10 |
| Image Resolution | 64 × 64 pixels |
| Image Format | RGB (JPEG) |
| Number of Channels | 3 |
| Classification Type | Multiclass Image Classification |

### Land Cover Categories

The dataset contains the following 10 classes:

1. Annual Crop
2. Forest
3. Herbaceous Vegetation
4. Highway
5. Industrial
6. Pasture
7. Permanent Crop
8. Residential
9. River
10. Sea/Lake

### Dataset Sources

**Official EuroSAT Dataset Repository:**

https://github.com/phelber/EuroSAT

**EuroSAT RGB Dataset — Download:**

https://zenodo.org/records/7711810

**EuroSAT Dataset Documentation:**

https://www.tensorflow.org/datasets/catalog/eurosat

The project primarily uses the RGB version of EuroSAT for image classification.

The original dataset also provides a multispectral version containing 13 spectral bands, allowing further investigation of the impact of spectral information on classification performance.

---

## Project Methodology

The project follows a Deep Learning workflow consisting of the following stages.

### 1. Data Exploration and Analysis

Exploratory Data Analysis (EDA) is performed to understand the dataset characteristics, visualize satellite images, and examine the distribution of land cover categories.

### 2. Data Preprocessing

Satellite images are prepared for model training using:

- Image resizing
- Image normalization
- Training, validation, and testing splits
- Data augmentation

### 3. Deep Learning Models

The project investigates Convolutional Neural Networks and Transfer Learning approaches.

The main architectures include:

**CNN:** A custom convolutional neural network for land cover classification.

**ResNet50:** A pretrained deep learning model used for feature extraction and classification.

**EfficientNet-B0:** An efficient pretrained architecture used for comparison with ResNet50.

### 4. Model Evaluation

Classification performance is evaluated using:

- Accuracy
- Precision
- Recall
- F1-Score
- Confusion Matrix

The evaluation process helps compare different models and identify classification errors.

### 5. Multispectral Image Classification

The project also investigates the use of Sentinel-2 multispectral imagery containing 13 spectral bands.

This analysis aims to compare RGB and multispectral image classification and examine the effect of additional spectral information.

### 6. Explainable AI

Grad-CAM is used to investigate which image regions influence the model's classification decisions.

---

## Interactive Dashboard

An interactive dashboard using **Streamlit** is part of the proposed system.

The dashboard is designed to allow users to:

- Explore satellite imagery and land cover categories.
- View dataset statistics and visualizations.
- Examine model evaluation results.
- Upload satellite images for classification.
- View predicted land cover categories and confidence scores.

## Technologies

The main technologies and tools used in the project include:

- **Python:** Main programming language.
- **PyTorch & Torchvision:** Deep Learning model development and training.
- **Scikit-learn:** Model evaluation and classification metrics.
- **NumPy & Pandas:** Data processing and analysis.
- **Matplotlib & Seaborn:** Data visualization.
- **Jupyter Notebook & Google Colab:** Development and experimentation.
- **Streamlit:** Interactive dashboard development.
- **Git & GitHub:** Version control and project collaboration.

## Applications

Satellite-based land cover classification can support several applications, including:

- Environmental monitoring
- Agricultural land analysis
- Urban planning
- Land use classification
- Remote sensing research

## Future Enhancements

Further improvements may include evaluating the classification models on Egyptian satellite imagery, improving model generalization, and expanding multispectral classification capabilities.

---

**Digital Egypt Pioneers Initiative (DEPI) — Graduation Project**
