# ♻️ EcoSort-RecycleIR
### AI-Powered Waste Classification & Recycling Guidance

## 📌 Project Overview

**EcoSort-RecycleIR** is a computer vision project that classifies waste images into 10 categories using deep learning and transfer learning with **EfficientNetV2S** pretrained on ImageNet.
Beyond image classification, the project integrates a structured recycling knowledge base that provides category-specific disposal instructions, recycling recommendations, and safety precautions based on the predicted waste type.
The project connects AI-powered waste recognition with practical recycling guidance to support more responsible waste sorting.

## 🚀 Live Demo

Try the deployed Streamlit application:

**[Launch EcoSort-RecycleIR](https://ecovision-recycling-2nr25qyqddet7dp7ha8d8e.streamlit.app/)**

## 🎥 Application Demo Video

[Watch the EcoSort-RecycleIR App Demo](https://drive.google.com/file/d/1rJ6wMq06SpyQxptg-TEOeKUtCz9fkgAQ/view?usp=sharing)

## 🎥 MLflow Demo

[Watch the MLflow Demo](https://drive.google.com/file/d/1RfhGmA7K1SU_yV9UU9cPFe14zwKaDN04/view?usp=sharing)

## 🔄 Kaggle Notebook

[Explore the Kaggle Notebook](https://www.kaggle.com/code/ranaahmedelmaghraby/waste-ai-transfer-learning)

## 🎯 Objectives

- Classify waste images into 10 categories.
- Apply transfer learning using EfficientNetV2S.
- Prepare and explore the image dataset.
- Monitor training and validation performance.
- Evaluate the trained model on unseen test data.
- Generate a classification report and confusion matrix.
- Save the trained model and supporting artifacts.
- Build a structured recycling knowledge base.
- Implement a reusable recycling guide function.
- Integrate image classification with recycling guidance.
- Develop and deploy an interactive Streamlit web application.

## 🗂️ Waste Categories

The model recognizes the following categories:

1. Battery
2. Biological
3. Cardboard
4. Clothes
5. Glass
6. Metal
7. Paper
8. Plastic
9. Shoes
10. Trash

## 📈 MLflow Experiment Tracking & Demo

MLflow was integrated into the project to track machine learning experiments and support model management.

Key features:

- Track model training experiments and evaluation metrics.
- Log model parameters and performance results.
- Save and organize experiment artifacts.
- Inspect and compare experiment runs through the MLflow UI.
- Demonstrate the experiment tracking workflow through a recorded demo.

The demo showcases the MLflow interface and recorded experiment results, providing a practical view of the model development and tracking workflow.

## 🔄 Notebook Workflow

### 1. Importing Libraries

Import the required libraries for image processing, data manipulation, visualization, model training, evaluation, and file management.

Key libraries include TensorFlow/Keras, NumPy, Matplotlib, Scikit-learn, Seaborn, and JSON utilities.

### 2. Handling Image Extensions

Handle image file extensions and ensure that supported image files can be loaded by the data pipeline.

### 3. Setup

Configure the notebook environment, dataset paths, image dimensions, batch size, class names, and other model-related variables.

### 4. ETL and Data Preparation

Load and prepare the image dataset, organize the required data splits, and prepare images and labels for training, validation, and testing.

### 5. Data Visualization

Explore the dataset and visualize training behavior to better understand the data and monitor model performance.

### 6. Checking Class Imbalance

Inspect the distribution of images across the waste categories to identify underrepresented classes.

### 7. Class Weights

Apply class weights during model training to give additional importance to underrepresented classes when appropriate.

### 8. Building the Model

Build a transfer learning architecture based on EfficientNetV2S with pretrained ImageNet weights.

The classification head includes:

- Data augmentation using random flipping, rotation, zoom, height, and width transformations.
- A pretrained EfficientNetV2S feature extractor.
- Global Average Pooling.
- A Dense layer with 128 units and ReLU activation.
- A Dropout layer with a rate of 0.2.
- An output Dense layer with 10 units and Softmax activation.

### 9. Training the Model

Train the model on the training dataset and monitor its performance using validation data.

The available training history records **12 epochs**, including training accuracy, validation accuracy, training loss, validation loss, and learning rate.

### 10. Checking Overfitting

Visualize training and validation accuracy and loss together to inspect convergence and identify possible signs of overfitting.

### 11. Testing the Model

Evaluate the trained model on the held-out test dataset to measure its performance on unseen images.

### 12. Model Evaluation

Evaluate classification performance using:

- Test accuracy and test loss.
- Precision, recall, and F1-score for each class.
- Macro and weighted average metrics.
- A confusion matrix to inspect correct predictions and misclassifications.

### 13. Saving the Model and Artifacts

Save the trained model and supporting files for reuse, MLflow integration, and deployment.

### 14. Building the Knowledge Base

Create a structured JSON knowledge base containing recycling information for each category, including:

- Category descriptions.
- Disposal instructions.
- Recycling recommendations.
- Safety precautions.

### 15. Recycling Guide Function

Implement `get_recycling_guide()` to retrieve guidance associated with the model's predicted waste category.

This separates image classification from knowledge retrieval and makes the guidance function reusable in future applications.

### 16. Final Model Prediction

Select a sample image from the test dataset and:

1. Display the image.
2. Show the actual category.
3. Predict the waste category.
4. Display the prediction confidence.
5. Compare the predicted category with the actual label.
6. Retrieve and display the corresponding recycling guide.

This final step demonstrates the complete classification and knowledge retrieval workflow.

## 🌐 Streamlit Web Application

The project is deployed as an interactive web application using **Streamlit**, integrating the trained EfficientNetV2S model with the recycling knowledge base to provide an end-to-end waste classification and recycling guidance experience.

### Key Features

- Upload waste images through a user-friendly interface.
- Classify uploaded images into one of the 10 waste categories.
- Display prediction confidence and the top-3 predicted categories.
- Retrieve category-specific recycling recommendations and safety precautions.
- Present results through an interactive interface with an eco-friendly green theme.
- Integrate deep learning inference with structured knowledge retrieval.

### Live Application

**[Open the EcoSort-RecycleIR Web App](https://ecovision-recycling-2nr25qyqddet7dp7ha8d8e.streamlit.app/)**

### Running the Application Locally

Run the following command from the project directory:

```bash
python -m streamlit run app.py
```

### Application Structure

```text
EcoSort-RecycleIR/
├── app.py
├── .streamlit/
│   └── config.toml
└── ecosort_artifacts/
    ├── EcoSort_EfficientNetV2S.keras
    ├── class_names.json
    └── recycling_knowledge.json
```

- `app.py`: Streamlit interface and prediction workflow.
- `EcoSort_EfficientNetV2S.keras`: Saved classification model.
- `class_names.json`: Class labels used during prediction.
- `recycling_knowledge.json`: Recycling guidance for all 10 categories.
- `config.toml`: Application theme configuration.

## 🧠 Model Information

| Property | Details |
|---|---|
| Model Architecture | EfficientNetV2S |
| Learning Approach | Transfer Learning |
| Pretrained Weights | ImageNet |
| Number of Classes | 10 |
| Dense Layer | 128 units, ReLU |
| Dropout Rate | 0.2 |
| Output Activation | Softmax |
| Recorded Training Epochs | 12 |

## 📊 Final Results

The model was evaluated on **1,216 test images** across the 10 waste categories.

### Overall Performance

| Metric | Score |
|---|---:|
| Test Accuracy | **95.72%** |
| Macro Precision | 95.62% |
| Macro Recall | 95.30% |
| Macro F1-score | 95.43% |
| Weighted Precision | 95.78% |
| Weighted Recall | 95.72% |
| Weighted F1-score | 95.73% |
| Test Images | 1,216 |

### Per-Class Classification Report

| Class | Precision | Recall | F1-score | Support |
|---|---:|---:|---:|---:|
| Battery | 1.0000 | 0.9730 | 0.9863 | 74 |
| Biological | 0.9506 | 0.9625 | 0.9565 | 80 |
| Cardboard | 0.9370 | 0.9520 | 0.9444 | 125 |
| Clothes | 1.0000 | 1.0000 | 1.0000 | 183 |
| Glass | 0.9477 | 0.9295 | 0.9385 | 156 |
| Metal | 0.8981 | 0.9700 | 0.9327 | 100 |
| Paper | 0.9590 | 0.9070 | 0.9323 | 129 |
| Plastic | 0.9286 | 0.9548 | 0.9415 | 177 |
| Shoes | 1.0000 | 0.9928 | 0.9964 | 138 |
| Trash | 0.9412 | 0.8889 | 0.9143 | 54 |
| **Overall Accuracy** | — | — | **0.9572** | **1,216** |

### Dataset Category Distribution

The following image counts were recorded for the dataset:

| Category | Image Count |
|---|---:|
| Battery | 617 |
| Biological | 551 |
| Cardboard | 1,145 |
| Clothes | 1,527 |
| Glass | 1,391 |
| Metal | 721 |
| Paper | 1,077 |
| Plastic | 1,274 |
| Shoes | 1,147 |
| Trash | 358 |
| **Total** | **9,808** |

*These counts are reported separately from the test support counts. Confirm the dataset split they represent before interpreting them as training-only counts.*

### Results Interpretation

The model achieved **95.72% test accuracy** and a weighted F1-score of **95.73%**, indicating strong overall performance on the test dataset.

- **Clothes** achieved an F1-score of 1.0000 across 183 test images.
- **Shoes** achieved an F1-score of 0.9964.
- **Battery** achieved an F1-score of 0.9863.
- **Trash** had the lowest recall at 0.8889, suggesting that some trash images were classified as other categories.
- **Metal** had a recall of 0.9700 but a precision of 0.8981, suggesting some false-positive metal predictions.
- **Paper** had a recall of 0.9070, making it another category worth investigating for potential misclassifications.

The confusion matrix can help identify specific class pairs that the model frequently confuses.

*Test loss can be added when the numerical output from the model's evaluation step is available.*

## 💾 Saved Artifacts

The notebook saves its output files under:

`/kaggle/working/ecosort_artifacts/`

| File | Purpose |
|---|---|
| `EcoSort_EfficientNetV2S.keras` | Saved trained classification model |
| `class_names.json` | Class names used during prediction |
| `training_history.json` | Training and validation history |
| `recycling_knowledge.json` | Recycling information for all categories |

These artifacts support inference, MLflow experiment tracking, and deployment workflows.

## 🛠️ Technologies Used

- **Language:** Python
- **Deep Learning:** TensorFlow / Keras
- **Model Architecture:** EfficientNetV2S
- **Data Processing:** NumPy
- **Visualization:** Matplotlib, Seaborn
- **Evaluation:** Scikit-learn
- **Knowledge Storage:** JSON
- **Experiment Tracking:** MLflow
- **Web Application:** Streamlit
- **Development Environment:** Kaggle Notebooks, Local Python Environment

## 🚀 Future Improvements

- Expand the knowledge base with verified, location-specific recycling rules.
- Investigate misclassifications using the confusion matrix.
- Improve generalization through additional data and model tuning.
- Track experiments and model versions using MLflow.
- Explore API-based deployment for integration with other applications.
- Improve the application's usability and accessibility based on user feedback.

