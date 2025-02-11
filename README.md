# Breast Cancer Detection: ML Classifiers vs. CNNs

## Overview
This project compares traditional machine learning classifiers with convolutional neural networks (CNNs) for breast cancer detection. The goal is to evaluate the effectiveness of different approaches in classifying breast cancer images. We analyze performance metrics to determine the most suitable model for this task.

## Dataset
The dataset used for this study consists of ultrasound images from Baheya Hospital. It contains labeled samples indicating whether a tumor is (benign or malignant) or Normal. The dataset is preprocessed before training to ensure optimal performance.

## Models Compared
We evaluate the following models:

### Machine Learning Classifiers:
1. **Naive Bayes**
2. **Decision Tree**
3. **k-Nearest Neighbors (k-NN)**
4. **Support Vector Machine (SVM)**

### Convolutional Neural Networks (CNNs):
1. **LeNet-5** - A classic CNN architecture designed for digit recognition, adapted for medical image classification.
2. **AlexNet** - A deep CNN known for its superior performance in image classification tasks.
3. **Custom CNN Model** - A CNN designed from scratch with optimized layers and hyperparameters for breast cancer detection.

## Methodology
1. **Data Preprocessing:**
   - Image resizing, normalization, and augmentation.
   - Splitting into training, validation, and test sets.
   
2. **Feature Extraction (for ML Classifiers):**
   - Using hand-crafted features such as texture, color, and shape descriptors.
   - Normalization.

3. **Model Training & Evaluation:**
   - Training classifiers on extracted features.
   - Training CNN models directly on raw images.
   - Evaluating models using metrics like accuracy, precision and recall.

## Results and Comparison
- **ML Classifiers:** Traditional classifiers perform well with carefully extracted features but may struggle with complex image representations.
- **CNNs:** Deep learning models, particularly AlexNet and the custom CNN, achieve higher accuracy due to automatic feature extraction.
- **Performance Metrics:** The CNN models outperform traditional classifiers in most cases, but computational cost and training time are higher.

## Web Application
After identifying the best-performing model, a web application was developed using Flask to allow users to upload ultrasound images and receive breast cancer predictions in real-time.


### Features:
- User-friendly interface for uploading images.
- Model inference and classification results displayed instantly.
- Deployment-ready setup for integration with healthcare systems.

### Important Note on Model File
Due to GitHub's file size limit, the best model (best_model.h5) is not included in the repository. Users who wish to use the application must either train the model using the provided scripts (Breast_Cancer_Detection.ipynb) or download the pre-trained model from an external source (to be specified in the repository).

## Conclusion
- CNN-based models provide superior performance in breast cancer detection compared to traditional ML classifiers.
- Feature extraction plays a crucial role in ML classifier performance.
- The trade-off between interpretability and accuracy should be considered when choosing a model.

## Future Work
- Experimenting with more advanced CNN architectures (e.g., ResNet, EfficientNet).
- Implementing explainability techniques (e.g., Grad-CAM) to interpret CNN decisions.
- Exploring transfer learning for improved performance with limited data.
- Enhancing the web application with additional functionalities like patient history tracking and AI-assisted diagnosis explanations.

## References
- https://www.sciencedirect.com/science/article/pii/S2352340919312181 
- https://www.sciencedirect.com/science/article/pii/S2405959520300801 
- https://www.sciencedirect.com/science/article/pii/S1877050921014629
- https://www.hindawi.com/journals/jhe/2021/5528622/
- https://www.semanticscholar.org/paper/Breast-Cancer-Detection-using-DeepConvolutional-Mechria-Gouider/f690703f3102b239f5077ee24bb7c385cae1670e
- https://machinelearningmastery.com/k-fold-cross-validation/
- https://www.analyticsvidhya.com/blog/2021/03/the-architecture-of-lenet-5/
- https://www.geeksforgeeks.org/ml-getting-started-with-alexnet/
