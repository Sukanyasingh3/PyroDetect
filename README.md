<div align='center'>
 
# PyroDetect: Fire Detection for CCTV
![Screenshot (184)](https://github.com/Sukanyasingh3/PyroDetect/assets/113462236/7f80da27-3482-419a-98ab-60a34b4c17ae)

### PyroDetect is a comprehensive project featuring a Convolutional Neural Network (CNN) model tailored for fire detection in CCTV footage. Crafted with TensorFlow and Keras, this innovative solution efficiently analyzes image data to classify the presence or absence of fire. Leveraging deep learning techniques, the model enhances surveillance systems by providing real-time identification of potential fire incidents. Its robust design ensures accuracy and reliability in discerning critical situations, making it a valuable addition to security infrastructure. This repository serves as a resource for implementing cutting-edge fire detection capabilities, contributing to enhanced safety measures in diverse environments.
</div>
 
# Dataset
The dataset used for training and evaluating the fire detection model can be found on Kaggle at the following link:
##   [Fire Detection Dataset](https://www.kaggle.com/datasets/tharakan684/urecamain) 


The Fire Detection Dataset is a comprehensive collection of images meticulously labeled as either containing instances of fire or being devoid of any fire-related elements. This binary classification is crucial for training the PyroDetect model to accurately discern the presence or absence of fire in diverse scenarios.

The dataset is designed to encompass a wide range of scenarios and environmental conditions. Images within the dataset capture varying lighting conditions, perspectives, and contexts, ensuring that the PyroDetect model is exposed to the diversity it may encounter in real-world surveillance footage.
# Libraries Used
The project utilizes the following libraries: 

 - TensorFlow 

 - Keras 

 - NumPy 

 - OpenCV (cv2) 

 - Matplotlib

## Installation

To use this project, follow these steps:

1. Clone the repository:

```bash
   git clone https://github.com/Sukanyasingh3/PyroDetect.git
   ```
## Usage

1. Navigate to the project directory:

```bash
   cd PyroDetect
   ```
2. Install the required dependencies:
  ```bash
   python app.py
```
This will start the plant disease diagnosis application.


# Model Architecture
The fire detection model is built using a Convolutional Neural Network (CNN) architecture. The sequential model is constructed with layers such as Conv2D, MaxPooling2D, AveragePooling2D, Flatten, Dense, and BatchNormalization.

To train the model, run:
  ```bash
   python train_model.py
```

# Results
After training, the model achieves an accuracy of 92% on the train set and 76% on the test set.

## Acuracy:
![Accuracy](https://github.com/Sukanyasingh3/Fire-Detection-for-CCTV/assets/113462236/3b4376cf-aa98-408b-8f14-b060bdae6872)

## Loss:
![Loss](https://github.com/Sukanyasingh3/Fire-Detection-for-CCTV/assets/113462236/e207a52d-20a6-4858-a98f-fb31b43702e8)

# Contributing
![gif3](https://github.com/Sukanyasingh3/Fire-Detection-for-CCTV/assets/113462236/6f4ebeb3-cc50-4e33-babc-32f9f6012709)



If you would like to contribute to the project, follow these steps:

 - Fork the repository.
 - Create a new branch for your feature or bug fix.
 - Make your changes and submit a pull request.
