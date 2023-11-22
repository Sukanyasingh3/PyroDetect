# Fire Detection for CCTV

Fire Detection for CCTV is a comprehensive project featuring a Convolutional Neural Network (CNN) model tailored for fire detection in CCTV footage. Crafted with TensorFlow and Keras, this innovative solution efficiently analyzes image data to classify the presence or absence of fire. Leveraging deep learning techniques, the model enhances surveillance systems by providing real-time identification of potential fire incidents. Its robust design ensures accuracy and reliability in discerning critical situations, making it a valuable addition to security infrastructure. This repository serves as a resource for implementing cutting-edge fire detection capabilities, contributing to enhanced safety measures in diverse environments.

 
# Dataset
The dataset used for training and evaluating the fire detection model can be found on Kaggle at the following link: [Fire Detection Dataset](https://www.kaggle.com/datasets/tharakan684/urecamain). The dataset consists of images labeled as either containing fire or not containing fire.

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
   git clone https://github.com/Sukanyasingh3/Fire-Detection-for-CCTV.git
   ```
2. Install the required dependencies:
  ```bash
   pip install -r requirements.txt
```
## Usage

1. Navigate to the project directory:

```bash
   cd Fire-Detection-for-CCTV
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

<img src="https://github.com/Sukanyasingh3/Sukanyasingh3/blob/main/gif2.gif" />

If you would like to contribute to the project, follow these steps:

 - Fork the repository.
 - Create a new branch for your feature or bug fix.
 - Make your changes and submit a pull request.
