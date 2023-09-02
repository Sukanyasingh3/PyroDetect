# Fire-Detection-for-CCTV
 CNN model for fire detection using CCTV footage, built using TensorFlow and Keras, and is designed to classify whether a given image footage contains fire or not
 
# Dataset
The dataset used for training and evaluating the fire detection model can be found on Kaggle at the following link: [Fire Detection Dataset](https://www.kaggle.com/datasets/tharakan684/urecamain). The dataset consists of images labeled as either containing fire or not containing fire.

# Libraries Used
The project utilizes the following libraries:

TensorFlow

Keras

NumPy

OpenCV (cv2)

Matplotlib

# Model Architecture
The fire detection model is built using a Convolutional Neural Network (CNN) architecture. The sequential model is constructed with layers such as Conv2D, MaxPooling2D, AveragePooling2D, Flatten, Dense, and BatchNormalization.

# Results
After training, the model achieves an accuracy of 92% on the train set and 76% on the test set.
