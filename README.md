<details>
  <summary>Table Of Contents</summary>
  
  1. [About The Project](#about-the-project) 
        - [Built With](#built-with)
  2. [Getting Started](#getting-started)
  3. [Preprocess Data](#preprocess-data)
  4. [Build a Convolutional Neural Network for image classification](#build-cnn-model)
</details>

# About The Project
The purpose of this project is to classify images taken from a satellite into 4 respective categories (cloudy, desert, green_area and water), through the use of a convolutional Neural Network (CNN).

### Built With
<img src="https://sgx-1-asia-southeast1.prod.fire.glass/resources?rid=75c07dd05e09c53f2c7e4ddb7eae3b5f560382475403cd98f5a6e8a0415f6770&amp;url=data%3A97934ea858a5cb4e72c0c2a30462f1a61ced12ab&amp;cid=__FGL__16813121b163bf6bec691f2cb14212e5050875f60000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000&amp;bdk=cookiesEncryptionDisabled&amp;eid=24" alt="IT12A01: FUNDAMENTALS OF PYTHON PROGRAMMING (SF) (SYNCHRONOUS E-LEARNING) -  NTUC LearningHub" width = "200" />![image](https://github.com/JevTeo123/satellite_image_classification/assets/123255675/a5f637b3-1058-4167-a906-2352509ac77f)<br>
<img src="https://sgx-1-asia-southeast1.prod.fire.glass/resources?rid=153b8b497aced89e07fe6518df8c4859b75ca7bd9d32f192b366a881c2dac510&amp;url=data%3A59c705a57f1f2b61682f3b6f6e02cfa5affb5a19&amp;cid=__FGL__16813121b163bf6bec691f2cb14212e5050875f60000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000&amp;bdk=cookiesEncryptionDisabled&amp;eid=24" alt="TensorFlow 1.0 vs 2.0, Part 3: tf.keras | by Yusup | AI³ | Theory,  Practice, Business | Medium" width = "200"/>![image](https://github.com/JevTeo123/satellite_image_classification/assets/123255675/a5d7ebb2-0bcc-44dd-81e0-3ace7b6e93c9) <br>
<img src="https://sgx-1-asia-southeast1.prod.fire.glass/resources?rid=1e02864a0cae24b159b69acb3d6c76c12b4fdcd7c3dca78b643d69665d4ce88b&amp;url=data%3A7c4df48df980ef7d66b5f2b07b3ca6c867f1307b&amp;cid=__FGL__16813121b163bf6bec691f2cb14212e5050875f60000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000&amp;bdk=cookiesEncryptionDisabled&amp;eid=24" alt="pandas (software) - Wikipedia" width = "200"/>![image](https://github.com/JevTeo123/satellite_image_classification/assets/123255675/e70eb94d-126e-44e8-9b36-9055116e6ea1) <br>
<img src="https://sgx-1-asia-southeast1.prod.fire.glass/resources?rid=7cac74fa2ea50755e752878c16019ebc5b31fb90cc6e7c12840c76264a1cc4ee&amp;url=data%3A3280b18c25b165489dd6343aa795256e43e47b0a&amp;cid=__FGL__16813121b163bf6bec691f2cb14212e5050875f60000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000&amp;bdk=cookiesEncryptionDisabled&amp;eid=24" alt="Important Libraries of OpenCV. OpenCV is a cross-platform library used… |  by Prithvi Dev | Javarevisited | Medium" width = "200"/>![image](https://github.com/JevTeo123/satellite_image_classification/assets/123255675/a8eecc2f-ce22-4bda-93b3-199215386167)<br>

# Getting Started
Before getting started on the creation of the CNN model, we first have to load the images that are provided for us into a format suitable for the model to make predictions on.

Firstly, data has to be loaded from directory that contains all of our images. By using the code below, we are able to load the images from our data folder into a ```py tf.data.Dataset``` format.
```py 
data = tf.keras.utils.image_dataset_from_directory("data")```
```
# Preprocess Data
## Data Normalizaton
To preprocess the data, the ```py .numpy_iterator``` is used to convert the images into a numpy array format such that we can perform scaling on the data and scale the images into values in the range of 0 and 1, since the pixel values of images ranges fro 0 to 255 corresponding to the RGB channels of an image.
```py
data = data.map(lambda x, y: (x/ 255, y))
scaled_iterator = data.as_numpy_iterator()
scaled_iterator.next()[0].max()

## Train Test Split
After Data Normalization, train test split is done to split the data into training, testing and validation data. The train data would be used to train the model on the data to predict the different categories of the target variable, the testing data would be used to test the model's accuracy on unseen data while the validation data is used to provide and unbiased evaluation of the model's performance and to fine tune the model's parameters. This is done through the code below:
```py

train_size = int(len(data) * .7) (# 70 percent for training)
val_size = int(len(data) *.2) + 1 (# 20 percent for validation)
test_size = int(len(data) *.1) + 1 (# 10 percent for testing)
train = data.take(train_size)
test = data.take(test_size)
val = data.take(val_size)

# Building Deep Learning Model







