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
![image](https://github.com/JevTeo123/satellite_image_classification/assets/123255675/dea8b062-0d3c-4570-821c-927b631beefd){: width="200"}
![image](https://github.com/JevTeo123/satellite_image_classification/assets/123255675/4232b983-6f1a-4e81-bd42-46308b39009b){: width="200"}
![image](https://github.com/JevTeo123/satellite_image_classification/assets/123255675/d0883df7-26f5-4685-b4bc-a275c1d64d47){: width="200"}
![image](https://github.com/JevTeo123/satellite_image_classification/assets/123255675/76f4c7e2-26f9-489a-9cdb-cac182552bd8){: width="200"}


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
```

# Building Deep Learning Model







