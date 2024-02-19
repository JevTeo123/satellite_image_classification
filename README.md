<details>
  <summary>Table Of Contents</summary>
  
  1. [About The Project](#about-the-project) 
        - [Built With](#built-with)
  2. [Getting Started](#getting-started)
        - [Data Normalization](#data-normalization)
        - [Train Test Split](#train-test-split)
  4. [Preprocess Data](#preprocess-data)
  5. [Building Deep Leearning Model](#building-deep-learning-model)
        - [What is CNN and how does it work?](#what-is-cnn-and-how-does-it-work)
        - [Implementation of CNN in our project](#implementation-of-cnn-in-our-project)
</details>

# About The Project
The purpose of this project is to classify images taken from a satellite into 4 respective categories (cloudy, desert, green_area and water), through the use of a convolutional Neural Network (CNN).

### Built With
<img src = "https://github.com/JevTeo123/satellite_image_classification/assets/123255675/dea8b062-0d3c-4570-821c-927b631beefd" width = "200"><br>
<img src = "https://github.com/JevTeo123/satellite_image_classification/assets/123255675/4232b983-6f1a-4e81-bd42-46308b39009b" width="200"><br>
<img src = "https://github.com/JevTeo123/satellite_image_classification/assets/123255675/d0883df7-26f5-4685-b4bc-a275c1d64d47" width="200"><br>
<img src = "https://github.com/JevTeo123/satellite_image_classification/assets/123255675/76f4c7e2-26f9-489a-9cdb-cac182552bd8" width="200"><br>


# Getting Started
Before getting started on the creation of the CNN model, we first have to load the images that are provided for us into a format suitable for the model to make predictions on.

Firstly, data has to be loaded from directory that contains all of our images. By using the code below, we are able to load the images from our data folder into a ```tf.data.Dataset``` format.
```py 
data = tf.keras.utils.image_dataset_from_directory("data")
```
# Preprocess Data
## Data Normalizaton
To preprocess the data, the ```py .numpy_iterator``` is used to convert the images into a numpy array format such that we can perform scaling on the data and scale the images into values in the range of 0 and 1, since the pixel values of images ranges fro 0 to 255 corresponding to the RGB channels of an image.
```py
data = data.map(lambda x, y: (x/ 255, y))
scaled_iterator = data.as_numpy_iterator()
scaled_iterator.next()[0].max()
```

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
After Preprocessing of data is done, it is time to build the convulational neural network for image classification. 

## What is CNN and how does it work?
CNNs is a feed forward neural network as the information moves from one layer to the next.
Layers in CNN:
1. Convulational Layer (Input Layer)
2. ReLu Layer (Hidden Layer)
3. Pooling Layer (Hidden Layer)
4. Normnalization Layer (Hidden Layer)
5. Fully Connected Layer aka. Dense Layer (Hidden Layer)
6. Sigmoid/ Softmax Layer (Output Layer)

### Convulational Layer
Extracts features from an input image. Uses a matrix filter to go along the pixels of an image to extract out any patterns that may be available in an image. Convolution is a mathematical operation that happens between two matrices to form a third matrix as an output aka convoluted matrix.

Arguments:
  - Filters: The number of feature detectors
  - Kernel_size: The shape of the feature detector
  - Strides: Controls how many units the filter would shift
  - Input Shape: Standardizes the shape of the image into the neural network
  - Padding: Used to control the dimensionality of the convolved feature with respect to the input filters
Filters
  - Helps in finding out edges, curves and details like height width etc.
  - Slides over images to extract different components or patterns of an image
  - First filters learns to extract ou simple features in initial convulated layers however, it learns to extract out significantly more complex features later on.
  - Rotate this filter over an input matrix and get an output of lesser dimension

### Padding
After a filter is applied, it results in an output of lesser dimension. This may be a problem it may lead to information loss from the edges and corners of the images. To preserve the information from the edges and corners, padding can be used.
Types of Padding:
- Zero Padding: Pad the images with zeros so that information at the edges and corners are not lost
- Valid Padding: Drop the part of the image where the filter does not fit, however, this is dangerous as we might lose valuable data from the images

### ReLu Layer:
- Increase the complexity of the neural network by introducing non-linearity into the ConvNets.
- Performs element-wise operation and set negative pixels to zero.
### Pooling Layer:
- Added after the convulational layer.
- Output of convulational layer acts as input for pooling layer
- Pooling down samples the data reducing dimensionality by retaining important information
- Does further feature extraction and detects multiple components of images like edges and corners

### Fully Connected Layer (Dense Layer)
- Connect every neuron in one layer to all the neurons in the output layer.

## Implementation of CNN in our project
```py
model.add(Conv2D(16, (3, 3), 1, activation = "relu", input_shape = (256, 256, 3)))
model.add(MaxPooling2D())

model.add(Conv2D(32, (3, 3), 1, activation = "relu"))
model.add(MaxPooling2D())

model.add(Conv2D(64, (3, 3), 1, activation = "relu"))
model.add(MaxPooling2D())

model.add(Flatten())

model.add(Dense(256, activation = "relu"))
model.add(Dense(4, activation = "softmax")) (# softmax activation for multiclass)
model.compile("adam", loss = "sparse_categorical_crossentropy", metrics = ["accuracy"]) (#sparse_categorical crossentropy for multiclass classification)
```
Final Accuracy for our CNN:
| Train Accuracy | Validation Accuracy |
| :---: | :---: |
| 0.945 | 0.955 |






