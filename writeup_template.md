#**Traffic Sign Recognition** 

---

**Build a Traffic Sign Recognition Project**

The goals / steps of this project are the following:
* Load the data set 
* Explore, summarize and visualize the data set
* Design, train and test a model architecture
* Use the model to make predictions on new images
* Analyze the softmax probabilities of the new images
* Summarize the results with a written report


[//]: # (Image References)

[image1]: ./examples/visualization.jpg "Visualization"
[image2]: ./examples/grayscale.jpg "Grayscaling"
[image3]: ./examples/random_noise.jpg "Random Noise"
[image4]: ./examples/placeholder.png "Traffic Sign 1"
[image5]: ./examples/placeholder.png "Traffic Sign 2"
[image6]: ./examples/placeholder.png "Traffic Sign 3"
[image7]: ./examples/placeholder.png "Traffic Sign 4"
[image8]: ./examples/placeholder.png "Traffic Sign 5"

## Rubric Points

###Data Set Summary & Exploration

####1. Dataset Loading and summary

The code for this step is contained in the 1st and 2nd code cell of the IPython notebook.  

I used the pandas and numpy library to calculate summary statistics of the traffic
signs data set:

* The size of training set is 34799
* The size of validation set is 4410
* The size of test set is 12630
* The shape of a traffic sign image is (32,32,3)
* The number of unique classes/labels in the data set is 43

####2. Dataset Exploration

The code for this step is contained in the 3rd and 4th and 5th code cell of the IPython notebook.  

* Added validation dataset into training dataset to form one bigger training dataset. 
* Later on model is dividing this combined dataset into training and validation dataset randomly using 25-75% rule.
* After combining the training and validation dataset code is showing the image from both original training dataset and combined dataset with their corresponding code. This ensures that images are displayed correcty with right code. It also ensures that combining training and validation dataset has not corrupted the images and corresponding lables.


###Design and Test a Model Architecture

####1. Preprocssing of images

The code for this step is contained in the 3rd, 4th and 5th code cell of the IPython notebook.

* Training and validation datasets are combined to form one large training dataset.
* Training images are in color. But for our model color doesn't matter. So model is converting all images to grayscale.
* After converting training dataset images to grayscale, code is normalizing the images. This is needed so that all training vector values can be distributed between specified values and that helps in finding better maxima and faster learning.
* 10 image output is shown in output of Cell 4 in Traffic_Sign_Classifier.ipynb after preprocssing to display before and after preprocessed images.


####2. Dividing data into training and validation dataset.

The code for splitting the data into training and validation sets is contained in the fifth code cell of the IPython notebook.  

To cross validate my model, I randomly split the training data into a training set and validation set. I did this by shuffing and then splitting the total training dataset in to training and validation set by using 75-25% rule.

My final training set had 29407 number of images. My validation set and test set had 9802 and 12630(unchanged) number of images.

I didn't opt for augmenting the dataset further since I was able to get about 98% accuracy with my preprocessing. 



####3. Architecture of model

The code for my final model is located in the seventh cell of the ipython notebook. 

Input
The LeNet architecture accepts a 32x32x3 image as input. 

Architecture
Layer 1: Convolutional. The output shape should be 28x28x6.

Activation. RELU activation function.

Pooling. The output shape should be 14x14x6.

Layer 2: Convolutional. The output shape should be 10x10x16.

Activation. RELU activation function.

Pooling. The output shape should be 5x5x16.

Flatten. Flatten the output shape of the final pooling layer such that it's 1D instead of 3D.

Layer 3: Fully Connected. This should have 120 outputs.

Activation. Your choice of activation function.

Layer 4: Fully Connected. This should have 84 outputs.

Activation. RELU activation function.

Layer 5: Fully Connected (Logits). This should have 43 outputs.

Output

Return the result of the 2nd fully connected layer.



####4. Training the model.


The code for training the model is located in the 8-11 cells of the ipython notebook. 

To train the model, I used an below parameters

EPOCH- 60

Optimizer - AdamOptimizer

Learning rate- 0.001

Batch Size - 128

####5. Approch for the soultion

The code for calculating the accuracy of the model is located in the 10-11 cell of the Ipython notebook.

My final model results were:
* validation set accuracy of 0.987
* test set accuracy of o.925

I first started with wellknown LENET model. I started with this model since it was recommeded by Udacity and also it is well known model that works very well with MNIST. 

I got around 0.86 accuracy on validation dataset by using this model.

To increase the accuracy I first converted images to grayscale since color is not important factor and also LENET is based on grayscale images. By doing this I got around 0.9 accuracy.

Then I preprocessed images with minmax normalizer. This helped me to increase accuracty to about 0.94

Then I tried combined training and validation sets and then shuffing and dividing with 75-25% rule. This gave me about 0.93 accuracy.

Then I increased number of EPOCH to 60 and it gave me 0.987 accuracy.

After this, I tried this on training dataset and it gave me 0.925 accuracy.  

###Test a Model on New Images

####1. Five image testing

Six German traffic sign is displyed in output of 11th cell of the Ipython notebook


####2. 

The code for making predictions on my final model is located in the tenth cell of the Ipython notebook.

Here are the results of the prediction:

| Image			        |     Prediction	        					| 
|:---------------------:|:---------------------------------------------:| 
| Traffic signal     		| Traffic Signal   									| 
| Right Turn   			| Right Turn										|
| Pedestrians					| Pedestrians											|
| Straight Ahead	      		| Straight Ahead					 				|
| Children crossing			| Children crossing     							|
| Road work			| Road work     							|



The model was able to correctly guess 6 of the 6 traffic signs, which gives an accuracy of 100%. This compares favorably to the accuracy on the test set of 92.4%

####3. Describe how certain the model is when predicting on each of the five new images by looking at the softmax probabilities for each prediction and identify where in your code softmax probabilities were outputted. Provide the top 5 softmax probabilities for each image along with the sign type of each probability. (OPTIONAL: as described in the "Stand Out Suggestions" part of the rubric, visualizations can also be provided such as bar charts)

The code for making predictions on my final model is located in the 11th cell of the Ipython notebook.

For all 6 images model is quite sure about it's prediction. This is surprizing to me as well. I think it is probably because I choose very nice images from the web comapred to test sets.

1st Image

| Probability         	|     Prediction	        					| 
|:---------------------:|:---------------------------------------------:| 
| 1.0        			| Traffic Signal   									| 
| less than 0.001    				| General caution 										|
| less than 0.001					| Speed limit (30km/h)											|
| less than 0.001	      			| Keep right					 				|
| less than 0.001				    | Turn left ahead      							|


2nd Image

| Probability         	|     Prediction	        					| 
|:---------------------:|:---------------------------------------------:| 
| 1.0         			| Right Turn   									| 
| less than 0.001    				| Speed limit (30km/h) 										|
| less than 0.001					| Roundabout mandatory											|
| less than 0.001	      			| Priority road					 				|
| less than 0.001				    | Speed limit (70km/h)     							| 

3rd Image

| Probability         	|     Prediction	        					| 
|:---------------------:|:---------------------------------------------:| 
| .99         			| Pedestrians   									| 
| less than 0.001    				| Right-of-way at the next intersection 										|
| less than 0.001					| Children crossing											|
| less than 0.001	      			| General caution				 				|
| less than 0.001				    | Dangerous curve to the right     							| 

4th Image

| Probability         	|     Prediction	        					| 
|:---------------------:|:---------------------------------------------:| 
| 1.0        			| Straight Ahead   									| 
| less than 0.001    				| Speed limit (60km/h) 										|
| less than 0.001					| No passing										|
| less than 0.001	      			| Speed limit (20km/h)				 				|
| less than 0.001				    | Speed limit (30km/h)     							| 

5th Image

| Probability         	|     Prediction	        					| 
|:---------------------:|:---------------------------------------------:| 
| 1.0        			| Children crossing   									| 
| less than 0.001    				| Right-of-way at the next intersection 										|
| less than 0.001					| Beware of ice/snow									|
| less than 0.001	      			| Dangerous curve to the right				 				|
| less than 0.001				    | Dangerous curve to the left   							| 

6th Image

| Probability         	|     Prediction	        					| 
|:---------------------:|:---------------------------------------------:| 
| 1.0        			| Road work     									| 
| less than 0.001    				| Dangerous curve to the right 										|
| less than 0.001					| Speed limit (20km/h)										|
| less than 0.001	      			| Speed limit (30km/h)				 				|
| less than 0.001				    | Speed limit (50km/h)     							| 
