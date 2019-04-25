[ Follow Me ]

In this project we train a Fully Convolutional Network (FCN) to be able to identify a target in a simulated drone environment. A FCN is similar to a Convolutional Neural Network (CNN) but adds Skip Connections and the encode/decode step.

[ The Main Libraries ] 

	TensorFlow is an open source deep learning library for numerical computation using data flow graphs. The graph nodes represent mathematical operations, while the graph edges represent the multidimensional data arrays (tensors) that flow between them. This flexible architecture enables you to deploy computation to one or more CPUs or GPUs in a desktop, server, or mobile device without rewriting code. 
	Keras is a high level deep learning API that allows you to focus on network architecture rather than on smaller details building up to the network. Keras has recently been included in the TensorFlow library. 


[ The Data Set ]

01- Training data 

	   train.zip: training set images: 4131 RGB images
	   masks: 4131 mask images Target: Blue; Pedestrian: Green; Background: Red

02- Validation data
 
	 validation.zip: validation set images: 1184 RGB images
	 masks: 1184 mask images
 
	-Measures how well the network is trained [ avoids overfitting!!! ]

03- Test Images 

	following_images: 542 RGB images
	masks: 542 mask images

04- data/weights directory for running on simulation:

	2 files:

	1- configuration_weights file. 
	2- model_weights



[ Network Architecture ]

Fully Convolutional Encoder-Decoder neural network, these are the FCN layers:

Build the Model

	1- Encoder block
		a- separable convolutions
		b- batch normalization
	2- Decoder block
  		a- bilinear upsampling
		b- layer concatenation
		c- additional separable convolution layers
	3- Fully Convolutional Network


1- Encoder Block

a- Separable Convolutions

	convolution performed over each channel  
	
	different than regular convolutions bc reduction in the number of parameters which improves runtime 	performance 

 	reducing overfitting because fewer parameters

Coding Seperable Convolutions

An optimized version of separable convolutions = provided in the utils module of the provided repo. 
implemented as follows:

	output = SeparableConv2DKeras(filters, kernel_size, strides,
                             	padding, activation)(input)

	input = input layer,
	filters = number of output filters (the depth),
	kernel_size = number that specifies the (width, height) of the kernel,
	padding =  "same" or "valid"
	activation = activation function [ ex. “relu” ]




b- Batch Normalization

 	instead of just normalizing the inputs to the network ----> normalize the inputs to layers within the network. 
  	during “ training” we use the mean and variance of the values in the current mini-batch.
	A network is a series of layers, where the output of one layer becomes the input to another. 
	Benefits of Batch Normalization

		Networks train faster
		Allows higher learning rates
Simplifies the creation of deeper networks
a bit of regularization

	Coding Batch Normalization


In tf.contrib.keras, batch normalization can be implemented with the following function definition:

	from tensorflow.contrib.keras.python.keras import layers

	output = layers.BatchNormalization()(input) 


	Encoder Block Process 

	Create an encoder block that includes a separable convolution layer using the 	separable_conv2d_batchnorm()  function.

	separable_conv2d_batchnorm() function adds a batch normalization layer after the separable convolution 	layer
 
	The filters parameter defines the size or depth of the output layer = 32 or 64. 

def encoder_block(input_layer, filters, strides):
    
    # TODO Create a separable convolution layer using the separable_conv2d_batchnorm() function.
    output_layer = separable_conv2d_batchnorm(input_layer, filters, strides)
    return output_layer


















2- Decoder Block

a- bilinear upsampling

Bilinear upsampling is a resampling technique that utilizes the weighted average of four nearest known pixels, located diagonally to a given pixel.

The weighted average is usually distance dependent

	Coding Bilinear Upsampler
An optimized version of a bilinear upsampler has been provided --->  utils module of the provided repo 

	implemented as follows:

	output = BilinearUpSampling2D(row, col)(input)

	input = input layer,
	row = upsampling factor for the rows of the output layer
	col = upsampling factor for the columns of the output layer
	output = output layer.


b- layer concatenation step. 

	Concatenating two layers, the upsampled layer and a layer with more spatial information than the upsampled one, presents us with the same functionality. 

	implemented as follows:

	from tensorflow.contrib.keras.python.keras import layers
	output = layers.concatenate(inputs)

	inputs = list of the layers that you are concatenating.

	output = layers.concatenate([input_layer_1, input_layer_2])	

	This step is similar to skip connections. will concatenate the small_ip_layer and the 	large_ip_layer.


c- Some (one or two) additional separable convolution layers to extract some more spatial information from prior layers.


def decoder_block(small_ip_layer, large_ip_layer, filters):
    
    # TODO Upsample the small input layer using the bilinear_upsample() function.
    upsampled_layer = bilinear_upsample(small_ip_layer)
    # TODO Concatenate the upsampled and large input layers using layers.concatenate
    output_layer = layers.concatenate([upsampled_layer, large_ip_layer])
    # TODO Add some number of separable convolution layers
    output_layer = separable_conv2d_batchnorm(output_layer, filters)
    return output_layer
3- Fully convolutional Network used for identification in the follow me code:

I created 2 encoder/decoder levels because I was able to tweak the network paymasters enough to achieve a high enough final score. The FCN model used for the project contains two encoder block layers, a 1x1 convolution layer, two decoder block layers, and filter depths between 32 and 64.

1.  Instead of fully connected layers We use 1x1 convolution layer

2. Up-sampling is done by using bi-linear up-sampling,which help in up sampling the previous layer to a desired resolution or dimension and help speed up performance. 

3.  Skip connections, which allows to use information from multiple resolution scales
from the encoder to decoder process


Encoder block layers

	1st convolution encoder layer

	filter size = 32  
	stride = 2 
	padding = 'same'
 
	2nd convolution  encoder layer

	filter size = 64  
	stride = 2 
	padding = 'same' 

	The padding and the stride of 2 cause each layer to halve the image size, while increasing the 	depth to match the filter size used

1x1 convolution layer

	filter size = 128, with the standard kernel and 
	stride = 1.

Decoder block layers

	First decoder block layer
	small input layer = output from the 1x1 convolution as the 
	large input layer = the first convolution layer,  mimicking a skip connection
	filter size  =  64
 
	second decoder block layer
	small input layer = output from the first decoder block  
	large input layer = original image  mimicking a skip connection better  
	filter size = 32.
The output convolution layer applies a softmax activation function to the output of the second decoder block.

FCN code here: 

def fcn_model(inputs, num_classes):
    
    # TODO Add Encoder Blocks.  
    output_layer = encoder_block(inputs, filters = 32, 2)
    ol_2 = encoder_block(output_layer, filters = 64, 2)
    
    # TODO Add 1x1 Convolution layer using conv2d_batchnorm().
    small_ip_layer = conv2d_batchnorm(ol_2, 128, kernel_size=1, strides=1)
    
    # TODO: Add the same number of Decoder Blocks as the number of Encoder Blocks
    x_almost = decoder_block(small_ip_layer, output_layer, filters = 64)
    x = decoder_block(x_almost, inputs ,filters = 32)
    
    # The function returns the output layer of your model. "x" is the final layer obtained from the last                                                                	decoder_block()

    return layers.Conv2D(num_classes, 1, activation='softmax', padding='same')(x)


FCN image:


















Hyperparameters [ Tuning for Performance ]

batch_size: number of training samples/images that get propagated through the network in a single 		        pass.

num_epochs: number of times the entire training dataset gets propagated through the network.

steps_per_epoch: number of batches of training images that go through the network in 1 epoch. We 			       have provided you with a default value. One recommended value to try would be          		       based on the total number of images in training dataset divided by the batch_size.

validation_steps: number of batches of validation images that go through the network in 1 epoch. This 		      is similar to steps_per_epoch, except validation_steps is for the validation dataset.
 
workers: maximum number of processes to spin up. This can affect your training speed and is 			    dependent on your hardware. 



Test_01
 
learning_rate = 0.004
batch_size = 128
num_epochs = 20
steps_per_epoch = 100
validation_steps = 50
workers = 2
Time = 196 s per epoch 
final score = 0.390761421554







Test_02
 
learning_rate = 0.004
batch_size = 64
num_epochs = 35
steps_per_epoch = 50
validation_steps = 50
workers = 8
Time = 47 s per epoch 
final score = 0.421483909688


Test_03 

learning_rate = 0.004
batch_size = 32
num_epochs = 35
steps_per_epoch = 100
validation_steps = 100
workers = 32
Time =  62s
final score = 0.392293308138




Test_04 
 
learning_rate = 0.004
batch_size = 64
num_epochs = 35
steps_per_epoch = 75
validation_steps = 75
workers = 32
Time = 86s 
final score = 0.39924187389526666


Test_05 
 
learning_rate = 0.003
batch_size = 64
num_epochs = 35
steps_per_epoch = 100
validation_steps = 100
workers = 32

Time =  103s
final score_1  = 0.404880830407881



Time_2 =  
final score_2 =









Time_3 = 
final sore_3 =  





































Final Score 

measure the model's performance:

 	IOU (intersection over union) metric is used which takes the intersection of the prediction 	pixels and ground truth pixels and divides it by the union of them.

Training GPU: udacity classroom workspace

target hero:  

	wearing red cloths 
	where once trained it will be able to identify the hero from common people 


Final Comments:

This could be used for training on other things other than the specified target. The fully convolutional network would work on other humans, animals, particular cars, vegetation, and other elements that would be part of identification. The training would need to be deployed in a manner to handle other information besides the target object leaving it on our responsibility to train, validate, and test on before running the simulation on another object...
