
# Deep Neural Network with Caffe
### Extracted and enhanced from [Caffe's Documentation](https://github.com/BVLC/caffe/blob/master/examples/00-classification.ipynb)


Coding by hand a problem such as speech recognition is nearly impossible due to the shear amount of variance in the data. The way our brain interprets these kind of problems are complex but turns out they can be modeled with a substantial amount of accuracy, sometimes beating humans itself. The whole concept of artifical neural network started evolving in and after the 1950's. Artificial Neurons gradually evolved from a simple Perceptron to Sigmoid neurons and then to many other forms. Earlier neurons were able to provide binary output to the many inputs that we provide. Newer algorithms and activation functions allow artificial neural network to make complex predictions by learning on its own.

In an artificial neural network, group of neurons are visualized as a layer. ANN's have multiple layers, each layer doing a specific task. The first layer is always the input layer where we input our training or test data. The last layer is the ouput layer where we get the output. Any layer in between is called a hidden layer, which does not take an input or give an output directly from us users.

Whenever you have many number of layers in your ANN, it is termed as a Deep Neural Network (DNN). DNN's have enabled complex Speech recognition, Natural language processing, Computer vision and many other once Star-Trek level Sci-Fi technologies in to existance now. Academic and industrial level research is greatly improving the performance and architecture of DNN's and it is an exciting field to work with.

With huge players like Google opensourcing part of their Machine Learning systems like the TensorFlow software library for numerical computation, there are many options for someone interested in starting off with Machine Learning/Neural Nets to choose from. **Caffe**, a deep learning framework developed by the **Berkeley Vision and Learning Center (BVLC)** and its contributors, comes to the play with a fresh cup of coffee.This tutorial aims in providing an introduction to **Caffe**. The installation instructions can be found [here](https://github.com/arundasan91/Caffe/blob/master/Caffe%20Installation%20Instructions.md).

We will first use a pre-trained model and figure out how Caffe handles a Classification problem. The example is extracted from Caffe's own example section on their GitHub page. Caffe's [Github repo provides examples](https://github.com/BVLC/caffe/tree/master/examples) on many different algorithms and approaches.
I have added my own code and enhancements to help better understand the working of Caffe. Once we are through our Classification example using the pre-trained network, we will try to architect our own network by defining each and every layer. 

First let us import some libraries that are required to visualize the trained Neural net. These include the Numpy library for saving the trained images as arrays and Matplotlib for plotting various figures and graphs out of it.
We also tune the plot parameters as mentioned below.


```python
# set up Python environment: numpy for numerical routines, and matplotlib for plotting
import numpy as np
import matplotlib.pyplot as plt
# display plots in this notebook
%matplotlib inline

# set display defaults
# these are for the matplotlib figure's.
plt.rcParams['figure.figsize'] = (10, 10)        # large images
plt.rcParams['image.interpolation'] = 'nearest'  # don't interpolate: show square pixels
plt.rcParams['image.cmap'] = 'gray'  # use grayscale output rather than a (potentially misleading) color heatmap
```

Caffe's examples are hosted in an example directory inside its root directory. We should run the following python code from the example directory to make sure that the scripts work. We include the *sys* and *os* modules to work with the file paths and the working directory. The caffe's python folder must be fixed as the python path.


```python
# The caffe module needs to be on the Python path;
import sys
import os

caffe_root = '/root/caffe/' # The caffe_root is changed to reflect the actual folder in the server.
sys.path.insert(0, caffe_root + 'python') # Correct the python path

import caffe
# Successfully imported Caffe !

```

    /root/caffe/python/caffe/pycaffe.py:13: RuntimeWarning: to-Python converter for boost::shared_ptr<caffe::Net<float> > already registered; second conversion method ignored.
      from ._caffe import Net, SGDSolver, NesterovSolver, AdaGradSolver, \
    /root/caffe/python/caffe/pycaffe.py:13: RuntimeWarning: to-Python converter for boost::shared_ptr<caffe::Blob<float> > already registered; second conversion method ignored.
      from ._caffe import Net, SGDSolver, NesterovSolver, AdaGradSolver, \
    /root/caffe/python/caffe/pycaffe.py:13: RuntimeWarning: to-Python converter for boost::shared_ptr<caffe::Solver<float> > already registered; second conversion method ignored.
      from ._caffe import Net, SGDSolver, NesterovSolver, AdaGradSolver, \


Training a model is computationally expensive and time consuming. For this example, let us stick on to a pre-trained network bundled with Caffe. We will search for the caffemodel and start from there. Caffemodel is the trained model. During the training phase, on the set time intervals or iterations, Caffe saves caffemodel file which saves the state of the net at that particular time. For example, if we have a total of 10000 iterations to perform and we explicitly mention that we need to save the state of the net at intervals of 2000 iterations, Caffe will generate 5 caffemodel files and 5 solver files, which saves the respective state of the net at iterations 2000,4000,6000,8000 and 10000.

Since we explicitly fixed the working directory, it is not required to have the Notebook in the same directory as that of the example. We define the Model definitions and weights of the pre-trained network by including the correct path. The neural net is defined by using the defenitions and weigths saved earlier. Since the network was already trained for a huge dataset, we can choose the Test mode in caffe and not perform dropout's while defining the net. More info on dropout [here](https://www.cs.toronto.edu/~hinton/absps/JMLRdropout.pdf). 


```python
# set Caffe mode as CPU only. This is to be done because OCI servers are not equipped with GPU's yet.
caffe.set_mode_cpu()

# set the model definitions since we are using a pretrained network here.
# this protoype definitions can be changed to make significant changes in the learning method.
model_def = '/root/machineLearning/deepNeuralNet/caffe/caffemodels/bvlc/caffenet/deploy_changed_net.prototxt'
model_weights = '/root/machineLearning/deepNeuralNet/caffe/caffemodels/bvlc/caffenet/caffenet.caffemodel'

net = caffe.Net(model_def,      # defines the structure of the model
                model_weights,  # contains the trained weights
                caffe.TEST)     # use test mode (e.g., don't perform dropout)
```

We can visualize our network architecture by converting the model defenition prototxt file into an image. The `draw_net.py` python code will allow us to do that. Let us now see the prototxt file and its visual interpretation. The image here in the notebook is pretty small, you can view it better [here](https://raw.githubusercontent.com/arundasan91/Caffe/master/Data/deploy_changed_net.png).


```python
%%bash
cat /root/machineLearning/deepNeuralNet/caffe/caffemodels/bvlc/caffenet/deploy_changed_net.prototxt
python $CAFFE_ROOT/python/draw_net.py \
/root/machineLearning/deepNeuralNet/caffe/caffemodels/bvlc/caffenet/deploy_changed_net.prototxt \
/root/machineLearning/deepNeuralNet/caffe/caffemodels/bvlc/caffenet/deploy_changed_net.png
```

    name: "CaffeNet"
    layer {
      name: "data"
      type: "Input"
      top: "data"
      input_param { shape: { dim: 10 dim: 3 dim: 227 dim: 227 } }
    }
    layer {
      name: "conv1"
      type: "Convolution"
      bottom: "data"
      top: "conv1"
      convolution_param {
        num_output: 96
        kernel_size: 11
        stride: 4
      }
    }
    layer {
      name: "relu1"
      type: "ReLU"
      bottom: "conv1"
      top: "conv1"
    }
    layer {
      name: "pool1"
      type: "Pooling"
      bottom: "conv1"
      top: "pool1"
      pooling_param {
        pool: MAX
        kernel_size: 3
        stride: 2
      }
    }
    layer {
      name: "norm1"
      type: "LRN"
      bottom: "pool1"
      top: "norm1"
      lrn_param {
        local_size: 5
        alpha: 0.0001
        beta: 0.75
      }
    }
    layer {
      name: "conv2"
      type: "Convolution"
      bottom: "norm1"
      top: "conv2"
      convolution_param {
        num_output: 256
        pad: 2
        kernel_size: 5
        group: 2
      }
    }
    layer {
      name: "relu2"
      type: "ReLU"
      bottom: "conv2"
      top: "conv2"
    }
    layer {
      name: "pool2"
      type: "Pooling"
      bottom: "conv2"
      top: "pool2"
      pooling_param {
        pool: MAX
        kernel_size: 3
        stride: 2
      }
    }
    layer {
      name: "norm2"
      type: "LRN"
      bottom: "pool2"
      top: "norm2"
      lrn_param {
        local_size: 5
        alpha: 0.0001
        beta: 0.75
      }
    }
    layer {
      name: "conv3"
      type: "Convolution"
      bottom: "norm2"
      top: "conv3"
      convolution_param {
        num_output: 384
        pad: 1
        kernel_size: 3
      }
    }
    layer {
      name: "relu3"
      type: "ReLU"
      bottom: "conv3"
      top: "conv3"
    }
    layer {
      name: "conv4"
      type: "Convolution"
      bottom: "conv3"
      top: "conv4"
      convolution_param {
        num_output: 384
        pad: 1
        kernel_size: 3
        group: 2
      }
    }
    layer {
      name: "relu4"
      type: "ReLU"
      bottom: "conv4"
      top: "conv4"
    }
    layer {
      name: "conv5"
      type: "Convolution"
      bottom: "conv4"
      top: "conv5"
      convolution_param {
        num_output: 256
        pad: 1
        kernel_size: 3
        group: 2
      }
    }
    layer {
      name: "relu5"
      type: "ReLU"
      bottom: "conv5"
      top: "conv5"
    }
    layer {
      name: "pool5"
      type: "Pooling"
      bottom: "conv5"
      top: "pool5"
      pooling_param {
        pool: MAX
        kernel_size: 3
        stride: 2
      }
    }
    layer {
      name: "fc6"
      type: "InnerProduct"
      bottom: "pool5"
      top: "fc6"
      inner_product_param {
        num_output: 4096
      }
    }
    layer {
      name: "relu6"
      type: "ReLU"
      bottom: "fc6"
      top: "fc6"
    }
    layer {
      name: "drop6"
      type: "Dropout"
      bottom: "fc6"
      top: "fc6"
      dropout_param {
        dropout_ratio: 0.5
      }
    }
    layer {
      name: "fc7"
      type: "InnerProduct"
      bottom: "fc6"
      top: "fc7"
      inner_product_param {
        num_output: 4096
      }
    }
    layer {
      name: "relu7"
      type: "ReLU"
      bottom: "fc7"
      top: "fc7"
    }
    layer {
      name: "drop7"
      type: "Dropout"
      bottom: "fc7"
      top: "fc7"
      dropout_param {
        dropout_ratio: 0.5
      }
    }
    layer {
      name: "fc8"
      type: "InnerProduct"
      bottom: "fc7"
      top: "fc8"
      inner_product_param {
        num_output: 1000
      }
    }
    layer {
      name: "prob"
      type: "Softmax"
      bottom: "fc8"
      top: "prob"
    }
    Drawing net to /root/machineLearning/deepNeuralNet/caffe/caffemodels/bvlc/caffenet/deploy_changed_net.png


    /root/caffe/python/caffe/pycaffe.py:13: RuntimeWarning: to-Python converter for boost::shared_ptr<caffe::Net<float> > already registered; second conversion method ignored.
      from ._caffe import Net, SGDSolver, NesterovSolver, AdaGradSolver, \
    /root/caffe/python/caffe/pycaffe.py:13: RuntimeWarning: to-Python converter for boost::shared_ptr<caffe::Blob<float> > already registered; second conversion method ignored.
      from ._caffe import Net, SGDSolver, NesterovSolver, AdaGradSolver, \
    /root/caffe/python/caffe/pycaffe.py:13: RuntimeWarning: to-Python converter for boost::shared_ptr<caffe::Solver<float> > already registered; second conversion method ignored.
      from ._caffe import Net, SGDSolver, NesterovSolver, AdaGradSolver, \



```python
image = caffe.io.load_image('/root/machineLearning/deepNeuralNet/caffe/caffemodels/bvlc/caffenet/deploy_changed_net.png')
plt.imshow(image)
```




    <matplotlib.image.AxesImage at 0x7f8ee578c810>




![png](output_15_1.png)


***The image here in the notebook is pretty small, you can view it better [here](The image here in the notebook is pretty small, you can view it better here.).***

Now we can create a transformer to input data into our net. Input data here are images. The subtracted-mean of the images in the dataset considered are to be set in the transformer. Mean subtraction is a way of preprocessing the image. The mean is subtracted across every individual feature in the dataset. This can be interpreted as the centering of a cloud of data around the origin along every dimension. With our input data fixed as images, this relates to subtracting the mean from each of the pixels, seperately across the three channels. More on it [here](http://cs231n.github.io/neural-networks-2/).

These are the steps usually carried out in each transformers:
1. Transpose the data from (height, width, channels) to (channels, width, height)
2. Swap the color channels from RGB to BGR
3. Subtract the mean pixel value of the training dataset (unless you disable that feature).

More information on these [here](https://groups.google.com/forum/#!topic/digits-users/FIh6VyU1XqQ), [here](https://github.com/NVIDIA/DIGITS/issues/59) and [here](https://github.com/NVIDIA/DIGITS/blob/v1.1.0/digits/model/tasks/caffe_train.py#L938-L961).


```python
# load the mean ImageNet image (as distributed with Caffe) for subtraction
mu = np.load('/root/machineLearning/deepNeuralNet/caffe/datasets/ilsvrc12/ilsvrc_2012_mean.npy')
mu = mu.mean(1).mean(1)  # average over pixels to obtain the mean (BGR) pixel values
print 'mean-subtracted values:', zip('BGR', mu)

# create transformer for the input called 'data'
transformer = caffe.io.Transformer({'data': net.blobs['data'].data.shape})

transformer.set_transpose('data', (2,0,1))  # move image channels to outermost dimension
transformer.set_mean('data', mu)            # subtract the dataset-mean value in each channel
transformer.set_raw_scale('data', 255)      # rescale from [0, 1] to [0, 255]
transformer.set_channel_swap('data', (2,1,0))  # swap channels from RGB to BGR
```

    mean-subtracted values: [('B', 104.0069879317889), ('G', 116.66876761696767), ('R', 122.6789143406786)]


If needed, we can reshape the data to meet our specifications. In the particular example the batch size, number of channels and image size is explicitly specified as below.


```python
# set the size of the input (we can skip this if we're happy
#  with the default; we can also change it later, e.g., for different batch sizes)
#net.blobs['data'].reshape(50,        # batch size
#                          3,         # 3-channel (BGR) images
#                          227, 227)  # image size is 227x227
```

Any image can be now loaded into caffe. For simplicity let us now stick with Caffe's example image ($CAFFE_ROOT/examples/images/cat.jpg). The image is then transformed as mentioned above using the transformer that we defined. Finally, the image is plotted using matplotlib.pyplot imported as plt.


```python
image = caffe.io.load_image('/root/machineLearning/deepNeuralNet/caffe/datasets/images/samples/cat.jpg')
transformed_image = transformer.preprocess('data', image)
plt.imshow(image)
```




    <matplotlib.image.AxesImage at 0x7f8ee4585a50>




![png](output_22_1.png)


Great ! Now we have our net ready so is the image that we need to classify. Remember that data in Caffe is interpreted using blobs. Quoting from [Caffe's Documentation](http://caffe.berkeleyvision.org/tutorial/net_layer_blob.html) : *As data and derivatives flow through the network in the forward and backward passes Caffe stores, communicates, and manipulates the information as blobs: the blob is the standard array and unified memory interface for the framework.* 

For caffe to get information from the image, it needs to be copied to the memory allocated by Caffe. 

Once the image is loaded into memory, we can perform classification with it. To start classification, we call ***net.forward()*** and redirect its output to a variavle named output (name can be anything obviously). The probability of the output is saved in a vector format. Since we gave a batch size of 50, there will be 50 input images at once. The probability of our image will be saved in the **[0]**th location. The output probability can be extracted out by properly calling it. Finally the predicted class of the image can be extracted by using argmax which returns the indices of the maximum values along an axis. 


```python
# copy the image data into the memory allocated for the net
net.blobs['data'].data[...] = transformed_image

### perform classification
output = net.forward()

output_prob = output['prob'][0]  # the output probability vector for the first image in the batch

print 'predicted class is:', output_prob.argmax()
```

    predicted class is: 281


In our case, for the cute cat, the predicted class is 281. Make sure that you are getting the same (just in case).

For our eyes the image is a cute cat, agreed. To see what our net thinks it is, let us fetch the label of the predicted/classified image. Load the labels file from the dataset and output the specific label.


```python
# load ImageNet labels

labels_file = '/root/machineLearning/deepNeuralNet/caffe/datasets/ilsvrc12/synset_words.txt'

if not os.path.exists(labels_file):
    !/root/caffe/data/ilsvrc12/get_ilsvrc_aux.sh
    
labels = np.loadtxt(labels_file, str, delimiter='\t')

print 'output label:', labels[output_prob.argmax()]
```

    output label: n02123045 tabby, tabby cat


What do you think about the prediction ? Fair ? Let us see a quantitative result. We will output the top five predictions from the output layer (softmax layer).


```python
# sort top five predictions from softmax output
top_inds = output_prob.argsort()[::-1][:5]  # reverse sort and take five largest items

print 'probabilities and labels:'
zip(output_prob[top_inds], labels[top_inds])
```

    probabilities and labels:





    [(0.31243613, 'n02123045 tabby, tabby cat'),
     (0.23797171, 'n02123159 tiger cat'),
     (0.1238724, 'n02124075 Egyptian cat'),
     (0.10075741, 'n02119022 red fox, Vulpes vulpes'),
     (0.07095696, 'n02127052 lynx, catamount')]



To find the time required to train the network for the particular input, let us use timeit function.


```python
# find the time required to train the network
%timeit net.forward()
```

    1 loop, best of 3: 694 ms per loop


**blob.data.shape** can be used to find the shape of the different layers in your net. Loop across it to get shape of each layer.


```python
# for each layer, show the output shape
for layer_name, blob in net.blobs.iteritems():
    print layer_name + '\t' + str(blob.data.shape)
```

    data	(10, 3, 227, 227)
    conv1	(10, 96, 55, 55)
    pool1	(10, 96, 27, 27)
    norm1	(10, 96, 27, 27)
    conv2	(10, 256, 27, 27)
    pool2	(10, 256, 13, 13)
    norm2	(10, 256, 13, 13)
    conv3	(10, 384, 13, 13)
    conv4	(10, 384, 13, 13)
    conv5	(10, 256, 13, 13)
    pool5	(10, 256, 6, 6)
    fc6	(10, 4096)
    fc7	(10, 4096)
    fc8	(10, 1000)
    prob	(10, 1000)



```python
for layer_name, param in net.params.iteritems():
    print layer_name + '\t' + str(param[0].data.shape), str(param[1].data.shape)
```

    conv1	(96, 3, 11, 11) (96,)
    conv2	(256, 48, 5, 5) (256,)
    conv3	(384, 256, 3, 3) (384,)
    conv4	(384, 192, 3, 3) (384,)
    conv5	(256, 192, 3, 3) (256,)
    fc6	(4096, 9216) (4096,)
    fc7	(4096, 4096) (4096,)
    fc8	(1000, 4096) (1000,)



```python
def vis_square(data):
    """Take an array of shape (n, height, width) or (n, height, width, 3)
       and visualize each (height, width) thing in a grid of size approx. sqrt(n) by sqrt(n)"""
    
    # normalize data for display
    data = (data - data.min()) / (data.max() - data.min())
    
    # force the number of filters to be square
    n = int(np.ceil(np.sqrt(data.shape[0])))
    padding = (((0, n ** 2 - data.shape[0]),
               (0, 1), (0, 1))                 # add some space between filters
               + ((0, 0),) * (data.ndim - 3))  # don't pad the last dimension (if there is one)
    data = np.pad(data, padding, mode='constant', constant_values=1)  # pad with ones (white)
    
    # tile the filters into an image
    data = data.reshape((n, n) + data.shape[1:]).transpose((0, 2, 1, 3) + tuple(range(4, data.ndim + 1)))
    data = data.reshape((n * data.shape[1], n * data.shape[3]) + data.shape[4:])
    
    plt.imshow(data); plt.axis('off')
```

We can filter out the weights and biases of each layer to visualize the changes happening in each layer. This is a powerful way of analyzing the net as it gives intuition into what is happening inside it. The image drawn above (layer by layer representation) together with the visualization of things happening in each layer will help us understand the net in more depth. Here we use the **conv1** layer for the same.


```python
# the parameters are a list of [weights, biases]
filters = net.params['conv1'][0].data
filters.shape
```




    (96, 3, 11, 11)



If you noticed, the shape of the filter is different from the function vis_square. So we need to transpose the vector accordingly before passing it into the function to visualize the layers. We pass in the data of first convolution layer. The image below shows that the lower layer is working as an edge detector of sort.


```python
vis_square(filters.transpose(0, 2, 3, 1))
```


![png](output_38_0.png)


To visualize the data as such, we can use the net.blobs instead of net.params. This will give us a visual clue on what the data may look like and what is happening at the same time. We are doing it for the **conv1** layer.


```python
feat = net.blobs['conv1'].data[0, :36]
vis_square(feat)
```


![png](output_40_0.png)


Similarly for the **pool5** layer.


```python
feat = net.blobs['pool5'].data[0]
vis_square(feat)
```


![png](output_42_0.png)


We can plot graphs using the various data saved in the layers. The fully connected layer fc6 will result in the following plot.


```python
feat = net.blobs['fc6'].data[0]
plt.subplot(2, 1, 1)
plt.plot(feat.flat)
plt.subplot(2, 1, 2)
_ = plt.hist(feat.flat[feat.flat > 0], bins=100)
```


![png](output_44_0.png)


The probability of predicting the correct label for the particular image we classified can be plotted as well. X-axis is the Feature's label number and Y-Axis is the probability of correct classification.


```python
feat = net.blobs['prob'].data[0]
plt.figure(figsize=(15, 3))
plt.plot(feat.flat)
```




    [<matplotlib.lines.Line2D at 0x7f8ee440e310>]




![png](output_46_1.png)


Now, let us download an image of our own and try to classify it. Here, a http link of the image is used to download the image. The image is then loaded into Caffe. The image is then preprocessed using the transformer we defined earlier.

Once we are done with the preprocessing, we have a formated image in the memory that is ready to be classified. Perform the classification by running **net.forward()**. The output probability can be found just like earlier.The top 5 probabilities are dound out, the image displayed and the 5 probabilities are printed out.


```python
# download an image
# for example:
# my_image_url = "https://upload.wikimedia.org/wikipedia/commons/b/be/Orang_Utan%2C_Semenggok_Forest_Reserve%2C_Sarawak%2C_Borneo%2C_Malaysia.JPG"
# my_image_url = "https://www.petfinder.com/wp-content/uploads/2012/11/140272627-grooming-needs-senior-cat-632x475.jpg"  # paste your URL here
# my_image_url = "http://kids.nationalgeographic.com/content/dam/kids/photos/animals/Mammals/H-P/lion-male-roar.jpg"

my_image_url ="http://www.depositagift.com/img/bank_assets/Band-Aid.jpg"

!wget -O image.jpg $my_image_url

# transform it and copy it into the net
image = caffe.io.load_image('image.jpg')
net.blobs['data'].data[...] = transformer.preprocess('data', image)

# perform classification
net.forward()

# obtain the output probabilities
output_prob = net.blobs['prob'].data[0]

# sort top five predictions from softmax output
top_inds = output_prob.argsort()[::-1][:5]

plt.imshow(image)

print 'probabilities and labels:'
zip(output_prob[top_inds], labels[top_inds])
```

    --2016-04-16 01:54:15--  http://www.depositagift.com/img/bank_assets/Band-Aid.jpg
    Resolving www.depositagift.com (www.depositagift.com)... 50.28.4.115
    Connecting to www.depositagift.com (www.depositagift.com)|50.28.4.115|:80... connected.
    HTTP request sent, awaiting response... 200 OK
    Length: 34197 (33K) [image/jpeg]
    Saving to: 'image.jpg'
    
    100%[======================================>] 34,197      --.-K/s   in 0.1s    
    
    2016-04-16 01:54:15 (230 KB/s) - 'image.jpg' saved [34197/34197]
    
    probabilities and labels:





    [(0.94379258, 'n02786058 Band Aid'),
     (0.0064510447, 'n03530642 honeycomb'),
     (0.0061318246, 'n07684084 French loaf'),
     (0.0045337547, 'n04476259 tray'),
     (0.0042723794, 'n03314780 face powder')]




![png](output_48_2.png)


If you ever wonder what are the labels in the dataset, run the following:


```python
zip(labels[:])
```




    [('n01440764 tench, Tinca tinca',),
     ('n01443537 goldfish, Carassius auratus',),
     ('n01484850 great white shark, white shark, man-eater, man-eating shark, Carcharodon carcharias',),
     ('n01491361 tiger shark, Galeocerdo cuvieri',),
     ('n01494475 hammerhead, hammerhead shark',),
     ('n01496331 electric ray, crampfish, numbfish, torpedo',),
     ('n01498041 stingray',),
     ('n01514668 cock',),
     ('n01514859 hen',),
     ('n01518878 ostrich, Struthio camelus',),
     ('n01530575 brambling, Fringilla montifringilla',),
     ('n01531178 goldfinch, Carduelis carduelis',),
     ('n01532829 house finch, linnet, Carpodacus mexicanus',),
     ('n01534433 junco, snowbird',),
     ('n01537544 indigo bunting, indigo finch, indigo bird, Passerina cyanea',),
     ('n01558993 robin, American robin, Turdus migratorius',),
     ('n01560419 bulbul',),
     ('n01580077 jay',),
     ('n01582220 magpie',),
     ('n01592084 chickadee',),
     ('n01601694 water ouzel, dipper',),
     ('n01608432 kite',),
     ('n01614925 bald eagle, American eagle, Haliaeetus leucocephalus',),
     ('n01616318 vulture',),
     ('n01622779 great grey owl, great gray owl, Strix nebulosa',),
     ('n01629819 European fire salamander, Salamandra salamandra',),
     ('n01630670 common newt, Triturus vulgaris',),
     ('n01631663 eft',),
     ('n01632458 spotted salamander, Ambystoma maculatum',),
     ('n01632777 axolotl, mud puppy, Ambystoma mexicanum',),
     ('n01641577 bullfrog, Rana catesbeiana',),
     ('n01644373 tree frog, tree-frog',),
     ('n01644900 tailed frog, bell toad, ribbed toad, tailed toad, Ascaphus trui',),
     ('n01664065 loggerhead, loggerhead turtle, Caretta caretta',),
     ('n01665541 leatherback turtle, leatherback, leathery turtle, Dermochelys coriacea',),
     ('n01667114 mud turtle',),
     ('n01667778 terrapin',),
     ('n01669191 box turtle, box tortoise',),
     ('n01675722 banded gecko',),
     ('n01677366 common iguana, iguana, Iguana iguana',),
     ('n01682714 American chameleon, anole, Anolis carolinensis',),
     ('n01685808 whiptail, whiptail lizard',),
     ('n01687978 agama',),
     ('n01688243 frilled lizard, Chlamydosaurus kingi',),
     ('n01689811 alligator lizard',),
     ('n01692333 Gila monster, Heloderma suspectum',),
     ('n01693334 green lizard, Lacerta viridis',),
     ('n01694178 African chameleon, Chamaeleo chamaeleon',),
     ('n01695060 Komodo dragon, Komodo lizard, dragon lizard, giant lizard, Varanus komodoensis',),
     ('n01697457 African crocodile, Nile crocodile, Crocodylus niloticus',),
     ('n01698640 American alligator, Alligator mississipiensis',),
     ('n01704323 triceratops',),
     ('n01728572 thunder snake, worm snake, Carphophis amoenus',),
     ('n01728920 ringneck snake, ring-necked snake, ring snake',),
     ('n01729322 hognose snake, puff adder, sand viper',),
     ('n01729977 green snake, grass snake',),
     ('n01734418 king snake, kingsnake',),
     ('n01735189 garter snake, grass snake',),
     ('n01737021 water snake',),
     ('n01739381 vine snake',),
     ('n01740131 night snake, Hypsiglena torquata',),
     ('n01742172 boa constrictor, Constrictor constrictor',),
     ('n01744401 rock python, rock snake, Python sebae',),
     ('n01748264 Indian cobra, Naja naja',),
     ('n01749939 green mamba',),
     ('n01751748 sea snake',),
     ('n01753488 horned viper, cerastes, sand viper, horned asp, Cerastes cornutus',),
     ('n01755581 diamondback, diamondback rattlesnake, Crotalus adamanteus',),
     ('n01756291 sidewinder, horned rattlesnake, Crotalus cerastes',),
     ('n01768244 trilobite',),
     ('n01770081 harvestman, daddy longlegs, Phalangium opilio',),
     ('n01770393 scorpion',),
     ('n01773157 black and gold garden spider, Argiope aurantia',),
     ('n01773549 barn spider, Araneus cavaticus',),
     ('n01773797 garden spider, Aranea diademata',),
     ('n01774384 black widow, Latrodectus mactans',),
     ('n01774750 tarantula',),
     ('n01775062 wolf spider, hunting spider',),
     ('n01776313 tick',),
     ('n01784675 centipede',),
     ('n01795545 black grouse',),
     ('n01796340 ptarmigan',),
     ('n01797886 ruffed grouse, partridge, Bonasa umbellus',),
     ('n01798484 prairie chicken, prairie grouse, prairie fowl',),
     ('n01806143 peacock',),
     ('n01806567 quail',),
     ('n01807496 partridge',),
     ('n01817953 African grey, African gray, Psittacus erithacus',),
     ('n01818515 macaw',),
     ('n01819313 sulphur-crested cockatoo, Kakatoe galerita, Cacatua galerita',),
     ('n01820546 lorikeet',),
     ('n01824575 coucal',),
     ('n01828970 bee eater',),
     ('n01829413 hornbill',),
     ('n01833805 hummingbird',),
     ('n01843065 jacamar',),
     ('n01843383 toucan',),
     ('n01847000 drake',),
     ('n01855032 red-breasted merganser, Mergus serrator',),
     ('n01855672 goose',),
     ('n01860187 black swan, Cygnus atratus',),
     ('n01871265 tusker',),
     ('n01872401 echidna, spiny anteater, anteater',),
     ('n01873310 platypus, duckbill, duckbilled platypus, duck-billed platypus, Ornithorhynchus anatinus',),
     ('n01877812 wallaby, brush kangaroo',),
     ('n01882714 koala, koala bear, kangaroo bear, native bear, Phascolarctos cinereus',),
     ('n01883070 wombat',),
     ('n01910747 jellyfish',),
     ('n01914609 sea anemone, anemone',),
     ('n01917289 brain coral',),
     ('n01924916 flatworm, platyhelminth',),
     ('n01930112 nematode, nematode worm, roundworm',),
     ('n01943899 conch',),
     ('n01944390 snail',),
     ('n01945685 slug',),
     ('n01950731 sea slug, nudibranch',),
     ('n01955084 chiton, coat-of-mail shell, sea cradle, polyplacophore',),
     ('n01968897 chambered nautilus, pearly nautilus, nautilus',),
     ('n01978287 Dungeness crab, Cancer magister',),
     ('n01978455 rock crab, Cancer irroratus',),
     ('n01980166 fiddler crab',),
     ('n01981276 king crab, Alaska crab, Alaskan king crab, Alaska king crab, Paralithodes camtschatica',),
     ('n01983481 American lobster, Northern lobster, Maine lobster, Homarus americanus',),
     ('n01984695 spiny lobster, langouste, rock lobster, crawfish, crayfish, sea crawfish',),
     ('n01985128 crayfish, crawfish, crawdad, crawdaddy',),
     ('n01986214 hermit crab',),
     ('n01990800 isopod',),
     ('n02002556 white stork, Ciconia ciconia',),
     ('n02002724 black stork, Ciconia nigra',),
     ('n02006656 spoonbill',),
     ('n02007558 flamingo',),
     ('n02009229 little blue heron, Egretta caerulea',),
     ('n02009912 American egret, great white heron, Egretta albus',),
     ('n02011460 bittern',),
     ('n02012849 crane',),
     ('n02013706 limpkin, Aramus pictus',),
     ('n02017213 European gallinule, Porphyrio porphyrio',),
     ('n02018207 American coot, marsh hen, mud hen, water hen, Fulica americana',),
     ('n02018795 bustard',),
     ('n02025239 ruddy turnstone, Arenaria interpres',),
     ('n02027492 red-backed sandpiper, dunlin, Erolia alpina',),
     ('n02028035 redshank, Tringa totanus',),
     ('n02033041 dowitcher',),
     ('n02037110 oystercatcher, oyster catcher',),
     ('n02051845 pelican',),
     ('n02056570 king penguin, Aptenodytes patagonica',),
     ('n02058221 albatross, mollymawk',),
     ('n02066245 grey whale, gray whale, devilfish, Eschrichtius gibbosus, Eschrichtius robustus',),
     ('n02071294 killer whale, killer, orca, grampus, sea wolf, Orcinus orca',),
     ('n02074367 dugong, Dugong dugon',),
     ('n02077923 sea lion',),
     ('n02085620 Chihuahua',),
     ('n02085782 Japanese spaniel',),
     ('n02085936 Maltese dog, Maltese terrier, Maltese',),
     ('n02086079 Pekinese, Pekingese, Peke',),
     ('n02086240 Shih-Tzu',),
     ('n02086646 Blenheim spaniel',),
     ('n02086910 papillon',),
     ('n02087046 toy terrier',),
     ('n02087394 Rhodesian ridgeback',),
     ('n02088094 Afghan hound, Afghan',),
     ('n02088238 basset, basset hound',),
     ('n02088364 beagle',),
     ('n02088466 bloodhound, sleuthhound',),
     ('n02088632 bluetick',),
     ('n02089078 black-and-tan coonhound',),
     ('n02089867 Walker hound, Walker foxhound',),
     ('n02089973 English foxhound',),
     ('n02090379 redbone',),
     ('n02090622 borzoi, Russian wolfhound',),
     ('n02090721 Irish wolfhound',),
     ('n02091032 Italian greyhound',),
     ('n02091134 whippet',),
     ('n02091244 Ibizan hound, Ibizan Podenco',),
     ('n02091467 Norwegian elkhound, elkhound',),
     ('n02091635 otterhound, otter hound',),
     ('n02091831 Saluki, gazelle hound',),
     ('n02092002 Scottish deerhound, deerhound',),
     ('n02092339 Weimaraner',),
     ('n02093256 Staffordshire bullterrier, Staffordshire bull terrier',),
     ('n02093428 American Staffordshire terrier, Staffordshire terrier, American pit bull terrier, pit bull terrier',),
     ('n02093647 Bedlington terrier',),
     ('n02093754 Border terrier',),
     ('n02093859 Kerry blue terrier',),
     ('n02093991 Irish terrier',),
     ('n02094114 Norfolk terrier',),
     ('n02094258 Norwich terrier',),
     ('n02094433 Yorkshire terrier',),
     ('n02095314 wire-haired fox terrier',),
     ('n02095570 Lakeland terrier',),
     ('n02095889 Sealyham terrier, Sealyham',),
     ('n02096051 Airedale, Airedale terrier',),
     ('n02096177 cairn, cairn terrier',),
     ('n02096294 Australian terrier',),
     ('n02096437 Dandie Dinmont, Dandie Dinmont terrier',),
     ('n02096585 Boston bull, Boston terrier',),
     ('n02097047 miniature schnauzer',),
     ('n02097130 giant schnauzer',),
     ('n02097209 standard schnauzer',),
     ('n02097298 Scotch terrier, Scottish terrier, Scottie',),
     ('n02097474 Tibetan terrier, chrysanthemum dog',),
     ('n02097658 silky terrier, Sydney silky',),
     ('n02098105 soft-coated wheaten terrier',),
     ('n02098286 West Highland white terrier',),
     ('n02098413 Lhasa, Lhasa apso',),
     ('n02099267 flat-coated retriever',),
     ('n02099429 curly-coated retriever',),
     ('n02099601 golden retriever',),
     ('n02099712 Labrador retriever',),
     ('n02099849 Chesapeake Bay retriever',),
     ('n02100236 German short-haired pointer',),
     ('n02100583 vizsla, Hungarian pointer',),
     ('n02100735 English setter',),
     ('n02100877 Irish setter, red setter',),
     ('n02101006 Gordon setter',),
     ('n02101388 Brittany spaniel',),
     ('n02101556 clumber, clumber spaniel',),
     ('n02102040 English springer, English springer spaniel',),
     ('n02102177 Welsh springer spaniel',),
     ('n02102318 cocker spaniel, English cocker spaniel, cocker',),
     ('n02102480 Sussex spaniel',),
     ('n02102973 Irish water spaniel',),
     ('n02104029 kuvasz',),
     ('n02104365 schipperke',),
     ('n02105056 groenendael',),
     ('n02105162 malinois',),
     ('n02105251 briard',),
     ('n02105412 kelpie',),
     ('n02105505 komondor',),
     ('n02105641 Old English sheepdog, bobtail',),
     ('n02105855 Shetland sheepdog, Shetland sheep dog, Shetland',),
     ('n02106030 collie',),
     ('n02106166 Border collie',),
     ('n02106382 Bouvier des Flandres, Bouviers des Flandres',),
     ('n02106550 Rottweiler',),
     ('n02106662 German shepherd, German shepherd dog, German police dog, alsatian',),
     ('n02107142 Doberman, Doberman pinscher',),
     ('n02107312 miniature pinscher',),
     ('n02107574 Greater Swiss Mountain dog',),
     ('n02107683 Bernese mountain dog',),
     ('n02107908 Appenzeller',),
     ('n02108000 EntleBucher',),
     ('n02108089 boxer',),
     ('n02108422 bull mastiff',),
     ('n02108551 Tibetan mastiff',),
     ('n02108915 French bulldog',),
     ('n02109047 Great Dane',),
     ('n02109525 Saint Bernard, St Bernard',),
     ('n02109961 Eskimo dog, husky',),
     ('n02110063 malamute, malemute, Alaskan malamute',),
     ('n02110185 Siberian husky',),
     ('n02110341 dalmatian, coach dog, carriage dog',),
     ('n02110627 affenpinscher, monkey pinscher, monkey dog',),
     ('n02110806 basenji',),
     ('n02110958 pug, pug-dog',),
     ('n02111129 Leonberg',),
     ('n02111277 Newfoundland, Newfoundland dog',),
     ('n02111500 Great Pyrenees',),
     ('n02111889 Samoyed, Samoyede',),
     ('n02112018 Pomeranian',),
     ('n02112137 chow, chow chow',),
     ('n02112350 keeshond',),
     ('n02112706 Brabancon griffon',),
     ('n02113023 Pembroke, Pembroke Welsh corgi',),
     ('n02113186 Cardigan, Cardigan Welsh corgi',),
     ('n02113624 toy poodle',),
     ('n02113712 miniature poodle',),
     ('n02113799 standard poodle',),
     ('n02113978 Mexican hairless',),
     ('n02114367 timber wolf, grey wolf, gray wolf, Canis lupus',),
     ('n02114548 white wolf, Arctic wolf, Canis lupus tundrarum',),
     ('n02114712 red wolf, maned wolf, Canis rufus, Canis niger',),
     ('n02114855 coyote, prairie wolf, brush wolf, Canis latrans',),
     ('n02115641 dingo, warrigal, warragal, Canis dingo',),
     ('n02115913 dhole, Cuon alpinus',),
     ('n02116738 African hunting dog, hyena dog, Cape hunting dog, Lycaon pictus',),
     ('n02117135 hyena, hyaena',),
     ('n02119022 red fox, Vulpes vulpes',),
     ('n02119789 kit fox, Vulpes macrotis',),
     ('n02120079 Arctic fox, white fox, Alopex lagopus',),
     ('n02120505 grey fox, gray fox, Urocyon cinereoargenteus',),
     ('n02123045 tabby, tabby cat',),
     ('n02123159 tiger cat',),
     ('n02123394 Persian cat',),
     ('n02123597 Siamese cat, Siamese',),
     ('n02124075 Egyptian cat',),
     ('n02125311 cougar, puma, catamount, mountain lion, painter, panther, Felis concolor',),
     ('n02127052 lynx, catamount',),
     ('n02128385 leopard, Panthera pardus',),
     ('n02128757 snow leopard, ounce, Panthera uncia',),
     ('n02128925 jaguar, panther, Panthera onca, Felis onca',),
     ('n02129165 lion, king of beasts, Panthera leo',),
     ('n02129604 tiger, Panthera tigris',),
     ('n02130308 cheetah, chetah, Acinonyx jubatus',),
     ('n02132136 brown bear, bruin, Ursus arctos',),
     ('n02133161 American black bear, black bear, Ursus americanus, Euarctos americanus',),
     ('n02134084 ice bear, polar bear, Ursus Maritimus, Thalarctos maritimus',),
     ('n02134418 sloth bear, Melursus ursinus, Ursus ursinus',),
     ('n02137549 mongoose',),
     ('n02138441 meerkat, mierkat',),
     ('n02165105 tiger beetle',),
     ('n02165456 ladybug, ladybeetle, lady beetle, ladybird, ladybird beetle',),
     ('n02167151 ground beetle, carabid beetle',),
     ('n02168699 long-horned beetle, longicorn, longicorn beetle',),
     ('n02169497 leaf beetle, chrysomelid',),
     ('n02172182 dung beetle',),
     ('n02174001 rhinoceros beetle',),
     ('n02177972 weevil',),
     ('n02190166 fly',),
     ('n02206856 bee',),
     ('n02219486 ant, emmet, pismire',),
     ('n02226429 grasshopper, hopper',),
     ('n02229544 cricket',),
     ('n02231487 walking stick, walkingstick, stick insect',),
     ('n02233338 cockroach, roach',),
     ('n02236044 mantis, mantid',),
     ('n02256656 cicada, cicala',),
     ('n02259212 leafhopper',),
     ('n02264363 lacewing, lacewing fly',),
     ("n02268443 dragonfly, darning needle, devil's darning needle, sewing needle, snake feeder, snake doctor, mosquito hawk, skeeter hawk",),
     ('n02268853 damselfly',),
     ('n02276258 admiral',),
     ('n02277742 ringlet, ringlet butterfly',),
     ('n02279972 monarch, monarch butterfly, milkweed butterfly, Danaus plexippus',),
     ('n02280649 cabbage butterfly',),
     ('n02281406 sulphur butterfly, sulfur butterfly',),
     ('n02281787 lycaenid, lycaenid butterfly',),
     ('n02317335 starfish, sea star',),
     ('n02319095 sea urchin',),
     ('n02321529 sea cucumber, holothurian',),
     ('n02325366 wood rabbit, cottontail, cottontail rabbit',),
     ('n02326432 hare',),
     ('n02328150 Angora, Angora rabbit',),
     ('n02342885 hamster',),
     ('n02346627 porcupine, hedgehog',),
     ('n02356798 fox squirrel, eastern fox squirrel, Sciurus niger',),
     ('n02361337 marmot',),
     ('n02363005 beaver',),
     ('n02364673 guinea pig, Cavia cobaya',),
     ('n02389026 sorrel',),
     ('n02391049 zebra',),
     ('n02395406 hog, pig, grunter, squealer, Sus scrofa',),
     ('n02396427 wild boar, boar, Sus scrofa',),
     ('n02397096 warthog',),
     ('n02398521 hippopotamus, hippo, river horse, Hippopotamus amphibius',),
     ('n02403003 ox',),
     ('n02408429 water buffalo, water ox, Asiatic buffalo, Bubalus bubalis',),
     ('n02410509 bison',),
     ('n02412080 ram, tup',),
     ('n02415577 bighorn, bighorn sheep, cimarron, Rocky Mountain bighorn, Rocky Mountain sheep, Ovis canadensis',),
     ('n02417914 ibex, Capra ibex',),
     ('n02422106 hartebeest',),
     ('n02422699 impala, Aepyceros melampus',),
     ('n02423022 gazelle',),
     ('n02437312 Arabian camel, dromedary, Camelus dromedarius',),
     ('n02437616 llama',),
     ('n02441942 weasel',),
     ('n02442845 mink',),
     ('n02443114 polecat, fitch, foulmart, foumart, Mustela putorius',),
     ('n02443484 black-footed ferret, ferret, Mustela nigripes',),
     ('n02444819 otter',),
     ('n02445715 skunk, polecat, wood pussy',),
     ('n02447366 badger',),
     ('n02454379 armadillo',),
     ('n02457408 three-toed sloth, ai, Bradypus tridactylus',),
     ('n02480495 orangutan, orang, orangutang, Pongo pygmaeus',),
     ('n02480855 gorilla, Gorilla gorilla',),
     ('n02481823 chimpanzee, chimp, Pan troglodytes',),
     ('n02483362 gibbon, Hylobates lar',),
     ('n02483708 siamang, Hylobates syndactylus, Symphalangus syndactylus',),
     ('n02484975 guenon, guenon monkey',),
     ('n02486261 patas, hussar monkey, Erythrocebus patas',),
     ('n02486410 baboon',),
     ('n02487347 macaque',),
     ('n02488291 langur',),
     ('n02488702 colobus, colobus monkey',),
     ('n02489166 proboscis monkey, Nasalis larvatus',),
     ('n02490219 marmoset',),
     ('n02492035 capuchin, ringtail, Cebus capucinus',),
     ('n02492660 howler monkey, howler',),
     ('n02493509 titi, titi monkey',),
     ('n02493793 spider monkey, Ateles geoffroyi',),
     ('n02494079 squirrel monkey, Saimiri sciureus',),
     ('n02497673 Madagascar cat, ring-tailed lemur, Lemur catta',),
     ('n02500267 indri, indris, Indri indri, Indri brevicaudatus',),
     ('n02504013 Indian elephant, Elephas maximus',),
     ('n02504458 African elephant, Loxodonta africana',),
     ('n02509815 lesser panda, red panda, panda, bear cat, cat bear, Ailurus fulgens',),
     ('n02510455 giant panda, panda, panda bear, coon bear, Ailuropoda melanoleuca',),
     ('n02514041 barracouta, snoek',),
     ('n02526121 eel',),
     ('n02536864 coho, cohoe, coho salmon, blue jack, silver salmon, Oncorhynchus kisutch',),
     ('n02606052 rock beauty, Holocanthus tricolor',),
     ('n02607072 anemone fish',),
     ('n02640242 sturgeon',),
     ('n02641379 gar, garfish, garpike, billfish, Lepisosteus osseus',),
     ('n02643566 lionfish',),
     ('n02655020 puffer, pufferfish, blowfish, globefish',),
     ('n02666196 abacus',),
     ('n02667093 abaya',),
     ("n02669723 academic gown, academic robe, judge's robe",),
     ('n02672831 accordion, piano accordion, squeeze box',),
     ('n02676566 acoustic guitar',),
     ('n02687172 aircraft carrier, carrier, flattop, attack aircraft carrier',),
     ('n02690373 airliner',),
     ('n02692877 airship, dirigible',),
     ('n02699494 altar',),
     ('n02701002 ambulance',),
     ('n02704792 amphibian, amphibious vehicle',),
     ('n02708093 analog clock',),
     ('n02727426 apiary, bee house',),
     ('n02730930 apron',),
     ('n02747177 ashcan, trash can, garbage can, wastebin, ash bin, ash-bin, ashbin, dustbin, trash barrel, trash bin',),
     ('n02749479 assault rifle, assault gun',),
     ('n02769748 backpack, back pack, knapsack, packsack, rucksack, haversack',),
     ('n02776631 bakery, bakeshop, bakehouse',),
     ('n02777292 balance beam, beam',),
     ('n02782093 balloon',),
     ('n02783161 ballpoint, ballpoint pen, ballpen, Biro',),
     ('n02786058 Band Aid',),
     ('n02787622 banjo',),
     ('n02788148 bannister, banister, balustrade, balusters, handrail',),
     ('n02790996 barbell',),
     ('n02791124 barber chair',),
     ('n02791270 barbershop',),
     ('n02793495 barn',),
     ('n02794156 barometer',),
     ('n02795169 barrel, cask',),
     ('n02797295 barrow, garden cart, lawn cart, wheelbarrow',),
     ('n02799071 baseball',),
     ('n02802426 basketball',),
     ('n02804414 bassinet',),
     ('n02804610 bassoon',),
     ('n02807133 bathing cap, swimming cap',),
     ('n02808304 bath towel',),
     ('n02808440 bathtub, bathing tub, bath, tub',),
     ('n02814533 beach wagon, station wagon, wagon, estate car, beach waggon, station waggon, waggon',),
     ('n02814860 beacon, lighthouse, beacon light, pharos',),
     ('n02815834 beaker',),
     ('n02817516 bearskin, busby, shako',),
     ('n02823428 beer bottle',),
     ('n02823750 beer glass',),
     ('n02825657 bell cote, bell cot',),
     ('n02834397 bib',),
     ('n02835271 bicycle-built-for-two, tandem bicycle, tandem',),
     ('n02837789 bikini, two-piece',),
     ('n02840245 binder, ring-binder',),
     ('n02841315 binoculars, field glasses, opera glasses',),
     ('n02843684 birdhouse',),
     ('n02859443 boathouse',),
     ('n02860847 bobsled, bobsleigh, bob',),
     ('n02865351 bolo tie, bolo, bola tie, bola',),
     ('n02869837 bonnet, poke bonnet',),
     ('n02870880 bookcase',),
     ('n02871525 bookshop, bookstore, bookstall',),
     ('n02877765 bottlecap',),
     ('n02879718 bow',),
     ('n02883205 bow tie, bow-tie, bowtie',),
     ('n02892201 brass, memorial tablet, plaque',),
     ('n02892767 brassiere, bra, bandeau',),
     ('n02894605 breakwater, groin, groyne, mole, bulwark, seawall, jetty',),
     ('n02895154 breastplate, aegis, egis',),
     ('n02906734 broom',),
     ('n02909870 bucket, pail',),
     ('n02910353 buckle',),
     ('n02916936 bulletproof vest',),
     ('n02917067 bullet train, bullet',),
     ('n02927161 butcher shop, meat market',),
     ('n02930766 cab, hack, taxi, taxicab',),
     ('n02939185 caldron, cauldron',),
     ('n02948072 candle, taper, wax light',),
     ('n02950826 cannon',),
     ('n02951358 canoe',),
     ('n02951585 can opener, tin opener',),
     ('n02963159 cardigan',),
     ('n02965783 car mirror',),
     ('n02966193 carousel, carrousel, merry-go-round, roundabout, whirligig',),
     ("n02966687 carpenter's kit, tool kit",),
     ('n02971356 carton',),
     ('n02974003 car wheel',),
     ('n02977058 cash machine, cash dispenser, automated teller machine, automatic teller machine, automated teller, automatic teller, ATM',),
     ('n02978881 cassette',),
     ('n02979186 cassette player',),
     ('n02980441 castle',),
     ('n02981792 catamaran',),
     ('n02988304 CD player',),
     ('n02992211 cello, violoncello',),
     ('n02992529 cellular telephone, cellular phone, cellphone, cell, mobile phone',),
     ('n02999410 chain',),
     ('n03000134 chainlink fence',),
     ('n03000247 chain mail, ring mail, mail, chain armor, chain armour, ring armor, ring armour',),
     ('n03000684 chain saw, chainsaw',),
     ('n03014705 chest',),
     ('n03016953 chiffonier, commode',),
     ('n03017168 chime, bell, gong',),
     ('n03018349 china cabinet, china closet',),
     ('n03026506 Christmas stocking',),
     ('n03028079 church, church building',),
     ('n03032252 cinema, movie theater, movie theatre, movie house, picture palace',),
     ('n03041632 cleaver, meat cleaver, chopper',),
     ('n03042490 cliff dwelling',),
     ('n03045698 cloak',),
     ('n03047690 clog, geta, patten, sabot',),
     ('n03062245 cocktail shaker',),
     ('n03063599 coffee mug',),
     ('n03063689 coffeepot',),
     ('n03065424 coil, spiral, volute, whorl, helix',),
     ('n03075370 combination lock',),
     ('n03085013 computer keyboard, keypad',),
     ('n03089624 confectionery, confectionary, candy store',),
     ('n03095699 container ship, containership, container vessel',),
     ('n03100240 convertible',),
     ('n03109150 corkscrew, bottle screw',),
     ('n03110669 cornet, horn, trumpet, trump',),
     ('n03124043 cowboy boot',),
     ('n03124170 cowboy hat, ten-gallon hat',),
     ('n03125729 cradle',),
     ('n03126707 crane',),
     ('n03127747 crash helmet',),
     ('n03127925 crate',),
     ('n03131574 crib, cot',),
     ('n03133878 Crock Pot',),
     ('n03134739 croquet ball',),
     ('n03141823 crutch',),
     ('n03146219 cuirass',),
     ('n03160309 dam, dike, dyke',),
     ('n03179701 desk',),
     ('n03180011 desktop computer',),
     ('n03187595 dial telephone, dial phone',),
     ('n03188531 diaper, nappy, napkin',),
     ('n03196217 digital clock',),
     ('n03197337 digital watch',),
     ('n03201208 dining table, board',),
     ('n03207743 dishrag, dishcloth',),
     ('n03207941 dishwasher, dish washer, dishwashing machine',),
     ('n03208938 disk brake, disc brake',),
     ('n03216828 dock, dockage, docking facility',),
     ('n03218198 dogsled, dog sled, dog sleigh',),
     ('n03220513 dome',),
     ('n03223299 doormat, welcome mat',),
     ('n03240683 drilling platform, offshore rig',),
     ('n03249569 drum, membranophone, tympan',),
     ('n03250847 drumstick',),
     ('n03255030 dumbbell',),
     ('n03259280 Dutch oven',),
     ('n03271574 electric fan, blower',),
     ('n03272010 electric guitar',),
     ('n03272562 electric locomotive',),
     ('n03290653 entertainment center',),
     ('n03291819 envelope',),
     ('n03297495 espresso maker',),
     ('n03314780 face powder',),
     ('n03325584 feather boa, boa',),
     ('n03337140 file, file cabinet, filing cabinet',),
     ('n03344393 fireboat',),
     ('n03345487 fire engine, fire truck',),
     ('n03347037 fire screen, fireguard',),
     ('n03355925 flagpole, flagstaff',),
     ('n03372029 flute, transverse flute',),
     ('n03376595 folding chair',),
     ('n03379051 football helmet',),
     ('n03384352 forklift',),
     ('n03388043 fountain',),
     ('n03388183 fountain pen',),
     ('n03388549 four-poster',),
     ('n03393912 freight car',),
     ('n03394916 French horn, horn',),
     ('n03400231 frying pan, frypan, skillet',),
     ('n03404251 fur coat',),
     ('n03417042 garbage truck, dustcart',),
     ('n03424325 gasmask, respirator, gas helmet',),
     ('n03425413 gas pump, gasoline pump, petrol pump, island dispenser',),
     ('n03443371 goblet',),
     ('n03444034 go-kart',),
     ('n03445777 golf ball',),
     ('n03445924 golfcart, golf cart',),
     ('n03447447 gondola',),
     ('n03447721 gong, tam-tam',),
     ('n03450230 gown',),
     ('n03452741 grand piano, grand',),
     ('n03457902 greenhouse, nursery, glasshouse',),
     ('n03459775 grille, radiator grille',),
     ('n03461385 grocery store, grocery, food market, market',),
     ('n03467068 guillotine',),
     ('n03476684 hair slide',),
     ('n03476991 hair spray',),
     ('n03478589 half track',),
     ('n03481172 hammer',),
     ('n03482405 hamper',),
     ('n03483316 hand blower, blow dryer, blow drier, hair dryer, hair drier',),
     ('n03485407 hand-held computer, hand-held microcomputer',),
     ('n03485794 handkerchief, hankie, hanky, hankey',),
     ('n03492542 hard disc, hard disk, fixed disk',),
     ('n03494278 harmonica, mouth organ, harp, mouth harp',),
     ('n03495258 harp',),
     ('n03496892 harvester, reaper',),
     ('n03498962 hatchet',),
     ('n03527444 holster',),
     ('n03529860 home theater, home theatre',),
     ('n03530642 honeycomb',),
     ('n03532672 hook, claw',),
     ('n03534580 hoopskirt, crinoline',),
     ('n03535780 horizontal bar, high bar',),
     ('n03538406 horse cart, horse-cart',),
     ('n03544143 hourglass',),
     ('n03584254 iPod',),
     ('n03584829 iron, smoothing iron',),
     ("n03590841 jack-o'-lantern",),
     ('n03594734 jean, blue jean, denim',),
     ('n03594945 jeep, landrover',),
     ('n03595614 jersey, T-shirt, tee shirt',),
     ('n03598930 jigsaw puzzle',),
     ('n03599486 jinrikisha, ricksha, rickshaw',),
     ('n03602883 joystick',),
     ('n03617480 kimono',),
     ('n03623198 knee pad',),
     ('n03627232 knot',),
     ('n03630383 lab coat, laboratory coat',),
     ('n03633091 ladle',),
     ('n03637318 lampshade, lamp shade',),
     ('n03642806 laptop, laptop computer',),
     ('n03649909 lawn mower, mower',),
     ('n03657121 lens cap, lens cover',),
     ('n03658185 letter opener, paper knife, paperknife',),
     ('n03661043 library',),
     ('n03662601 lifeboat',),
     ('n03666591 lighter, light, igniter, ignitor',),
     ('n03670208 limousine, limo',),
     ('n03673027 liner, ocean liner',),
     ('n03676483 lipstick, lip rouge',),
     ('n03680355 Loafer',),
     ('n03690938 lotion',),
     ('n03691459 loudspeaker, speaker, speaker unit, loudspeaker system, speaker system',),
     ("n03692522 loupe, jeweler's loupe",),
     ('n03697007 lumbermill, sawmill',),
     ('n03706229 magnetic compass',),
     ('n03709823 mailbag, postbag',),
     ('n03710193 mailbox, letter box',),
     ('n03710637 maillot',),
     ('n03710721 maillot, tank suit',),
     ('n03717622 manhole cover',),
     ('n03720891 maraca',),
     ('n03721384 marimba, xylophone',),
     ('n03724870 mask',),
     ('n03729826 matchstick',),
     ('n03733131 maypole',),
     ('n03733281 maze, labyrinth',),
     ('n03733805 measuring cup',),
     ('n03742115 medicine chest, medicine cabinet',),
     ('n03743016 megalith, megalithic structure',),
     ('n03759954 microphone, mike',),
     ('n03761084 microwave, microwave oven',),
     ('n03763968 military uniform',),
     ('n03764736 milk can',),
     ('n03769881 minibus',),
     ('n03770439 miniskirt, mini',),
     ('n03770679 minivan',),
     ('n03773504 missile',),
     ('n03775071 mitten',),
     ('n03775546 mixing bowl',),
     ('n03776460 mobile home, manufactured home',),
     ('n03777568 Model T',),
     ('n03777754 modem',),
     ('n03781244 monastery',),
     ('n03782006 monitor',),
     ('n03785016 moped',),
     ('n03786901 mortar',),
     ('n03787032 mortarboard',),
     ('n03788195 mosque',),
     ('n03788365 mosquito net',),
     ('n03791053 motor scooter, scooter',),
     ('n03792782 mountain bike, all-terrain bike, off-roader',),
     ('n03792972 mountain tent',),
     ('n03793489 mouse, computer mouse',),
     ('n03794056 mousetrap',),
     ('n03796401 moving van',),
     ('n03803284 muzzle',),
     ('n03804744 nail',),
     ('n03814639 neck brace',),
     ('n03814906 necklace',),
     ('n03825788 nipple',),
     ('n03832673 notebook, notebook computer',),
     ('n03837869 obelisk',),
     ('n03838899 oboe, hautboy, hautbois',),
     ('n03840681 ocarina, sweet potato',),
     ('n03841143 odometer, hodometer, mileometer, milometer',),
     ('n03843555 oil filter',),
     ('n03854065 organ, pipe organ',),
     ('n03857828 oscilloscope, scope, cathode-ray oscilloscope, CRO',),
     ('n03866082 overskirt',),
     ('n03868242 oxcart',),
     ('n03868863 oxygen mask',),
     ('n03871628 packet',),
     ('n03873416 paddle, boat paddle',),
     ('n03874293 paddlewheel, paddle wheel',),
     ('n03874599 padlock',),
     ('n03876231 paintbrush',),
     ("n03877472 pajama, pyjama, pj's, jammies",),
     ('n03877845 palace',),
     ('n03884397 panpipe, pandean pipe, syrinx',),
     ('n03887697 paper towel',),
     ('n03888257 parachute, chute',),
     ('n03888605 parallel bars, bars',),
     ('n03891251 park bench',),
     ('n03891332 parking meter',),
     ('n03895866 passenger car, coach, carriage',),
     ('n03899768 patio, terrace',),
     ('n03902125 pay-phone, pay-station',),
     ('n03903868 pedestal, plinth, footstall',),
     ('n03908618 pencil box, pencil case',),
     ('n03908714 pencil sharpener',),
     ('n03916031 perfume, essence',),
     ('n03920288 Petri dish',),
     ('n03924679 photocopier',),
     ('n03929660 pick, plectrum, plectron',),
     ('n03929855 pickelhaube',),
     ('n03930313 picket fence, paling',),
     ('n03930630 pickup, pickup truck',),
     ('n03933933 pier',),
     ('n03935335 piggy bank, penny bank',),
     ('n03937543 pill bottle',),
     ('n03938244 pillow',),
     ('n03942813 ping-pong ball',),
     ('n03944341 pinwheel',),
     ('n03947888 pirate, pirate ship',),
     ('n03950228 pitcher, ewer',),
     ("n03954731 plane, carpenter's plane, woodworking plane",),
     ('n03956157 planetarium',),
     ('n03958227 plastic bag',),
     ('n03961711 plate rack',),
     ('n03967562 plow, plough',),
     ("n03970156 plunger, plumber's helper",),
     ('n03976467 Polaroid camera, Polaroid Land camera',),
     ('n03976657 pole',),
     ('n03977966 police van, police wagon, paddy wagon, patrol wagon, wagon, black Maria',),
     ('n03980874 poncho',),
     ('n03982430 pool table, billiard table, snooker table',),
     ('n03983396 pop bottle, soda bottle',),
     ('n03991062 pot, flowerpot',),
     ("n03992509 potter's wheel",),
     ('n03995372 power drill',),
     ('n03998194 prayer rug, prayer mat',),
     ('n04004767 printer',),
     ('n04005630 prison, prison house',),
     ('n04008634 projectile, missile',),
     ('n04009552 projector',),
     ('n04019541 puck, hockey puck',),
     ('n04023962 punching bag, punch bag, punching ball, punchball',),
     ('n04026417 purse',),
     ('n04033901 quill, quill pen',),
     ('n04033995 quilt, comforter, comfort, puff',),
     ('n04037443 racer, race car, racing car',),
     ('n04039381 racket, racquet',),
     ('n04040759 radiator',),
     ('n04041544 radio, wireless',),
     ('n04044716 radio telescope, radio reflector',),
     ('n04049303 rain barrel',),
     ('n04065272 recreational vehicle, RV, R.V.',),
     ('n04067472 reel',),
     ('n04069434 reflex camera',),
     ('n04070727 refrigerator, icebox',),
     ('n04074963 remote control, remote',),
     ('n04081281 restaurant, eating house, eating place, eatery',),
     ('n04086273 revolver, six-gun, six-shooter',),
     ('n04090263 rifle',),
     ('n04099969 rocking chair, rocker',),
     ('n04111531 rotisserie',),
     ('n04116512 rubber eraser, rubber, pencil eraser',),
     ('n04118538 rugby ball',),
     ('n04118776 rule, ruler',),
     ('n04120489 running shoe',),
     ('n04125021 safe',),
     ('n04127249 safety pin',),
     ('n04131690 saltshaker, salt shaker',),
     ('n04133789 sandal',),
     ('n04136333 sarong',),
     ('n04141076 sax, saxophone',),
     ('n04141327 scabbard',),
     ('n04141975 scale, weighing machine',),
     ('n04146614 school bus',),
     ('n04147183 schooner',),
     ('n04149813 scoreboard',),
     ('n04152593 screen, CRT screen',),
     ('n04153751 screw',),
     ('n04154565 screwdriver',),
     ('n04162706 seat belt, seatbelt',),
     ('n04179913 sewing machine',),
     ('n04192698 shield, buckler',),
     ('n04200800 shoe shop, shoe-shop, shoe store',),
     ('n04201297 shoji',),
     ('n04204238 shopping basket',),
     ('n04204347 shopping cart',),
     ('n04208210 shovel',),
     ('n04209133 shower cap',),
     ('n04209239 shower curtain',),
     ('n04228054 ski',),
     ('n04229816 ski mask',),
     ('n04235860 sleeping bag',),
     ('n04238763 slide rule, slipstick',),
     ('n04239074 sliding door',),
     ('n04243546 slot, one-armed bandit',),
     ('n04251144 snorkel',),
     ('n04252077 snowmobile',),
     ('n04252225 snowplow, snowplough',),
     ('n04254120 soap dispenser',),
     ('n04254680 soccer ball',),
     ('n04254777 sock',),
     ('n04258138 solar dish, solar collector, solar furnace',),
     ('n04259630 sombrero',),
     ('n04263257 soup bowl',),
     ('n04264628 space bar',),
     ('n04265275 space heater',),
     ('n04266014 space shuttle',),
     ('n04270147 spatula',),
     ('n04273569 speedboat',),
     ("n04275548 spider web, spider's web",),
     ('n04277352 spindle',),
     ('n04285008 sports car, sport car',),
     ('n04286575 spotlight, spot',),
     ('n04296562 stage',),
     ('n04310018 steam locomotive',),
     ('n04311004 steel arch bridge',),
     ('n04311174 steel drum',),
     ('n04317175 stethoscope',),
     ('n04325704 stole',),
     ('n04326547 stone wall',),
     ('n04328186 stopwatch, stop watch',),
     ('n04330267 stove',),
     ('n04332243 strainer',),
     ('n04335435 streetcar, tram, tramcar, trolley, trolley car',),
     ('n04336792 stretcher',),
     ('n04344873 studio couch, day bed',),
     ('n04346328 stupa, tope',),
     ('n04347754 submarine, pigboat, sub, U-boat',),
     ('n04350905 suit, suit of clothes',),
     ('n04355338 sundial',),
     ('n04355933 sunglass',),
     ('n04356056 sunglasses, dark glasses, shades',),
     ('n04357314 sunscreen, sunblock, sun blocker',),
     ('n04366367 suspension bridge',),
     ('n04367480 swab, swob, mop',),
     ('n04370456 sweatshirt',),
     ('n04371430 swimming trunks, bathing trunks',),
     ('n04371774 swing',),
     ('n04372370 switch, electric switch, electrical switch',),
     ('n04376876 syringe',),
     ('n04380533 table lamp',),
     ('n04389033 tank, army tank, armored combat vehicle, armoured combat vehicle',),
     ('n04392985 tape player',),
     ('n04398044 teapot',),
     ('n04399382 teddy, teddy bear',),
     ('n04404412 television, television system',),
     ('n04409515 tennis ball',),
     ('n04417672 thatch, thatched roof',),
     ('n04418357 theater curtain, theatre curtain',),
     ('n04423845 thimble',),
     ('n04428191 thresher, thrasher, threshing machine',),
     ('n04429376 throne',),
     ('n04435653 tile roof',),
     ('n04442312 toaster',),
     ('n04443257 tobacco shop, tobacconist shop, tobacconist',),
     ('n04447861 toilet seat',),
     ('n04456115 torch',),
     ('n04458633 totem pole',),
     ('n04461696 tow truck, tow car, wrecker',),
     ('n04462240 toyshop',),
     ('n04465501 tractor',),
     ('n04467665 trailer truck, tractor trailer, trucking rig, rig, articulated lorry, semi',),
     ('n04476259 tray',),
     ('n04479046 trench coat',),
     ('n04482393 tricycle, trike, velocipede',),
     ('n04483307 trimaran',),
     ('n04485082 tripod',),
     ('n04486054 triumphal arch',),
     ('n04487081 trolleybus, trolley coach, trackless trolley',),
     ('n04487394 trombone',),
     ('n04493381 tub, vat',),
     ('n04501370 turnstile',),
     ('n04505470 typewriter keyboard',),
     ('n04507155 umbrella',),
     ('n04509417 unicycle, monocycle',),
     ('n04515003 upright, upright piano',),
     ('n04517823 vacuum, vacuum cleaner',),
     ('n04522168 vase',),
     ('n04523525 vault',),
     ('n04525038 velvet',),
     ('n04525305 vending machine',),
     ('n04532106 vestment',),
     ('n04532670 viaduct',),
     ('n04536866 violin, fiddle',),
     ('n04540053 volleyball',),
     ('n04542943 waffle iron',),
     ('n04548280 wall clock',),
     ('n04548362 wallet, billfold, notecase, pocketbook',),
     ('n04550184 wardrobe, closet, press',),
     ('n04552348 warplane, military plane',),
     ('n04553703 washbasin, handbasin, washbowl, lavabo, wash-hand basin',),
     ('n04554684 washer, automatic washer, washing machine',),
     ('n04557648 water bottle',),
     ('n04560804 water jug',),
     ('n04562935 water tower',),
     ('n04579145 whiskey jug',),
     ('n04579432 whistle',),
     ('n04584207 wig',),
     ('n04589890 window screen',),
     ('n04590129 window shade',),
     ('n04591157 Windsor tie',),
     ('n04591713 wine bottle',),
     ('n04592741 wing',),
     ('n04596742 wok',),
     ('n04597913 wooden spoon',),
     ('n04599235 wool, woolen, woollen',),
     ('n04604644 worm fence, snake fence, snake-rail fence, Virginia fence',),
     ('n04606251 wreck',),
     ('n04612504 yawl',),
     ('n04613696 yurt',),
     ('n06359193 web site, website, internet site, site',),
     ('n06596364 comic book',),
     ('n06785654 crossword puzzle, crossword',),
     ('n06794110 street sign',),
     ('n06874185 traffic light, traffic signal, stoplight',),
     ('n07248320 book jacket, dust cover, dust jacket, dust wrapper',),
     ('n07565083 menu',),
     ('n07579787 plate',),
     ('n07583066 guacamole',),
     ('n07584110 consomme',),
     ('n07590611 hot pot, hotpot',),
     ('n07613480 trifle',),
     ('n07614500 ice cream, icecream',),
     ('n07615774 ice lolly, lolly, lollipop, popsicle',),
     ('n07684084 French loaf',),
     ('n07693725 bagel, beigel',),
     ('n07695742 pretzel',),
     ('n07697313 cheeseburger',),
     ('n07697537 hotdog, hot dog, red hot',),
     ('n07711569 mashed potato',),
     ('n07714571 head cabbage',),
     ('n07714990 broccoli',),
     ('n07715103 cauliflower',),
     ('n07716358 zucchini, courgette',),
     ('n07716906 spaghetti squash',),
     ('n07717410 acorn squash',),
     ('n07717556 butternut squash',),
     ('n07718472 cucumber, cuke',),
     ('n07718747 artichoke, globe artichoke',),
     ('n07720875 bell pepper',),
     ('n07730033 cardoon',),
     ('n07734744 mushroom',),
     ('n07742313 Granny Smith',),
     ('n07745940 strawberry',),
     ('n07747607 orange',),
     ('n07749582 lemon',),
     ('n07753113 fig',),
     ('n07753275 pineapple, ananas',),
     ('n07753592 banana',),
     ('n07754684 jackfruit, jak, jack',),
     ('n07760859 custard apple',),
     ('n07768694 pomegranate',),
     ('n07802026 hay',),
     ('n07831146 carbonara',),
     ('n07836838 chocolate sauce, chocolate syrup',),
     ('n07860988 dough',),
     ('n07871810 meat loaf, meatloaf',),
     ('n07873807 pizza, pizza pie',),
     ('n07875152 potpie',),
     ('n07880968 burrito',),
     ('n07892512 red wine',),
     ('n07920052 espresso',),
     ('n07930864 cup',),
     ('n07932039 eggnog',),
     ('n09193705 alp',),
     ('n09229709 bubble',),
     ('n09246464 cliff, drop, drop-off',),
     ('n09256479 coral reef',),
     ('n09288635 geyser',),
     ('n09332890 lakeside, lakeshore',),
     ('n09399592 promontory, headland, head, foreland',),
     ('n09421951 sandbar, sand bar',),
     ('n09428293 seashore, coast, seacoast, sea-coast',),
     ('n09468604 valley, vale',),
     ('n09472597 volcano',),
     ('n09835506 ballplayer, baseball player',),
     ('n10148035 groom, bridegroom',),
     ('n10565667 scuba diver',),
     ('n11879895 rapeseed',),
     ('n11939491 daisy',),
     ("n12057211 yellow lady's slipper, yellow lady-slipper, Cypripedium calceolus, Cypripedium parviflorum",),
     ('n12144580 corn',),
     ('n12267677 acorn',),
     ('n12620546 hip, rose hip, rosehip',),
     ('n12768682 buckeye, horse chestnut, conker',),
     ('n12985857 coral fungus',),
     ('n12998815 agaric',),
     ('n13037406 gyromitra',),
     ('n13040303 stinkhorn, carrion fungus',),
     ('n13044778 earthstar',),
     ('n13052670 hen-of-the-woods, hen of the woods, Polyporus frondosus, Grifola frondosa',),
     ('n13054560 bolete',),
     ('n13133613 ear, spike, capitulum',),
     ('n15075141 toilet tissue, toilet paper, bathroom tissue',)]


