1. **Data Injestion and Preprocessing**
 * Data Injestion formats
    * Level DB or LMDB database.
    * Datat in memory (C++ or Python).
    * HDF5 formated data.
    * Image files.
 * **Preprocessing Tools**
     `~ Can be found at $CAFFE_ROOT/build/tools`
    * LevelDB or LMDB creation from raw images.
    * Training and validation set creation with shuffling algorithms.
    * Mean image generation.
 * **Data Transformation Tools**
    * Image Cropping, Scaling, Resizing and mirroring.
    * Mean Subtration.
    
2. **Model Defenition**
 * Defined in a prototxt format.
 * Prototxt is a human readable format developed by Google.
 * It autogenerates and checks Caffe code.
 * Used to define Caffe's network architecture and training parameters.
 * Can be written by hand without any coding.
 * Can be written as a python function also\. It autogenerates the prototxt file for you.
 * Example Model Defenition prototxt file:
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
     
     add more layers by simply mentioning them in the same format 
     
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
     
     end the prototxt file after mentioning all the layers
     
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
     ```
 * Example python code to define the model
     
     ```
     from caffe import layers as L, params as P

     def lenet(lmdb, batch_size):
     #our version of LeNet: a series of linear and simple nonlinear transformations
     n = caffe.NetSpec()
    
     n.data, n.label = L.Data(batch_size=batch_size, backend=P.Data.LMDB, source=lmdb,
                             transform_param=dict(scale=1./255), ntop=2)
    
     n.conv1 = L.Convolution(n.data, kernel_size=5, num_output=20, weight_filler=dict(type='xavier'))
     n.pool1 = L.Pooling(n.conv1, kernel_size=2, stride=2, pool=P.Pooling.MAX)
     n.conv2 = L.Convolution(n.pool1, kernel_size=5, num_output=50, weight_filler=dict(type='xavier'))
     n.pool2 = L.Pooling(n.conv2, kernel_size=2, stride=2, pool=P.Pooling.MAX)
     n.fc1 =   L.InnerProduct(n.pool2, num_output=500, weight_filler=dict(type='xavier'))
     n.relu1 = L.ReLU(n.fc1, in_place=True)
     n.score = L.InnerProduct(n.relu1, num_output=10, weight_filler=dict(type='xavier'))
     n.loss =  L.SoftmaxWithLoss(n.score, n.label)
    
     return n.to_proto()
     ```
     
    2.1 **Different fucntions and layers**
     * **Loss Functions**
         * Classification
             * Softmax
             * Hinge loss
         * Linear Regression
             * Euclidean Loss
         * Attributes/Multiclassification
             * Simoid cross entropy loss
     * **Layers**
         * Convolution
         * Pooling
         * Normalization
     * **Activation Functions**
         * ReLU
         * Sigmoid
         * Tanh
     
3. **Network Training - Solver Files**
 * Prototxt file to list the parameters of the neural net's training algorithm
 * Example Solver File:
     
     ```
    # The train/test net protocol buffer definition
    train_net: "mnist/lenet_auto_train.prototxt"
    test_net: "mnist/lenet_auto_test.prototxt"
    # test_iter specifies how many forward passes the test should carry out.
    # In the case of MNIST, we have test batch size 100 and 100 test iterations,
    # covering the full 10,000 testing images.
    test_iter: 100
    # Carry out testing every 500 training iterations.
    test_interval: 500
    # The base learning rate, momentum and the weight decay of the network.
    base_lr: 0.01
    momentum: 0.9
    weight_decay: 0.0005
    # The learning rate policy
    lr_policy: "inv"
    gamma: 0.0001
    power: 0.75
    # Display every 100 iterations
    display: 100
    # The maximum number of iterations
    max_iter: 10000
    # snapshot intermediate results
    snapshot: 5000
    snapshot_prefix: "mnist/lenet"
    ```
4. Optimization Algorithms
 * SGD + momentum
 * ADAGRAD
 * NAG

5. After training
 * Caffe training produces a binary file with extension *\.caffemodel*.
     * This is a machine readable file generally a few hundered mega bytes.
     * This model can be reused for further training and can be shared as well.
     * Caffemodel file can be used for implementation of the neural net.
 * Integrate the model into the data pipeline using Caffe command line or Matlab/Python.
 * Deploy model across hardware or OS environments with caffe installed.
 
 6. Share to ModelZoo
  * ModelZoo is a project/model sharing community.
  * Normally, when sharing the caffemodel, the following should be present:
      * Solver
      * Model Prototxt Files
      * readme.md file which describes the:
          * Caffe version.
          * URL and SHA1 of *\.caffemodel* file.
      * License
      * Description of the Training Data.
 
### Caffe: Extensible Code
Caffe's inbuilt data types, layer types or loss functions may not be directly relevant in our neural network architecture. In those cases, we can write our own datatype or layertype or loss function by writing specific a Python or C++ classes for the same. This way we can extend caffe's capabilities to meet our needs. The new layer or loss function class can now be properly used in the required prototxt file.

### Reference
1. nVIDIA QwikLab sessions, which can be found [here.](https://nvidia.qwiklab.com/)
2. Caffe's examples in $CAFFE_ROOT/examples/
