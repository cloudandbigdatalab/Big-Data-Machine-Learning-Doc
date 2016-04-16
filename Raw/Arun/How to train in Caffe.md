
# How to train your own network in Caffe

The main files, apart from the dataset, required to train your network are the model definitions and the solver definitions. These files are saved in a Google Protobuf format as .prototxt files. It is similar to a yaml file.

The model definition file defines the architecture of your neural net. The number of layers and its descriptions are to be written in them. The solver definition file is where you specify the learning rate, momentum, snapshot interval and a whole host of other key parameters required for training and testing your neural network. Please view the [Caffe: Things to know to train your network](https://github.com/arundasan91/Caffe/blob/master/Caffe_Things_to_know.md) file for more info. 

The data is as important as anything else and is to be preprocessed to one of the formats recognized by Caffe. lmdb formats works well with Caffe. Caffe also support hdf5 formated data and image files. If the data you want the neural net to be trained on is not a Caffe default, you can write your own Class for the respective type and include the proper layer.

Once you have the Data, ModelParameter and SolverParameter files, you can train it by going into caffe root directory and executing the following command:

```
./build/tools/caffe train --solver=/path/to/solver.prototxt
```

You can write your own python code to do the same thing by importing the caffe library. You can even write functions to create prototxt files for you according to the parameters you pass to it. If you have jupyter notebook installed in your system this becomes a fairly easy task. Also, many of Caffe's examples are provided in a Notebook format so that you can run it in your system learn on the go.

There are other ways also which I am learning/exploring. As of now, to sum up:

1. Define your network in a prototxt format by writing your own or using python code.
2. Define the solver parameters in a prototxt format.
3. Define, preprocess and ready your data.
4. Initialize training by specifying your two prototxt files.

Note that you can always resume your training with the snapshotted caffemodel files. To do this you have to specify the solverstate file while you want to use while training. Solverstate file is generated along with the caffemodel file while snaoshotting the trained neural network. An example:

```
./build/tools/caffe train --solver=/path/to/solver.prototxt --snapshot=/path/to/caffe_n_iter.solverstate
```

Learn from mistakes.
Happy Coding !
