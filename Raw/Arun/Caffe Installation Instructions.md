
# <center> ![Caffe](https://developer.nvidia.com/sites/default/files/akamai/cuda/images/deeplearning/caffe.png) </center>
## <center>Freshly brewed !</center>

<b>W</b>ith the availability of huge amount of data for research and powerfull machines to run your code on, Machine Learning and Neural Networks is gaining their foot again and impacting us more than ever in our everyday lives. With huge players like Google opensourcing part of their Machine Learning systems like the TensorFlow software library for numerical computation, there are many options for someone interested in starting off with Machine Learning/Neural Nets to choose from. Caffe, a deep learning framework developed by the Berkeley Vision and Learning Center (BVLC) and its contributors, comes to the play with a fresh cup of coffee.

# Installation Instructions (Ubuntu 14 Trusty)

The following section is divided in to two parts. Caffe's [documentation]( http://caffe.berkeleyvision.org/installation.html "Caffe Installation Instruction") suggests you to install [Anaconda](https://www.continuum.io/why-anaconda "Anaconda") Python distribution to make sure that you've installed necessary packages, with ease. If you're someone who do not want to install Anaconda in your system for some reason, I've covered that too. So in the first part you'll find information on <i><b> [how to install Caffe with Anaconda](#1.-Caffe-+-Anaconda)</i></b> and in the second part you'll find the information for <i><b> installing Caffe without Anaconda </i></b>.

Please note that the following instructions were tested on my local machine and in two Chameleon Cloud Instances. However I cannot garuntee success for anyone. Please be ready to see some errors on the way, but I hope you won't stumble into any if you follow the directions as is.

My local machine and the instances I used are <b>NOT</b> equipped with GPU's. So the installation instrucions are strictly for non-GPU based or more clearly CPU-only systems running Ubuntu 14 trusty. However, to install it in a GPU based system, you just have to install CUDA and necessary drivers for your GPU. You can find the instructions in [Stack Overflow](http://stackoverflow.com/ "Stack Overflow") or in the always go to friend [Google](https://www.google.com/ "GOOGLE SEARCH").

# For systems without GPU's (CPU_only)

# 1. Caffe + Anaconda

Anaconda python distribution includes scientific and analytic Python packages which are extremely useful. The complete list of packages can be found [here](http://docs.continuum.io/anaconda/pkg-docs "Anaconda Packages"). 

To install Anaconda, you have to <b>first download the Installer to your machine</b>. Go to this [website](https://www.continuum.io/downloads "Scroll to the 'Anaconda for Linux' section") to download the Installer. Scroll to the 'Anaconda for Linux' section and choose the installer to download depending on your system architecture.

Once you have the Installer in your machine, run the following code to install Anaconda.

    bash Anaconda2-2.5.0-Linux-x86_64.sh

If you fail to read the few lines printed after installation, you'll waste a good amount of your produtive time on trying to figure out what went wrong. An important line reads:

<b>For this change to become active, you have to open a new terminal.</b>

So, once the Anaconda installation is over, please open a new terminal. Period. 

After opening a new terminal, to verify the installation type:

    conda -V

This should give you the current version of conda, thus verifying the installation. Now that's done !

Now we will install OpenBLAS.

    sudo apt-get install libopenblas-dev

Next go ahead and install Boost. More info on boost [here](http://www.boost.org/ "boost.org")

I faced a problem while installing boost in all my machines. I fixed it by including multiverse repository into the sources.list. Since playing with sources.list is not reccomended, follow the steps for a better alternative.


    echo 'deb http://archive.ubuntu.com/ubuntu trusty main restricted universe multiverse' >>/tmp/multiverse.list

    sudo cp /tmp/multiverse.list /etc/apt/sources.list.d/

    rm /tmp/multiverse.list

The repo is saved to a temporary list named 'multiverse.list' in the /tmp folder. It is then copied to /etc/apt/sources.list.d/ folder. The file in /tmp folder is then removed. I found this fix in [Stack Exchange fourm](http://stackoverflow.com/questions/1584066/append-to-etc-apt-sources-list).

Now to install boost, run:

    sudo apt-get install libboost-all-dev

Now, let us install OpenCV. Go ahead and run:

    conda install opencv

    sudo apt-get install libopencv-dev

Now let us install some dependencies of Caffe. Run the following:

    sudo apt-get install libleveldb-dev libsnappy-dev libhdf5-serial-dev

    sudo apt-get install libgflags-dev libgoogle-glog-dev liblmdb-dev

    sudo apt-get install protobuf-compiler libprotobuf-dev

    conda install -c https://conda.anaconda.org/anaconda protobuf

Okay, that's it. Let us now download the Caffe. If you don't have git installed in your system yet, run this code really quick:

    sudo apt-get install git

We will clone the official [Caffe repository](https://github.com/BVLC/caffe "Caffe GitHub repo") from Github.

    git clone https://github.com/BVLC/caffe

Once the git is cloned, cd into caffe folder.

    cd caffe

We will edit the configuration file of Caffe now. We need to do it to specify that we are using a CPU-only system. (Tell compiler to disable GPU, CUDA etc). For this, make a copy of the Makefile.config.example.

    cp Makefile.config.example Makefile.config

Great ! Now go ahead and open the Makefile.config in your favourite text editor (vi or vim or gedit or ...). Change the following:

    1. Uncomment (No space in the beginning): 
    CPU_ONLY := 1

    2. Change:
    BLAS := atlas to BLAS := open

    3. Comment out:
    PYTHON_INCLUDE := /usr/include/python2.7 \
            /usr/lib/python2.7/dist-packages/numpy/core/include
       
    4. Uncomment:
    ANACONDA_HOME := $(HOME)/anaconda2
    
    PYTHON_INCLUDE := $(ANACONDA_HOME)/include \
        $(ANACONDA_HOME)/include/python2.7 \
        $(ANACONDA_HOME)/lib/python2.7/site-packages/numpy/core/include 

    5. Comment:
    PYTHON_LIB := /usr/lib

    6. Uncomment:
    PYTHON_LIB := $(ANACONDA_HOME)/lib
    
    7. Uncomment:
    USE_PKG_CONFIG := 1

Your Makefile.config should look something like this now: [Makefile.config](#Makefile.config)

Now that's done, let me share with you an error I came across. Our Makefile.config is okay. But while 'make'-ing / building the installation files, the hf5 dependeny gave me an error. This might not apply to you. I can't say for sure. The build required two files <b>libhdf5_h1.so.10</b> and <b>libhd5.so.10</b> but the files in the system were <b>libhdf5_h1.so.7</b> and <b>libhd5.so.7</b>. I fixed this by doing the following:

    cd /usr/lib/x86_64-linux-gnu/

    sudo cp libhdf5_hl.so.7 libhdf5_hl.so.10

    sudo cp libhdf5.so.7 libhdf5.so.10

We will now install the libraries listed in the requirements.txt file.

    cd ~/caffe/python

    sudo apt-get install python-pip && sudo pip install -r requirements.txt

Now, we can safely build the files in the caffe directory. We will run the make process as 4 jobs by specifying it like <b>-j4</b>. More on it [here](http://www.tutorialspoint.com/unix_commands/make.htm " make command")

    cd ~/caffe

    sudo make all -j4

I hope the make process went well. If not, please see which package failed by checking the logs or from terminal itself. Feel free to comment, I will help to the best of my knowledge. You can seek help from your go to friend Google or Stack Exchange as mentioned above.

Provided that the make process was successfull, continue with the rest of the installation process.

We will now make the Pycaffe files. Pycaffe is the Python interface of Caffe which allows you to use Caffe inside Python. More on it [here](http://caffe.berkeleyvision.org/tutorial/interfaces.html). We will also make distribute. This is explained in Caffe website.

    sudo make pycaffe

    sudo make distribute

Awesome! We are almost there. We just need to test whether everything went fine. For that make the files for testing and run the test.

    sudo make test

    sudo make runtest

If you succeed in all the tests then you've successfully installed Caffe in your system ! One good reason to smile !

Finally, we need to add the correct path to our installed modules. Using your favourite text editor, add the following to the <b>.bashrc</b> file in your <b>/home/user/</b> folder for Caffe to work properly. Please make sure you replace the < username > with your system's username.

```
#Anaconda if not present already
export PATH=/home/<username>/anaconda2/bin:$PATH
#Caffe Root
export CAFFE_ROOT=/home/<username>/caffe/
export PYTHONPATH=/home/<username>/caffe/distribute/python:$PYTHONPATH
export PYTHONPATH=/home/<username>/caffe/python:$PYTHONPATH
```

CHEERS ! You're done ! Now let's test if it really works.

Restart/reboot your system to ensure everything loads perfect.

    sudo reboot

Open Python and type:

    import caffe

You should be able to successfully load caffe.
Now let's start coding :)

# 2. Caffe without installing Anaconda

By preference, if you don't want to install Anaconda in your system, you can install Caffe by following the steps below. As mentioned earlier, installing all the dependencies can be difficult. If this tutorial does not work for you, please look into the errors, use our trusted friends.

To start with, we will update and upgrade the packages in our system. Then we will have to install the dependencies one by one on the machine. Type the following to get started.

    sudo apt-get update && sudo apt-get upgrade && sudo apt-get dist-upgrade && sudo apt-get autoremove

Now, let us install openblas.

    sudo apt-get -y install libopenblas-dev

Next go ahead and install Boost. More info on boost [here](http://www.boost.org/ "boost.org")

I faced a problem while installing boost in all my machines. I fixed it by including multiverse repository into the sources.list. Since playing with sources.list is not reccomended, follow the steps for a better alternative.

    echo 'deb http://archive.ubuntu.com/ubuntu trusty main restricted universe multiverse' >>/tmp/multiverse.list

    sudo cp /tmp/multiverse.list /etc/apt/sources.list.d/

    rm /tmp/multiverse.list

The repo is saved to a temporary list named 'multiverse.list' in the /tmp folder. It is then copied to /etc/apt/sources.list.d/ folder. The file in /tmp folder is then removed. I found this fix in [Stack Exchange fourm](http://stackoverflow.com/questions/1584066/append-to-etc-apt-sources-list).

Now to install boost, run:

    sudo apt-get update && sudo apt-get install libboost-all-dev

If later in the installation process you find that any of the boost related files are missing, run the following command. You can skip this one for now but won't hurt if you do it either.

    sudo apt-get -y install --fix-missing libboost-all-dev

Go ahead and install libfaac-dev package.

    sudo apt-get install libfaac-dev

Now, we need to install ffmpeg. Let us also make sure that the ffmpeg version is one which OpenCV and Caffe approves. We will remove any previous versions of ffmpeg and install new ones.

The following code will remove ffmpeg and related packages:

    sudo apt-get -y remove ffmpeg x264 libx264-dev

The mc3man repository hosts ffmpeg packages. I came to know about it from Stack Exchange forums. To include the repo, type this:

    sudo add-apt-repository ppa:mc3man/trusty-media

Update and install ffmpeg.

    sudo apt-get update && sudo apt-get install ffmpeg gstreamer0.10-ffmpeg

Now, we can install OpenCV. First let us install the dependencies. Building OpenCV can be challenging at first, but if you have all the dependencies correct it will be done in no time.

Go ahead and run the following lines:

    sudo apt-get install build-essential

The 'build-essential' ensures that we have the compilers ready. Now we will install some required packages. Run:

    sudo apt-get install cmake git libgtk2.0-dev pkg-config libavcodec-dev libavformat-dev libswscale-dev unzip

We will install some optional packages as well. Run:

    sudo apt-get install python-dev python-numpy libtbb2 libtbb-dev libjpeg-dev libpng-dev libtiff-dev libjasper-dev libdc1394-22-dev

    sudo apt-get install libtiff4-dev libopenexr-dev libeigen2-dev yasm libopencore-amrnb-dev libtheora-dev libvorbis-dev libxvidcore-dev

    sudo apt-get install python-tk libeigen3-dev libx264-dev libqt4-dev libqt4-opengl-dev sphinx-common texlive-latex-extra libv4l-dev default-jdk ant libvtk5-qt4-dev

Now we can go ahead and download the OpenCV build files. Go to your root folder first.

    cd ~

Download the files:

    wget http://sourceforge.net/projects/opencvlibrary/files/opencv-unix/2.4.9/opencv-2.4.9.zip

Unzip the file by:

    unzip opencv-2.4.9.zip

Go to the opencv folder by running:

    cd opencv-2.4.9

Make a build directory inside.

    mkdir build

Go inside the build directory.

    cd build

Build the files using cmake.

    cmake -D WITH_TBB=ON -D BUILD_NEW_PYTHON_SUPPORT=ON -D WITH_V4L=ON -D INSTALL_C_EXAMPLES=ON -D INSTALL_PYTHON_EXAMPLES=ON -D BUILD_EXAMPLES=ON -D WITH_QT=ON -D WITH_OPENGL=ON -D WITH_VTK=ON ..

In the summary, make sure that FFMPEG is installed, also check whether the Python, Numpy, Java and OpenCL are properly installed and recognized.

Now we will run the make process as 4 jobs by specifying it like <b>-j4</b>. More on it [here](http://www.tutorialspoint.com/unix_commands/make.htm " make command")

    sudo make -j4

Go ahead and continue installation.

    sudo make install

Once the installation is complete, do these steps to get OpenCV configured.

    sudo sh -c 'echo "/usr/local/lib" > /etc/ld.so.conf.d/opencv.conf'

    sudo ldconfig

    export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/usr/local/opencv/lib

Come out of the build folder if you haven't already by running:

    cd ~

Install python-pip:

    sudo apt-get install python-pip

Now, we will install the Scipy and other scientific packages which are key Caffe dependencies.

    sudo apt-get install python-numpy python-scipy python-matplotlib ipython ipython-notebook python-pandas python-sympy python-nose

We will install Cython now. (I wanted it to install scikit-image properly)

    wget http://cython.org/release/Cython-0.23.4.zip

    unzip Cython-0.23.4.zip

    cd Cython-0.23.4

    sudo python setup.py install
    
    cd ~

Now that we have Cython, go ahead and run the code below to install Scikit Image and Scikit Learn.

    sudo pip install scikit-image scikit-learn

We will now install some more crucial dependencies of Caffe

    sudo apt-get install libprotobuf-dev libleveldb-dev libsnappy-dev libopencv-dev libhdf5-serial-dev

    sudo apt-get install libgflags-dev libgoogle-glog-dev liblmdb-dev protobuf-compiler

    sudo pip install protobuf

Installing Pydot will be beneficial to view our net by saving it off in an image file.
    
    sudo apt-get install python-pydot

Now that all the dependencies are installed, we will go ahead and download the Caffe installation files. Go ahead and run:

    git clone https://github.com/BVLC/caffe
   

Go into the caffe folder and copy and rename the Makefile.config.example file to Makefile.config.

    cd caffe
    cp Makefile.config.example Makefile.config

Great ! Now go ahead and open the Makefile.config in your favourite text editor (vi or vim or gedit or ...). Change the following:
```
1. Uncomment (No space in the beginning): 
    CPU_ONLY := 1

2. Uncomment:
    USE_PKG_CONFIG := 1
```

We will install the packages listed in Caffe's requirements.txt file as well; just in case.

    cd ~/caffe/python
    sudo pip install -r requirements.txt

Now, we can safely build the files in the caffe directory. We will run the make process as 4 jobs by specifying it like <b>-j4</b>. More on it [here](http://www.tutorialspoint.com/unix_commands/make.htm " make command")

    cd ~/caffe
    sudo make all -j4

I hope the make process went well. If not, please see which package failed by checking the logs or from terminal itself. Feel free to comment, I will help to the best of my knowledge. You can seek help from your go to friend Google or Stack Exchange as mentioned above.

Provided that the make process was successfull, continue with the rest of the installation process.

We will now make the Pycaffe files. Pycaffe is the Python interface of Caffe which allows you to use Caffe inside Python. More on it [here](http://caffe.berkeleyvision.org/tutorial/interfaces.html). We will also make distribute. This is explained in Caffe website.

    sudo make pycaffe
    sudo make distribute

Awesome! We are almost there. We just need to test whether everything went fine. For that make the files for testing and run the test.

    sudo make test
    sudo make runtest

If you succeed in all the tests then you've successfully installed Caffe in your system ! One good reason to smile !

Finally, we need to add the correct path to our installed modules. Using your favourite text editor, add the following to the <b>.bashrc</b> file in your <b>/home/user/</b> folder for Caffe to work properly. Please make sure you replace the < username > with your system's username.

```
# Caffe Root
export CAFFE_ROOT=/home/<username>/caffe/
export PYTHONPATH=/home/<username>/caffe/distribute/python:$PYTHONPATH
export PYTHONPATH=/home/<username>/caffe/python:$PYTHONPATH
```

CHEERS ! You're done ! Now let's test if it really works.

Restart/reboot your system to ensure everything loads perfect.

    sudo reboot

Open Python and type:

    import caffe

You should be able to successfully load caffe. Now let's start coding :)

# Appendix

## Makefile.config

### For Caffe + Anaconda

    ## Refer to http://caffe.berkeleyvision.org/installation.html
    # Contributions simplifying and improving our build system are welcome!
    
    # cuDNN acceleration switch (uncomment to build with cuDNN).
    # USE_CUDNN := 1

    # CPU-only switch (uncomment to build without GPU support).
    CPU_ONLY := 1

    # uncomment to disable IO dependencies and corresponding data layers
    # USE_OPENCV := 0
    # USE_LEVELDB := 0
    # USE_LMDB := 0
    
    # uncomment to allow MDB_NOLOCK when reading LMDB files (only if necessary)
    #	You should not set this flag if you will be reading LMDBs with any
    #	possibility of simultaneous read and write
    # ALLOW_LMDB_NOLOCK := 1

    # Uncomment if you're using OpenCV 3
    # OPENCV_VERSION := 3

    # To customize your choice of compiler, uncomment and set the following.
    # N.B. the default for Linux is g++ and the default for OSX is clang++
    # CUSTOM_CXX := g++
    
    # CUDA directory contains bin/ and lib/ directories that we need.
    CUDA_DIR := /usr/local/cuda
    # On Ubuntu 14.04, if cuda tools are installed via
    # "sudo apt-get install nvidia-cuda-toolkit" then use this instead:
    # CUDA_DIR := /usr
    
    # CUDA architecture setting: going with all of them.
    # For CUDA < 6.0, comment the *_50 lines for compatibility.
    CUDA_ARCH := -gencode arch=compute_20,code=sm_20 \
            -gencode arch=compute_20,code=sm_21 \
            -gencode arch=compute_30,code=sm_30 \
            -gencode arch=compute_35,code=sm_35 \
            -gencode arch=compute_50,code=sm_50 \
            -gencode arch=compute_50,code=compute_50

    # BLAS choice:
    # atlas for ATLAS (default)
    # mkl for MKL
    # open for OpenBlas
    BLAS := open
    # Custom (MKL/ATLAS/OpenBLAS) include and lib directories.
    # Leave commented to accept the defaults for your choice of BLAS
    # (which should work)!
    # BLAS_INCLUDE := /path/to/your/blas
    # BLAS_LIB := /path/to/your/blas
    
    # Homebrew puts openblas in a directory that is not on the standard search path
    # BLAS_INCLUDE := $(shell brew --prefix openblas)/include
    # BLAS_LIB := $(shell brew --prefix openblas)/lib
    
    # This is required only if you will compile the matlab interface.
    # MATLAB directory should contain the mex binary in /bin.
    # MATLAB_DIR := /usr/local
    # MATLAB_DIR := /Applications/MATLAB_R2012b.app
    
    # NOTE: this is required only if you will compile the python interface.
    # We need to be able to find Python.h and numpy/arrayobject.h.
    #PYTHON_INCLUDE := /usr/include/python2.7 \
            /usr/lib/python2.7/dist-packages/numpy/core/include
    # Anaconda Python distribution is quite popular. Include path:
    # Verify anaconda location, sometimes it's in root.
    ANACONDA_HOME := $(HOME)/anaconda2
    PYTHON_INCLUDE := $(ANACONDA_HOME)/include \
             $(ANACONDA_HOME)/include/python2.7 \
             $(ANACONDA_HOME)/lib/python2.7/site-packages/numpy/core/include 
    
    # Uncomment to use Python 3 (default is Python 2)
    # PYTHON_LIBRARIES := boost_python3 python3.5m
    # PYTHON_INCLUDE := /usr/include/python3.5m \
    #                 /usr/lib/python3.5/dist-packages/numpy/core/include
    
    # We need to be able to find libpythonX.X.so or .dylib.
    #PYTHON_LIB := /usr/lib
    PYTHON_LIB := $(ANACONDA_HOME)/lib
    
    # Homebrew installs numpy in a non standard path (keg only)
    # PYTHON_INCLUDE += $(dir $(shell python -c 'import numpy.core; print(numpy.core.__file__)'))/include
    # PYTHON_LIB += $(shell brew --prefix numpy)/lib
    
    # Uncomment to support layers written in Python (will link against Python libs)
    # WITH_PYTHON_LAYER := 1
    
    # Whatever else you find you need goes here.
    INCLUDE_DIRS := $(PYTHON_INCLUDE) /usr/local/include
    LIBRARY_DIRS := $(PYTHON_LIB) /usr/local/lib /usr/lib
    
        # If Homebrew is installed at a non standard location (for example your home directory) and you use it for general dependencies
    # INCLUDE_DIRS += $(shell brew --prefix)/include
    # LIBRARY_DIRS += $(shell brew --prefix)/lib
    
    # Uncomment to use `pkg-config` to specify OpenCV library paths.
    # (Usually not necessary -- OpenCV libraries are normally installed in one of the above $LIBRARY_DIRS.)
    USE_PKG_CONFIG := 1
    
    BUILD_DIR := build
    DISTRIBUTE_DIR := distribute
    
    # Uncomment for debugging. Does not work on OSX due to https://github.com/BVLC/caffe/issues/171
    # DEBUG := 1
    
    # The ID of the GPU that 'make runtest' will use to run unit tests.
    TEST_GPUID := 0

    # enable pretty build (comment to see full commands)
    Q ?= @

### For Caffe without Anaconda

    ## Refer to http://caffe.berkeleyvision.org/installation.html
    # Contributions simplifying and improving our build system are welcome!
    
    # cuDNN acceleration switch (uncomment to build with cuDNN).
    # USE_CUDNN := 1
    
    # CPU-only switch (uncomment to build without GPU support).
    CPU_ONLY := 1
    
    # uncomment to disable IO dependencies and corresponding data layers
    # USE_OPENCV := 0
    # USE_LEVELDB := 0
    # USE_LMDB := 0
    
    # uncomment to allow MDB_NOLOCK when reading LMDB files (only if necessary)
    #	You should not set this flag if you will be reading LMDBs with any
    #	possibility of simultaneous read and write
    # ALLOW_LMDB_NOLOCK := 1
    
    # Uncomment if you're using OpenCV 3
    # OPENCV_VERSION := 3
    
    # To customize your choice of compiler, uncomment and set the following.
    # N.B. the default for Linux is g++ and the default for OSX is clang++
    # CUSTOM_CXX := g++
    
    # CUDA directory contains bin/ and lib/ directories that we need.
    CUDA_DIR := /usr/local/cuda
    # On Ubuntu 14.04, if cuda tools are installed via
    # "sudo apt-get install nvidia-cuda-toolkit" then use this instead:
    # CUDA_DIR := /usr
    
    # CUDA architecture setting: going with all of them.
    # For CUDA < 6.0, comment the *_50 lines for compatibility.
    CUDA_ARCH := -gencode arch=compute_20,code=sm_20 \
            -gencode arch=compute_20,code=sm_21 \
            -gencode arch=compute_30,code=sm_30 \
            -gencode arch=compute_35,code=sm_35 \
            -gencode arch=compute_50,code=sm_50 \
            -gencode arch=compute_50,code=compute_50
    
    # BLAS choice:
    # atlas for ATLAS (default)
    # mkl for MKL
    # open for OpenBlas
    BLAS := open
    # Custom (MKL/ATLAS/OpenBLAS) include and lib directories.
    # Leave commented to accept the defaults for your choice of BLAS
    # (which should work)!
    # BLAS_INCLUDE := /path/to/your/blas
    # BLAS_LIB := /path/to/your/blas
    
    # Homebrew puts openblas in a directory that is not on the standard search path
    # BLAS_INCLUDE := $(shell brew --prefix openblas)/include
    # BLAS_LIB := $(shell brew --prefix openblas)/lib
    
    # This is required only if you will compile the matlab interface.
    # MATLAB directory should contain the mex binary in /bin.
    # MATLAB_DIR := /usr/local
    # MATLAB_DIR := /Applications/MATLAB_R2012b.app
    
    # NOTE: this is required only if you will compile the python interface.
    # We need to be able to find Python.h and numpy/arrayobject.h.
    PYTHON_INCLUDE := /usr/include/python2.7 \
            /usr/lib/python2.7/dist-packages/numpy/core/include
    # Anaconda Python distribution is quite popular. Include path:
    # Verify anaconda location, sometimes it's in root.
    # ANACONDA_HOME := $(HOME)/anaconda
    # PYTHON_INCLUDE := $(ANACONDA_HOME)/include \
            # $(ANACONDA_HOME)/include/python2.7 \
            # $(ANACONDA_HOME)/lib/python2.7/site-packages/numpy/core/include \
    
    # Uncomment to use Python 3 (default is Python 2)
    # PYTHON_LIBRARIES := boost_python3 python3.5m
    # PYTHON_INCLUDE := /usr/include/python3.5m \
    #                 /usr/lib/python3.5/dist-packages/numpy/core/include
    
    # We need to be able to find libpythonX.X.so or .dylib.
    PYTHON_LIB := /usr/lib
    # PYTHON_LIB := $(ANACONDA_HOME)/lib
    
    # Homebrew installs numpy in a non standard path (keg only)
    # PYTHON_INCLUDE += $(dir $(shell python -c 'import numpy.core; print(numpy.core.__file__)'))/include
    # PYTHON_LIB += $(shell brew --prefix numpy)/lib
    
    # Uncomment to support layers written in Python (will link against Python libs)
    # WITH_PYTHON_LAYER := 1
    
    # Whatever else you find you need goes here.
    INCLUDE_DIRS := $(PYTHON_INCLUDE) /usr/local/include
    LIBRARY_DIRS := $(PYTHON_LIB) /usr/local/lib /usr/lib
    
        # If Homebrew is installed at a non standard location (for example your home directory) and you use it for general dependencies
    # INCLUDE_DIRS += $(shell brew --prefix)/include
    # LIBRARY_DIRS += $(shell brew --prefix)/lib
    
    # Uncomment to use `pkg-config` to specify OpenCV library paths.
    # (Usually not necessary -- OpenCV libraries are normally installed in one of the above $LIBRARY_DIRS.)
    # USE_PKG_CONFIG := 1
    
    BUILD_DIR := build
    DISTRIBUTE_DIR := distribute
    
    # Uncomment for debugging. Does not work on OSX due to https://github.com/BVLC/caffe/issues/171
    # DEBUG := 1
    
    # The ID of the GPU that 'make runtest' will use to run unit tests.
    TEST_GPUID := 0
    
    # enable pretty build (comment to see full commands)
    Q ?= @
