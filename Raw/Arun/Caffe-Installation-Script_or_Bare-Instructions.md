##Caffe Installation Script/Bare Instructions
###Go one step at a time, understand the script and then make it your own.
####Tested on Ubuntu 14.04
####Full explanations available here in my [Caffe Installation Tutorial](../Raw/Arun/Caffe Installation Instructions.md).

#####Add required repositories, update and upgrade database.
```
echo 'deb http://archive.ubuntu.com/ubuntu trusty main restricted universe multiverse' >>/tmp/multiverse.list
sudo cp /tmp/multiverse.list /etc/apt/sources.list.d/
rm /tmp/multiverse.list
sudo apt-get -y install software-properties-common
sudo add-apt-repository ppa:mc3man/trusty-media
sudo apt-get update && sudo apt-get upgrade && sudo apt-get dist-upgrade && sudo apt-get autoremove
```
##### Remove older versions and install new versions of relevant packages.
```
sudo apt-get -y remove ffmpeg x264 libx264-dev
```
```
sudo apt-get -y install libopenblas-dev libboost-all-dev libfaac-dev ffmpeg gstreamer0.10-ffmpeg build-essential cmake git libgtk2.0-dev pkg-config libavcodec-dev libavformat-dev libswscale-dev unzip python-dev python-numpy libtbb2 libtbb-dev libjpeg-dev libpng-dev libtiff-dev libjasper-dev libdc1394-22-dev libtiff4-dev libopenexr-dev libeigen2-dev yasm libopencore-amrnb-dev libtheora-dev libvorbis-dev libxvidcore-dev python-tk libeigen3-dev libx264-dev libqt4-dev libqt4-opengl-dev sphinx-common texlive-latex-extra libv4l-dev default-jdk ant libvtk5-qt4-dev python-pip python-numpy python-scipy python-matplotlib ipython ipython-notebook python-pandas python-sympy python-nose libprotobuf-dev libleveldb-dev libsnappy-dev libopencv-dev libhdf5-serial-dev libgflags-dev libgoogle-glog-dev liblmdb-dev protobuf-compiler python-pydot zip```
```
```
sudo apt-get -y install --fix-missing libboost-all-dev
```
#####Install OpenCV
```
cd ~
wget http://sourceforge.net/projects/opencvlibrary/files/opencv-unix/2.4.9/opencv-2.4.9.zip
unzip opencv-2.4.9.zip
cd opencv-2.4.9
mkdir build
cd build
cmake -D WITH_TBB=ON -D BUILD_NEW_PYTHON_SUPPORT=ON -D WITH_V4L=ON -D INSTALL_C_EXAMPLES=ON -D INSTALL_PYTHON_EXAMPLES=ON -D BUILD_EXAMPLES=ON -D WITH_QT=ON -D WITH_OPENGL=ON -D WITH_VTK=ON ..
sudo make -j10
sudo make install
sudo sh -c 'echo "/usr/local/lib" > /etc/ld.so.conf.d/opencv.conf'
sudo ldconfig
```
**Export the following to .bashrc file**
```
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/usr/local/opencv/lib
```
#####Install Cython
**Updated the download link, old one does not work anymore**
```
cd ~
wget https://pypi.python.org/packages/0a/b6/fd142319fd0fe83dc266cfe83bfd095bc200fae5190fce0a2482560acb55/Cython-0.23.4.zip#md5=84c8c764ffbeae5f4a513d25fda4fd5e
unzip Cython-0.23.4.zip
cd Cython-0.23.4
sudo python setup.py install
sudo pip install protobuf scikit-image scikit-learn
```
#####Make/Install Caffe
```
cd ~
git clone https://github.com/BVLC/caffe
cd caffe
cp Makefile.config.example Makefile.config
```
**Make necessary changes in Makefile.config**
```
cd ~/caffe/python
sudo pip install -r requirements.txt
cd ~/caffe
sudo make all -j10
sudo make pycaffe
sudo make distribute
sudo make test
sudo make runtest
```
**Update .bashrc with the following**
```
#Caffe Root

export CAFFE_ROOT=/home/<username>/caffe/
export PYTHONPATH=/home/<username>/caffe/distribute/python:$PYTHONPATH
export PYTHONPATH=/home/<username>/caffe/python:$PYTHONPATH
```
**REBOOT the system**
```
sudo reboot
```
