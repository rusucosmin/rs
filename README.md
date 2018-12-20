# CrowdAI Road Segmentation

**Artem Kislovskiy, Alexandru Mocanu, Cosmin Rusu**

This project presents *ViaNet* - a convolutional neural network for segmenting satellite images (labeling each pixel). It was developed as the final Project for the course "Machine Learning" at EPFL in 2018.
The set of satellite images are acquired from Google Maps. Provided were also ground-truth images where each pixel is labeled as road or background.
The dataset is available from the [CrowdAI page](https://www.crowdai.org/challenges/epfl-ml-road-segmentation).
Our best submission [22878](https://www.crowdai.org/challenges/47/submissions/2287), submitted from username [CosminRusu](https://www.crowdai.org/participants/cosmin-rusu), got an F1-score of `0.832`.


# Initial setup
In the directory containing the *run.py* script, create a folder called *data/*. In this folder, you will need to unzip your training and test datasets provided on CrowdAI.

# How to train
For training, you just need to run:
```
python run.py
```
This will automatically load the training set, build the model and train it. At the end of training, the model is also run on the test set.

# How to test
For testing, you first need to have your pretrained model, called *model.h5*, in the same directory with *run.py*. Afterwards, just run:
```
python run.py --train False
```
This will load the pretrained model, load the test dataset and run the predictions. In the end, you will have a predictions file, named *cnn_vCURRENT_TIME.csv*, in the format expected on CrowdAI. The predicted segmentations for the test images will be available in directory *predictions_test/*.

# Pretrained model location
If you want to run our pretrained model, you can download it from [here](https://drive.google.com/file/d/1TV85S74RPpRP8OwOVjyvU5XhP-LNi0RP/view?usp=sharing).

# Libraries

The following libraries must be installed to run the project:
* Keras 2.2.4
* TensorFlow 1.5.0

Keras is a deep learning library that can use either Theano, NLTK or TensorFlow as backend. We used Tensorflow, and therefore, this project must be run with TensorFlow backend to ensure reproducibility.

# Setup

The model has been trained using GPU acceleration on a cluster with the following hardware:

* `9` Nodes;
* `342` CPUs;
* `1.6` TB of Memory;
* `143.2` TB Disk.
