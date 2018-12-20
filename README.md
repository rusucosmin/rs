# Project Road Segmentation

**Artem Kislovskiy, Alexandru Mocanu, Cosmin Rusu**

This project presents *ViaNet* - a convolutional neural network for segmenting sattelite images (labeling each pixel). It was developed as the final Project for the course "Machine Learning" at EPFL in 2018.

The set of satellite images are acquired from Google Maps. Provided were also ground-truth images where each pixel is labeled as road or background.

The dataset is available from the [CrowdAI page](https://www.crowdai.org/challenges/epfl-ml-road-segmentation).

Our best submission [22878](https://www.crowdai.org/challenges/47/submissions/2287), got an F1-Score [F1 score](https://en.wikipedia.org/wiki/F1_score) of `0.832`.


# Libraries

The following libraries must be installed to run the project:

* Keras 2.2.5
* TensorFlow 1.5.0

Keras is a deep learning library that can use either Theano or TensorFlow as backend. We used Tensorflow, and therefore, this project must be run with TensorFlow backend to ensure reproducibility.

# Setup

The model has been trained using GPU acceleration on a cluster with the following hardware:

* `9` Nodes;
* `342` CPUs;
* `1.6` TB of Memory;
* `143.2` TB Disk.


