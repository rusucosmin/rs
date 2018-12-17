"""
Baseline for machine learning project on road segmentation.
This simple baseline consits of a CNN with two convolutional+pooling layers with a soft-max loss

Credits: Aurelien Lucchi, ETH Zürich
"""

import gzip
import os
import sys
import urllib
import numpy as np
import pandas as pd
import matplotlib.image as mpimg
from PIL import Image
from argparse import ArgumentParser

import code
import tensorflow.python.platform
import tensorflow as tf

NUM_CHANNELS = 3  # RGB images
PIXEL_DEPTH = 255
NUM_LABELS = 2
TRAINING_SIZE = 100
VALIDATION_SIZE = 5  # Size of the validation set.
SEED = 66478  # Set to None for random seed.
BATCH_SIZE = 16  # 64
NUM_EPOCHS = 100
RECORDING_STEP = 0

# Set image patch size in pixels
# IMG_PATCH_SIZE should be a multiple of 4
# image size should be an integer multiple of this number!
IMG_PATCH_SIZE = 16
SUB_PATCH_SIZE = 16

# Extract patches from a given image
def img_crop(im, w, h, step=None):
    list_patches = []
    imgwidth = im.shape[0]
    imgheight = im.shape[1]
    is_2d = len(im.shape) < 3

    # No step provided means that we'll take disjoint patches
    if step is None:
        step = h
    for i in range(0, imgheight - h + 1, step):
        for j in range(0, imgwidth - w + 1, step):
            if is_2d:
                im_patch = im[j:j+w, i:i+h]
            else:
                im_patch = im[j:j+w, i:i+h, :]
            list_patches.append(im_patch)
            # Add multiple of 90 degrees rotations as well
            list_patches.append(np.rot90(im_patch, 1))
            list_patches.append(np.rot90(im_patch, 2))
            list_patches.append(np.rot90(im_patch, 3))
    return list_patches


def extract_data(filename, num_images, crop_step=None):
    """Extract the images into a 4D tensor [image index, y, x, channels].
    Values are rescaled from [0, 255] down to [-0.5, 0.5].
    """
    imgs = []
    for i in range(1, num_images+1):
        imageid = "satImage_%.3d" % i
        image_filename = filename + imageid + ".png"
        if os.path.isfile(image_filename):
            print('Loading ' + image_filename)
            img = mpimg.imread(image_filename)
            imgs.append(img)
        else:
            print('File ' + image_filename + ' does not exist')

    num_images = len(imgs)
    IMG_WIDTH = imgs[0].shape[0]
    IMG_HEIGHT = imgs[0].shape[1]

    img_patches = [img_crop(imgs[i], IMG_PATCH_SIZE, IMG_PATCH_SIZE, crop_step) for i in range(num_images)]
    data = [img_patches[i][j] for i in range(len(img_patches)) for j in range(len(img_patches[i]))]

    return np.asarray(data)


# Assign a label to a patch v
def value_to_class(v):
    foreground_threshold = 0.25  # percentage of pixels > 1 required to assign a foreground label to a patch
    df = np.sum(v)
    if df > foreground_threshold:  # road
        return [0, 1]
    else:  # bgrd
        return [1, 0]


# Extract label images
def extract_labels(filename, num_images, crop_step=None):
    """Extract the labels into a 1-hot matrix [image index, label index]."""
    gt_imgs = []
    for i in range(1, num_images + 1):
        imageid = "satImage_%.3d" % i
        image_filename = filename + imageid + ".png"
        if os.path.isfile(image_filename):
            print('Loading ' + image_filename)
            img = mpimg.imread(image_filename)
            gt_imgs.append(img)
        else:
            print('File ' + image_filename + ' does not exist')

    num_images = len(gt_imgs)
    gt_patches = [img_crop(gt_imgs[i], IMG_PATCH_SIZE, IMG_PATCH_SIZE, crop_step) for i in range(num_images)]
    data = np.asarray([gt_patches[i][j] for i in range(len(gt_patches)) for j in range(len(gt_patches[i]))])
    labels = np.asarray([value_to_class(np.mean(data[i])) for i in range(len(data))])

    # Convert to dense 1-hot representation.
    return labels.astype(np.float32)


def error_rate(predictions, labels):
    """Return the error rate based on dense predictions and 1-hot labels."""
    return 100.0 - (
        100.0 *
        np.sum(np.argmax(predictions, 1) == np.argmax(labels, 1)) /
        predictions.shape[0])


# Write predictions from neural network to a file
def write_predictions_to_file(predictions, labels, filename):
    max_labels = np.argmax(labels, 1)
    max_predictions = np.argmax(predictions, 1)
    file = open(filename, "w")
    n = predictions.shape[0]
    for i in range(0, n):
        file.write(max_labels(i) + ' ' + max_predictions(i))
    file.close()


# Print predictions from neural network
def print_predictions(predictions, labels):
    max_labels = np.argmax(labels, 1)
    max_predictions = np.argmax(predictions, 1)
    print(str(max_labels) + ' ' + str(max_predictions))


# Convert array of labels to an image
def label_to_img(imgwidth, imgheight, w, h, labels):
    array_labels = np.zeros([imgwidth, imgheight])
    idx = 0
    for i in range(0, imgheight, h):
        for j in range(0, imgwidth, w):
            if labels[idx][0] > 0.5:  # bgrd
                l = 0
            else:
                l = 1
            array_labels[j:j+w, i:i+h] = l
            idx = idx + 1
    return array_labels


def img_float_to_uint8(img):
    rimg = img - np.min(img)
    rimg = (rimg / np.max(rimg) * PIXEL_DEPTH).round().astype(np.uint8)
    return rimg


def concatenate_images(img, gt_img):
    n_channels = len(gt_img.shape)
    w = gt_img.shape[0]
    h = gt_img.shape[1]
    if n_channels == 3:
        cimg = np.concatenate((img, gt_img), axis=1)
    else:
        gt_img_3c = np.zeros((w, h, 3), dtype=np.uint8)
        gt_img8 = img_float_to_uint8(gt_img)
        gt_img_3c[:, :, 0] = gt_img8
        gt_img_3c[:, :, 1] = gt_img8
        gt_img_3c[:, :, 2] = gt_img8
        img8 = img_float_to_uint8(img)
        cimg = np.concatenate((img8, gt_img_3c), axis=1)
    return cimg


def make_img_overlay(img, predicted_img):
    w = img.shape[0]
    h = img.shape[1]
    color_mask = np.zeros((w, h, 3), dtype=np.uint8)
    color_mask[:, :, 0] = predicted_img*PIXEL_DEPTH

    img8 = img_float_to_uint8(img)
    background = Image.fromarray(img8, 'RGB').convert("RGBA")
    overlay = Image.fromarray(color_mask, 'RGB').convert("RGBA")
    new_img = Image.blend(background, overlay, 0.2)
    return new_img


def main(argv=None):  # pylint: disable=unused-argument
    train, restore_model, model_dir, crop_step = argv

    if train:
        # Training
        data_dir = 'data/training/'
        train_data_filename = data_dir + 'images/'
        train_labels_filename = data_dir + 'groundtruth/'

        # Extract it into np arrays.
        train_data = extract_data(train_data_filename, TRAINING_SIZE, crop_step)
        train_labels = extract_labels(train_labels_filename, TRAINING_SIZE, crop_step)

        print("Data shape", train_data.shape, train_labels.shape)
        num_epochs = NUM_EPOCHS

        c0 = 0  # bgrd
        c1 = 0  # road
        for i in range(len(train_labels)):
            if train_labels[i][0] == 1:
                c0 = c0 + 1
            else:
                c1 = c1 + 1
        print('Number of data points per class: c0 = ' + str(c0) + ' c1 = ' + str(c1))

        print('Balancing training data...')
        min_c = min(c0, c1)
        idx0 = [i for i, j in enumerate(train_labels) if j[0] == 1]
        idx1 = [i for i, j in enumerate(train_labels) if j[1] == 1]
        new_indices = idx0[0:min_c] + idx1[0:min_c]
        print(len(new_indices))
        print(train_data.shape)
        train_data = train_data[new_indices, :, :, :]
        train_labels = train_labels[new_indices]

        train_size = train_labels.shape[0]

        c0 = 0
        c1 = 0
        for i in range(len(train_labels)):
            if train_labels[i][0] == 1:
                c0 = c0 + 1
            else:
                c1 = c1 + 1
        print('Number of data points per class: c0 = ' + str(c0) + ' c1 = ' + str(c1))

        # This is where training samples and labels are fed to the graph.
        # These placeholder nodes will be fed a batch of training data at each
        # training step using the {feed_dict} argument to the Run() call below.
        train_data_node = tf.placeholder(
            tf.float32,
            shape=(BATCH_SIZE, IMG_PATCH_SIZE, IMG_PATCH_SIZE, NUM_CHANNELS))
        train_labels_node = tf.placeholder(tf.float32,
                                           shape=(BATCH_SIZE, NUM_LABELS))
        train_all_data_node = tf.constant(train_data)
    else:
        # Testing
        data_dir = 'data/test_set_images'
        # Get data from all images
        images = os.listdir(data_dir)
        test_imgs = {}
        for image in images:
            image_filename = os.path.join(data_dir, image, image + '.png')
            print ('Loading ' + image_filename)
            img = mpimg.imread(image_filename)
            img_id = '0' + image.split('_')[1]
            if len(img_id) == 2:
                img_id = f'0{img_id}'
            test_imgs[img_id] = img

    # The variables below hold all the trainable weights. They are passed an
    # initial value which will be assigned when when we call:
    # {tf.initialize_all_variables().run()}
    conv1_weights = tf.Variable(
        tf.truncated_normal([5, 5, NUM_CHANNELS, 32],  # 5x5 filter, depth 32.
                            stddev=0.1,
                            seed=SEED))
    conv1_biases = tf.Variable(tf.zeros([32]))
    conv2_weights = tf.Variable(
        tf.truncated_normal([5, 5, 32, 64],
                            stddev=0.1,
                            seed=SEED))
    conv2_biases = tf.Variable(tf.constant(0.1, shape=[64]))
    fc1_weights = tf.Variable(  # fully connected, depth 512.
        tf.truncated_normal([int(IMG_PATCH_SIZE / 4 * IMG_PATCH_SIZE / 4 * 64), 512],
                            stddev=0.1,
                            seed=SEED))
    fc1_biases = tf.Variable(tf.constant(0.1, shape=[512]))
    fc2_weights = tf.Variable(
        tf.truncated_normal([512, NUM_LABELS],
                            stddev=0.1,
                            seed=SEED))
    fc2_biases = tf.Variable(tf.constant(0.1, shape=[NUM_LABELS]))

    # Make an image summary for 4d tensor image with index idx
    def get_image_summary(img, idx=0):
        V = tf.slice(img, (0, 0, 0, idx), (1, -1, -1, 1))
        img_w = img.get_shape().as_list()[1]
        img_h = img.get_shape().as_list()[2]
        min_value = tf.reduce_min(V)
        V = V - min_value
        max_value = tf.reduce_max(V)
        V = V / (max_value*PIXEL_DEPTH)
        V = tf.reshape(V, (img_w, img_h, 1))
        V = tf.transpose(V, (2, 0, 1))
        V = tf.reshape(V, (-1, img_w, img_h, 1))
        return V

    # Make an image summary for 3d tensor image with index idx
    def get_image_summary_3d(img):
        V = tf.slice(img, (0, 0, 0), (1, -1, -1))
        img_w = img.get_shape().as_list()[1]
        img_h = img.get_shape().as_list()[2]
        V = tf.reshape(V, (img_w, img_h, 1))
        V = tf.transpose(V, (2, 0, 1))
        V = tf.reshape(V, (-1, img_w, img_h, 1))
        return V

    # Get prediction for given input image
    def get_prediction(img):
        data = np.asarray(img_crop(img, IMG_PATCH_SIZE, IMG_PATCH_SIZE))
        data_node = tf.constant(data)
        output = tf.nn.softmax(model(data_node))
        output_prediction = s.run(output)
        img_prediction = label_to_img(img.shape[0], img.shape[1], IMG_PATCH_SIZE, IMG_PATCH_SIZE, output_prediction)

        return img_prediction

    # Get a concatenation of the prediction and groundtruth for given input file
    def get_prediction_with_groundtruth(filename, image_idx):

        imageid = "satImage_%.3d" % image_idx
        image_filename = filename + imageid + ".png"
        img = mpimg.imread(image_filename)

        img_prediction = get_prediction(img)
        cimg = concatenate_images(img, img_prediction)

        return cimg

    # Get prediction overlaid on the original image for given input file
    def get_prediction_with_overlay(filename, image_idx):

        imageid = "satImage_%.3d" % image_idx
        image_filename = filename + imageid + ".png"
        img = mpimg.imread(image_filename)

        img_prediction = get_prediction(img)
        oimg = make_img_overlay(img, img_prediction)

        return oimg

    # We will replicate the model structure for the training subgraph, as well
    # as the evaluation subgraphs, while sharing the trainable parameters.
    def model(data, train=False):
        """The Model definition."""
        # 2D convolution, with 'SAME' padding (i.e. the output feature map has
        # the same size as the input). Note that {strides} is a 4D array whose
        # shape matches the data layout: [image index, y, x, depth].
        conv = tf.nn.conv2d(data,
                            conv1_weights,
                            strides=[1, 1, 1, 1],
                            padding='SAME')
        # Bias and rectified linear non-linearity.
        relu = tf.nn.relu(tf.nn.bias_add(conv, conv1_biases))
        # Max pooling. The kernel size spec {ksize} also follows the layout of
        # the data. Here we have a pooling window of 2, and a stride of 2.
        pool = tf.nn.max_pool(relu,
                              ksize=[1, 2, 2, 1],
                              strides=[1, 2, 2, 1],
                              padding='SAME')

        conv2 = tf.nn.conv2d(pool,
                             conv2_weights,
                             strides=[1, 1, 1, 1],
                             padding='SAME')
        relu2 = tf.nn.relu(tf.nn.bias_add(conv2, conv2_biases))
        pool2 = tf.nn.max_pool(relu2,
                               ksize=[1, 2, 2, 1],
                               strides=[1, 2, 2, 1],
                               padding='SAME')

        # Uncomment these lines to check the size of each layer
        # print 'data ' + str(data.get_shape())
        # print 'conv ' + str(conv.get_shape())
        # print 'relu ' + str(relu.get_shape())
        # print 'pool ' + str(pool.get_shape())
        # print 'pool2 ' + str(pool2.get_shape())

        # Reshape the feature map cuboid into a 2D matrix to feed it to the
        # fully connected layers.
        pool_shape = pool2.get_shape().as_list()
        reshape = tf.reshape(
            pool2,
            [pool_shape[0], pool_shape[1] * pool_shape[2] * pool_shape[3]])
        # Fully connected layer. Note that the '+' operation automatically
        # broadcasts the biases.
        hidden = tf.nn.relu(tf.matmul(reshape, fc1_weights) + fc1_biases)
        # Add a 50% dropout during training only. Dropout also scales
        # activations such that no rescaling is needed at evaluation time.
        # if train:
        #    hidden = tf.nn.dropout(hidden, 0.5, seed=SEED)
        out = tf.matmul(hidden, fc2_weights) + fc2_biases

        if train:
            summary_id = '_0'
            s_data = get_image_summary(data)
            tf.summary.image('summary_data' + summary_id, s_data)
            s_conv = get_image_summary(conv)
            tf.summary.image('summary_conv' + summary_id, s_conv)
            s_pool = get_image_summary(pool)
            tf.summary.image('summary_pool' + summary_id, s_pool)
            s_conv2 = get_image_summary(conv2)
            tf.summary.image('summary_conv2' + summary_id, s_conv2)
            s_pool2 = get_image_summary(pool2)
            tf.summary.image('summary_pool2' + summary_id, s_pool2)
        return out

    if train:
        # Training computation: logits + cross-entropy loss.
        logits = model(train_data_node, True)  # BATCH_SIZE*NUM_LABELS
        # print 'logits = ' + str(logits.get_shape()) + ' train_labels_node = ' + str(train_labels_node.get_shape())

        loss = tf.reduce_mean(
            tf.nn.softmax_cross_entropy_with_logits_v2(labels=train_labels_node,
                                                       logits=logits))

        tf.summary.scalar('loss', loss)

        all_params_node = [conv1_weights, conv1_biases, conv2_weights, conv2_biases, fc1_weights, fc1_biases, fc2_weights, fc2_biases]
        all_params_names = ['conv1_weights', 'conv1_biases', 'conv2_weights', 'conv2_biases', 'fc1_weights', 'fc1_biases', 'fc2_weights', 'fc2_biases']
        all_grads_node = tf.gradients(loss, all_params_node)
        all_grad_norms_node = []
        for i in range(0, len(all_grads_node)):
            norm_grad_i = tf.global_norm([all_grads_node[i]])
            all_grad_norms_node.append(norm_grad_i)
            tf.summary.scalar(all_params_names[i], norm_grad_i)

        # L2 regularization for the fully connected parameters.
        regularizers = (tf.nn.l2_loss(fc1_weights) + tf.nn.l2_loss(fc1_biases) +
                        tf.nn.l2_loss(fc2_weights) + tf.nn.l2_loss(fc2_biases))
        # Add the regularization term to the loss.
        loss += 5e-4 * regularizers

        # Optimizer: set up a variable that's incremented once per batch and
        # controls the learning rate decay.
        batch = tf.Variable(0)
        # Decay once per epoch, using an exponential schedule starting at 0.01.
        learning_rate = tf.train.exponential_decay(
            0.01,                # Base learning rate.
            batch * BATCH_SIZE,  # Current index into the dataset.
            train_size,          # Decay step.
            0.95,                # Decay rate.
            staircase=True)
        # tf.scalar_summary('learning_rate', learning_rate)
        tf.summary.scalar('learning_rate', learning_rate)

        # Use simple momentum for the optimization.
        optimizer = tf.train.MomentumOptimizer(learning_rate,
                                               0.0).minimize(loss,
                                                             global_step=batch)

        # Predictions for the minibatch, validation set and test set.
        train_prediction = tf.nn.softmax(logits)
        # We'll compute them only once in a while by calling their {eval()} method.
        train_all_prediction = tf.nn.softmax(model(train_all_data_node))

    # Add ops to save and restore all the variables.
    saver = tf.train.Saver()

    # Create a local session to run this computation.
    with tf.Session() as s:
        if restore_model:
            # Restore variables from disk.
            saver.restore(s, model_dir + "/model.ckpt")
            print("Model restored.")

        else:
            # Run all the initializers to prepare the trainable parameters.
            tf.global_variables_initializer().run()

            # Build the summary operation based on the TF collection of Summaries.
            summary_op = tf.summary.merge_all()
            summary_writer = tf.summary.FileWriter(model_dir,
                                                   graph=s.graph)

            print('Initialized!')
            # Loop through training steps.
            print('Total number of iterations = ' + str(int(num_epochs * train_size / BATCH_SIZE)))

            training_indices = range(train_size)

            for iepoch in range(num_epochs):

                # Permute training indices
                perm_indices = np.random.permutation(training_indices)

                steps_per_epoch = int(train_size / BATCH_SIZE)

                for step in range(steps_per_epoch):

                    offset = (step * BATCH_SIZE) % (train_size - BATCH_SIZE)
                    batch_indices = perm_indices[offset:(offset + BATCH_SIZE)]

                    # Compute the offset of the current minibatch in the data.
                    # Note that we could use better randomization across epochs.
                    batch_data = train_data[batch_indices, :, :, :]
                    batch_labels = train_labels[batch_indices]
                    # This dictionary maps the batch data (as a np array) to the
                    # node in the graph is should be fed to.
                    feed_dict = {train_data_node: batch_data,
                                 train_labels_node: batch_labels}

                    if step == 0:
                        summary_str, _, l, lr, predictions = s.run(
                            [summary_op, optimizer, loss, learning_rate, train_prediction],
                            feed_dict=feed_dict)
                        summary_writer.add_summary(summary_str, iepoch * steps_per_epoch)
                        summary_writer.flush()

                        print('Epoch %d' % iepoch)
                        print('Minibatch loss: %.3f, learning rate: %.6f' % (l, lr))
                        print('Minibatch error: %.1f%%' % error_rate(predictions,
                                                                     batch_labels))

                        sys.stdout.flush()
                    else:
                        # Run the graph and fetch some of the nodes.
                        _, l, lr, predictions = s.run(
                            [optimizer, loss, learning_rate, train_prediction],
                            feed_dict=feed_dict)

                # Save the variables to disk.
                save_path = saver.save(s, model_dir + "/model.ckpt")
                print("Model saved in file: %s" % save_path)

        if train:
            # Training
            print("Running prediction on training set")
            prediction_training_dir = "predictions_training/"
            if not os.path.isdir(prediction_training_dir):
                os.mkdir(prediction_training_dir)
            for i in range(1, TRAINING_SIZE + 1):
                pimg = get_prediction_with_groundtruth(train_data_filename, i)
                Image.fromarray(pimg).save(prediction_training_dir + "prediction_" + str(i) + ".png")
                oimg = get_prediction_with_overlay(train_data_filename, i)
                oimg.save(prediction_training_dir + "overlay_" + str(i) + ".png")
        else:
            # Testing
            print("Running prediction on test set")
            prediction_test_dir = "predictions_test/"
            if not os.path.isdir(prediction_test_dir):
                os.mkdir(prediction_test_dir)

            submission = {'id': [], 'prediction': []}
            test_keys = sorted(test_imgs.keys())
            for key in test_keys:
                # Generate image files
                pimg = get_prediction(test_imgs[key])
                cimg = concatenate_images(test_imgs[key], pimg)
                Image.fromarray(cimg).save(os.path.join(prediction_test_dir, f"prediction_{key}.png"))
                oimg = make_img_overlay(test_imgs[key], pimg)
                oimg.save(os.path.join(prediction_test_dir, f"overlay_{key}.png"))
                # Generate submission
                n, m = pimg.shape
                for j in range(0, m, SUB_PATCH_SIZE):
                    for i in range(0, n, SUB_PATCH_SIZE):
                        patch = pimg[i:i+SUB_PATCH_SIZE, j:j+SUB_PATCH_SIZE]
                        foreground_thresh = 0.25
                        ones = np.sum(patch)
                        if ones > foreground_thresh * SUB_PATCH_SIZE ** 2:
                            submission['prediction'].append(1)
                        else:
                            submission['prediction'].append(0)
                        patch_id = f'{key}_{j}_{i}'
                        submission['id'].append(patch_id)
                # Write submission file
                submission_df = pd.DataFrame(submission)
                submission_df.to_csv(os.path.join(prediction_test_dir, 'submission.csv'), index=False)


if __name__ == '__main__':
    argparser = ArgumentParser()
    argparser.add_argument('--train', type=str, default='True',
        help='choose whether to train or test model')
    argparser.add_argument('--restore_model', type=str, default='False',
        help='choose whether to restore the model')
    argparser.add_argument('--model_dir', type=str, default='model',
        help='directory where to store the model')
    argparser.add_argument('--crop_step', type=int, default=16,
        help='distance in pixels between consecutive patches to use in the \
        training set')

    args = argparser.parse_args()
    args.train = (args.train == 'True')
    args.restore_model = (args.restore_model == 'True')
    # tf.app.run(argv=[args.train, args.restore_model])
    main([args.train, args.restore_model, args.model_dir])
