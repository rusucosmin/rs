import matplotlib.image as mpimg
import numpy as np
import matplotlib.pyplot as plt
import os,sys
from PIL import Image
from time import time
from helpers import *
from argparse import ArgumentParser
from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D, ReLU, Dropout, Flatten, Dense, ELU, Activation
from keras.regularizers import l2
from keras.optimizers import Adam
from keras.callbacks import ReduceLROnPlateau, EarlyStopping, ModelCheckpoint
from keras.preprocessing.image import ImageDataGenerator
from keras.utils import np_utils


# Extract patches from input images
WINDOW_SIZE = 72
PATCH_SIZE = 16 # each patch is 16*16 pixels
PADDING_SIZE = (WINDOW_SIZE - PATCH_SIZE) // 2


def predict(model, X):
    patches = img_crop_v2(pad_image(X, PADDING_SIZE), WINDOW_SIZE, WINDOW_SIZE, PATCH_SIZE, PATCH_SIZE)
    img_patches = np.zeros(shape=(len(patches), WINDOW_SIZE, WINDOW_SIZE, 3))
    for index_patch, patch in enumerate(patches):
        img_patches[index_patch] = patch
    Z = model.predict(img_patches)
    Z = (Z[:,0] < Z[:,1]) * 1
    Z = Z.reshape(img_patches.shape[0], -1)
    return Z


def mask_to_submission_strings(model, image_filename, prediction_dir):
    """ Reads a single image and outputs the strings that should go into the submission file. """
    img_number = int(re.search(r"\d+", image_filename).group(0))
    Xi = load_image(image_filename)
    Zi = predict(model, Xi)
    Zi = Zi.reshape(-1)
    nb = 0
    cnt = np.zeros(2)
    print("Processing " + image_filename)
    for j in range(0, Xi.shape[1], PATCH_SIZE):
        for i in range(0, Xi.shape[0], PATCH_SIZE):
            label = int(Zi[nb])
            cnt[label] += 1
            nb += 1
            yield("{:03d}_{}_{},{}".format(img_number, j, i, label))
    print("cnt[0]=" + str(cnt[0]))
    print("cnt[1]=" + str(cnt[1]))

    # Save prediction images
    pimg = label_to_img(Xi.shape[0], Xi.shape[1], PATCH_SIZE, PATCH_SIZE, Zi)
    cimg = concatenate_images(Xi, pimg)
    Image.fromarray(cimg).save(os.path.join(prediction_dir, f"prediction_{img_number}.png"))
    oimg = make_img_overlay(Xi, pimg)
    oimg.save(os.path.join(prediction_dir, f"overlay_{img_number}.png"))


def generate_submission(model, submission_filename, image_filenames):
    """ Generate a .csv containing the classification of the test set. """
    print("Running prediction on test set")
    prediction_test_dir = "predictions_test/"
    if not os.path.isdir(prediction_test_dir):
        os.mkdir(prediction_test_dir)

    with open(submission_filename, 'w') as f:
        f.write('id,prediction\n')
        for fn in image_filenames:
            f.writelines(['{}\n'.format(s) for s in mask_to_submission_strings(model, fn, prediction_test_dir)])


def build_model(useMaxPool=True, useReLU=True):
    INPUT_SHAPE=(WINDOW_SIZE, WINDOW_SIZE, 3)
    REG = 1e-6

    model = Sequential()
    model.add(Conv2D(filters=128,
                     kernel_size=(5, 5),
                     strides=(1, 1),
                     padding='same',
                     input_shape=INPUT_SHAPE))
    model.add(ReLU() if useReLU else ELU())
    model.add(MaxPooling2D(pool_size=(2, 2),
                           strides=None,
                           padding='same') if useMaxPool else \
                    Conv2D(filters = 128,
                         kernel_size = (5, 5),
                         strides = (2, 2),
                         padding='same',
                         input_shape=INPUT_SHAPE))
    model.add(Conv2D(filters = 256,
                     kernel_size = (5, 5),
                     strides = (1, 1),
                     padding='same',
                     input_shape=INPUT_SHAPE))
    model.add(ReLU() if useReLU else ELU())
    model.add(MaxPooling2D(pool_size=(2, 2),
                           strides=None,
                           padding='same') if useMaxPool else
                    Conv2D(filters = 256,
                         kernel_size = (4, 4),
                         strides = (2, 2),
                         padding='same',
                         input_shape=INPUT_SHAPE))
    model.add(Conv2D(filters = 512,
                     kernel_size = (4, 4),
                     strides = (1, 1),
                     padding='same',
                     input_shape=INPUT_SHAPE))
    model.add(ReLU() if useReLU else ELU())
    model.add(MaxPooling2D(pool_size=(2, 2),
                           strides=None,
                           padding='same') if useMaxPool else
                    Conv2D(filters = 512,
                         kernel_size = (4, 4),
                         strides = (2, 2),
                         padding='same',
                         input_shape=INPUT_SHAPE))
    model.add(Flatten())
    model.add(Dense(1024, kernel_regularizer=l2(REG)))
    model.add(ReLU() if useReLU else ELU())
    model.add(Dropout(0.25))
    model.add(Dense(512, kernel_regularizer=l2(REG)))
    model.add(ReLU() if useReLU else ELU())
    model.add(Dropout(0.5))
    model.add(Dense(128, kernel_regularizer=l2(REG)))
    model.add(ReLU() if useReLU else ELU())
    model.add(Dropout(0.5))
    model.add(Dense(2, kernel_regularizer=l2(REG)))
    model.add(Activation('softmax'))

    return model


def train(model, X, Y, augment_data=True):
    batch_size = 125
    samples_per_epoch = X.shape[0]*X.shape[1]*X.shape[2]//256
    nb_epoch = 200
    nb_classes = 2

    lr_callback = ReduceLROnPlateau(monitor='loss',
                                factor=0.2,
                                patience=5,
                                min_lr=0.001)
    stop_callback = EarlyStopping(monitor='acc',
                                  min_delta=0.0001,
                                  patience=11,
                                  verbose=1,
                                  mode='auto')
    checkpoint_callback = ModelCheckpoint('model.h5',
                                          monitor='loss',
                                          verbose=1,
                                          save_best_only=True,
                                          save_weights_only=False,
                                          mode='auto',
                                          period=1)

    def generate_minibatch():
        while 1:
            # Generate one minibatch
            X_batch = np.empty((batch_size, WINDOW_SIZE, WINDOW_SIZE, 3))
            Y_batch = np.empty((batch_size, 2))
            for i in range(batch_size):
                # Select a random image
                idx = np.random.choice(X.shape[0])
                shape = X[idx].shape

                # Sample a random window from the image
                center = np.random.randint(WINDOW_SIZE//2, shape[0] - WINDOW_SIZE//2, 2)
                sub_image = X[idx][center[0]-WINDOW_SIZE//2:center[0]+WINDOW_SIZE//2,
                                   center[1]-WINDOW_SIZE//2:center[1]+WINDOW_SIZE//2]
                gt_sub_image = Y[idx][center[0]-PATCH_SIZE//2:center[0]+PATCH_SIZE//2,
                                      center[1]-PATCH_SIZE//2:center[1]+PATCH_SIZE//2]

                # The label does not depend on the image rotation/flip (provided that the rotation is in steps of 90°)
                threshold = 0.25
                label = (np.array([np.mean(gt_sub_image)]) > threshold) * 1

                # Image augmentation
                # Random flip
                if np.random.choice(2) == 0:
                    # Flip vertically
                    sub_image = np.flipud(sub_image)
                if np.random.choice(2) == 0:
                    # Flip horizontally
                    sub_image = np.fliplr(sub_image)

                # Random rotation in steps of 90°
                num_rot = np.random.choice(4)
                sub_image = np.rot90(sub_image, num_rot)

                label = np_utils.to_categorical(label, nb_classes)
                X_batch[i] = sub_image
                Y_batch[i] = label

            yield (X_batch, Y_batch)

    if augment_data:
        model.fit_generator(generate_minibatch(),
            steps_per_epoch=samples_per_epoch//batch_size//10,
            epochs=nb_epoch,
            verbose=1,
            callbacks=[checkpoint_callback, stop_callback])
    else:
        model.fit(X, Y, callbacks=[stop_callback, checkpoint_callback])


if __name__ == '__main__':
    argparser = ArgumentParser()
    argparser.add_argument('--train', type=str, default='True',
        help='choose whether to train or test')
    args = argparser.parse_args()

    # Build model
    model = build_model()

    # Define optimizer
    opt = Adam(lr=0.001) # Adam optimizer with default initial learning rate
    model.compile(loss='categorical_crossentropy',
                  optimizer=opt,
                  metrics=['accuracy'])

    if args.train == 'True':
        # Loaded a set of images
        root_dir = "data/training/"

        image_dir = root_dir + "images/"
        files = os.listdir(image_dir)
        n = len(files)
        print("Loading " + str(n) + " images")
        imgs = [load_image(image_dir + files[i]) for i in range(n)]

        gt_dir = root_dir + "groundtruth/"
        print("Loading " + str(n) + " images")
        gt_imgs = [load_image(gt_dir + files[i]) for i in range(n)]

        # Train the model
        train(model, np.asarray(imgs), np.asarray(gt_imgs))
    else:
        model.load_weights('model.h5')

    generate_submission(model, f'cnn_v{int(time())}.csv', \
        [f'data/test_set_images/test_{str(i)}/test_{str(i)}.png' for i in range(1, 51)])
