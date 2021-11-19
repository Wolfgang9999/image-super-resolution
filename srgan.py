
from skimage.transform import resize as imresize
from PIL import Image
from matplotlib import pyplot as plt
from glob import glob
from imageio import imread
import warnings
from itertools import repeat
from numpy import asarray
import random
from keras.optimizers import Adam
from keras.models import Model
from keras.layers import Conv2D, UpSampling2D
from keras.layers import BatchNormalization, Activation, LeakyReLU, Add, Dense
from keras.callbacks import TensorBoard
from keras.applications import VGG19
from keras import Input
import glob
from tensorflow import keras
import tensorflow as tf
import os
import pandas as pd  # data processing, CSV file I/O (e.g. pd.read_csv)
import numpy as np  # linear algebra
import parameters
print(tf.__version__)


# warnings.filterwarnings("ignore")

print("Number of files in train directory => ",
      len(os.listdir(parameters.train_path)))

print("Number of files in val directory => ",
      len(os.listdir(parameters.val_path)))


# common_optimizer = Adam(0.0002, 0.5)
common_optimizer = Adam()
tf.random.set_seed(parameters.seed)


# %% [markdown]
# ## Model Architectures

# %% [markdown]
# 1. Generator Network
# 2. Discriminator Network
# 3. Feature extractor using VGG19 network
# 4. Adversarial framework

# %% [markdown]
# V1. Generator

# %% [markdown]
# 16 residual blocks & 2 upsampling blocks

# %% [code] {"execution":{"iopub.status.busy":"2021-11-16T17:42:32.068149Z","iopub.execute_input":"2021-11-16T17:42:32.068633Z","iopub.status.idle":"2021-11-16T17:42:32.078868Z","shell.execute_reply.started":"2021-11-16T17:42:32.068585Z","shell.execute_reply":"2021-11-16T17:42:32.077836Z"},"jupyter":{"outputs_hidden":false}}


def residual_block(x):

    filters = [64, 64]
    kernel_size = 3
    strides = 1
    padding = "same"
    momentum = 0.8
    activation = "relu"

    res = Conv2D(filters=filters[0], kernel_size=kernel_size,
                 strides=strides, padding=padding)(x)
    res = Activation(activation=activation)(res)
    res = BatchNormalization(momentum=momentum)(res)

    res = Conv2D(filters=filters[1], kernel_size=kernel_size,
                 strides=strides, padding=padding)(res)
    res = BatchNormalization(momentum=momentum)(res)

    res = Add()([res, x])

    return res

# %% [code] {"jupyter":{"outputs_hidden":false}}


def build_generator():

    residual_blocks = 16

    momentum = 0.8

    # 4x downsample of HR
    input_shape = (64, 64, 3)

    input_layer = Input(shape=input_shape)

    gen1 = Conv2D(filters=64, kernel_size=9, strides=1,
                  padding='same', activation='relu')(input_layer)

    # 16 residual blocks
    res = residual_block(gen1)
    for i in range(residual_blocks - 1):
        res = residual_block(res)

    gen2 = Conv2D(filters=64, kernel_size=3, strides=1, padding='same')(res)
    gen2 = BatchNormalization(momentum=momentum)(gen2)

    # take the sum of pre-residual block (gen1) and post-residual block (gen2)
    gen3 = Add()([gen2, gen1])

    # upsampling
    gen4 = UpSampling2D(size=2)(gen3)
    gen4 = Conv2D(filters=256, kernel_size=3, strides=1, padding='same')(gen4)
    gen4 = Activation('relu')(gen4)

    # upsampling
    gen5 = UpSampling2D(size=2)(gen4)
    gen5 = Conv2D(filters=256, kernel_size=3, strides=1, padding='same')(gen5)
    gen5 = Activation('relu')(gen5)

    # Output Image 3 channels RGB
    gen6 = Conv2D(filters=3, kernel_size=9, strides=1, padding='same')(gen5)
    output = Activation('tanh')(gen6)

    # model
    model = Model(inputs=[input_layer], outputs=[output], name='generator')

    return model


# %% [code] {"jupyter":{"outputs_hidden":false}}
generator = build_generator()

# %% [code] {"jupyter":{"outputs_hidden":false}}


def build_discriminator():

    leakyrelu_alpha = 0.2
    momentum = 0.8

    # the input is the HR shape
    input_shape = (256, 256, 3)

    # input layer for discriminator
    input_layer = Input(shape=input_shape)

    # 8 convolutional layers with batch normalization
    dis1 = Conv2D(filters=64, kernel_size=3, strides=1,
                  padding='same')(input_layer)
    dis1 = LeakyReLU(alpha=leakyrelu_alpha)(dis1)

    dis2 = Conv2D(filters=64, kernel_size=3, strides=2, padding='same')(dis1)
    dis2 = LeakyReLU(alpha=leakyrelu_alpha)(dis2)
    dis2 = BatchNormalization(momentum=momentum)(dis2)

    dis3 = Conv2D(filters=128, kernel_size=3, strides=1, padding='same')(dis2)
    dis3 = LeakyReLU(alpha=leakyrelu_alpha)(dis3)
    dis3 = BatchNormalization(momentum=momentum)(dis3)

    dis4 = Conv2D(filters=128, kernel_size=3, strides=2, padding='same')(dis3)
    dis4 = LeakyReLU(alpha=leakyrelu_alpha)(dis4)
    dis4 = BatchNormalization(momentum=0.8)(dis4)

    dis5 = Conv2D(256, kernel_size=3, strides=1, padding='same')(dis4)
    dis5 = LeakyReLU(alpha=leakyrelu_alpha)(dis5)
    dis5 = BatchNormalization(momentum=momentum)(dis5)

    dis6 = Conv2D(filters=256, kernel_size=3, strides=2, padding='same')(dis5)
    dis6 = LeakyReLU(alpha=leakyrelu_alpha)(dis6)
    dis6 = BatchNormalization(momentum=momentum)(dis6)

    dis7 = Conv2D(filters=512, kernel_size=3, strides=1, padding='same')(dis6)
    dis7 = LeakyReLU(alpha=leakyrelu_alpha)(dis7)
    dis7 = BatchNormalization(momentum=momentum)(dis7)

    dis8 = Conv2D(filters=512, kernel_size=3, strides=2, padding='same')(dis7)
    dis8 = LeakyReLU(alpha=leakyrelu_alpha)(dis8)
    dis8 = BatchNormalization(momentum=momentum)(dis8)

    # fully connected layer
    dis9 = Dense(units=1024)(dis8)
    dis9 = LeakyReLU(alpha=0.2)(dis9)

    # last fully connected layer - for classification
    output = Dense(units=1, activation='sigmoid')(dis9)

    model = Model(inputs=[input_layer], outputs=[output], name='discriminator')

    return model


# %% [code] {"jupyter":{"outputs_hidden":false}}
discriminator = build_discriminator()
discriminator.trainable = True
discriminator.compile(
    loss="mse", optimizer=common_optimizer, metrics=['accuracy'])

# %% [code] {"jupyter":{"outputs_hidden":false}}
!wget https: // github.com/fchollet/deep-learning-models/releases/download/v0.1/vgg19_weights_tf_dim_ordering_tf_kernels.h5

# %% [code] {"jupyter":{"outputs_hidden":false}}
VGG19_base = VGG19(weights="./vgg19_weights_tf_dim_ordering_tf_kernels.h5")

# %% [code] {"jupyter":{"outputs_hidden":false}}
VGG19_base.summary()

# %% [code] {"jupyter":{"outputs_hidden":false}}


def build_VGG19():

    input_shape = (256, 256, 3)
    VGG19_base.outputs = [VGG19_base.get_layer('block5_conv2').output]
    # VGG19_base.outputs = [VGG19_base.layers[9].output]
    input_layer = Input(shape=input_shape)
    features = VGG19_base(input_layer)
    model = Model(inputs=[input_layer], outputs=[features])

    return model

# %% [code] {"jupyter":{"outputs_hidden":false}}
# Combining them all to build adversarial network


# %% [code] {"jupyter":{"outputs_hidden":false}}
fe_model = build_VGG19()
fe_model.trainable = False
fe_model.compile(loss='mse', optimizer=common_optimizer, metrics=['accuracy'])

# %% [code] {"jupyter":{"outputs_hidden":false}}
# Final Adversarial Network


def build_adversarial_model(generator, discriminator, feature_extractor):

    input_high_resolution = Input(shape=high_resolution_shape)

    input_low_resolution = Input(shape=low_resolution_shape)

    generated_high_resolution_images = generator(input_low_resolution)

    features = feature_extractor(generated_high_resolution_images)

    # make a discriminator non trainable
    discriminator.trainable = False
    discriminator.compile(
        loss='mse', optimizer=common_optimizer, metrics=['accuracy'])

    # discriminator will give a prob estimation for generated high-resolution images
    probs = discriminator(generated_high_resolution_images)

    # create and compile
    adversarial_model = Model(
        [input_low_resolution, input_high_resolution], [probs, features])
    adversarial_model.compile(loss=['binary_crossentropy', 'mse'], loss_weights=[
                              1e-3, 1], optimizer=common_optimizer)

    return adversarial_model


# %% [code] {"jupyter":{"outputs_hidden":false}}
adversarial_model = build_adversarial_model(generator, discriminator, fe_model)

# %% [markdown]
# ## Model Training

# %% [code] {"jupyter":{"outputs_hidden":false}}
losses = {'d_history': [], "g_history": []}

psnr = {'psnr_quality': []}
ssim = {'ssim_quality': []}

# %% [code] {"jupyter":{"outputs_hidden":false}}


def get_train_images():

    # image_list = glob('../input/div2k-dataset/DIV2K_train_HR/DIV2K_train_HR/*')
    image_list = glob(
        '../input/dataset-image-super-resolution/finished/train/dataraw/hires/*')
    return image_list

# %% [code] {"jupyter":{"outputs_hidden":false}}


# %% [code] {"jupyter":{"outputs_hidden":false}}
# hr_images, lr_images = sample_images(image_list, batch_size = batch_size, low_resolution_shape = low_resolution_shape, high_resolution_shape = high_resolution_shape)

# %% [code] {"jupyter":{"outputs_hidden":false}}


def save_images(original_image, lr_image, sr_image, path):
    """
    Save LR, HR (original) and generated SR
    images in one panel
    """

    fig, ax = plt.subplots(1, 3, figsize=(10, 6))

    images = [original_image, lr_image, sr_image]
    titles = ['HR', 'LR', 'SR - generated']

    for idx, img in enumerate(images):
        # (X + 1)/2 to scale back from [-1,1] to [0,1]
        ax[idx].imshow((img + 1)/2.0, cmap='gray')
        ax[idx].axis("off")
    for idx, title in enumerate(titles):
        ax[idx].set_title('{}'.format(title))

    plt.savefig(path)

# %% [markdown]
# ### Only training for 3 epochs. This typically requires 30000 epochs. Kaggle NBs don't support this for GPU.
#
# ## TPU Implementation will be shared soon


# %% [code] {"jupyter":{"outputs_hidden":false}}
epochs = 3

for epoch in range(epochs):

    print("Epoch Number -> ", epoch)
    d_history = []
    g_history = []

    image_list = get_train_images()

    """
    Discriminator training
    """

    hr_images, lr_images = sample_images(
        image_list, batch_size=batch_size, low_resolution_shape=low_resolution_shape, high_resolution_shape=high_resolution_shape)

    # normalize images

    hr_images = hr_images / 127.5 - 1
    lr_images = lr_images / 127.5 - 1

    # generate high-resolution images from low-resolution images
    generated_high_resolution_images = generator.predict(lr_images)

    print("Generated high res images ...\n")
    # generate a batch of true and fake labels
    real_labels = np.ones((batch_size, 16, 16, 1))
    fake_labels = np.zeros((batch_size, 16, 16, 1))

    d_loss_real = discriminator.train_on_batch(hr_images, real_labels)
    d_loss_real = np.mean(d_loss_real)

    d_loss_fake = discriminator.train_on_batch(
        generated_high_resolution_images, fake_labels)
    d_loss_fake = np.mean(d_loss_fake)

    print("Discriminator Losses calculated...\n")
    d_loss = 0.5 * np.add(d_loss_real, d_loss_fake)
    losses['d_history'].append(d_loss)

    """
    Train the generator network
    """

    # hr_images, lr_images = sample_images(image_list, batch_size = batch_size, low_resolution_shape = low_resolution_shape,
    # high_resolution_shape = high_resolution_shape)

    # normalize the images
    # hr_images = hr_images/127.5 - 1
    # lr_images = lr_images/127.5 - 1

    # extract feature maps for true high-resolution images
    image_features = fe_model.predict(hr_images)

    # train the generator
    g_loss = adversarial_model.train_on_batch([lr_images, hr_images],
                                              [real_labels, image_features])

    losses['g_history'].append(0.5 * (g_loss[1]))
    print("Generator trained...\n")

    print("\n Calculating PSNR")
    # psnr
    ps = compute_psnr(hr_images, generated_high_resolution_images)
    psnr['psnr_quality'].append(ps)

    # ssim

    print("\nCalculating SSIM\n")
    ss = compute_ssim(hr_images, generated_high_resolution_images)
    ssim['ssim_quality'].append(ss)

    """
    save and print image samples
    """
    print("Epoch Completed...\n\n")
    if epoch % 2 == 0:
        hr_images, lr_images = sample_images(
            image_list, batch_size=batch_size, low_resolution_shape=low_resolution_shape, high_resolution_shape=high_resolution_shape)

        generated_images = generator.predict_on_batch(lr_images)

        for index, img in enumerate(generated_images):

            if index < 3:
                save_images(hr_images[index], lr_images[index], img,
                            path="/kaggle/working/img_{}_{}".format(epoch, index))

# %% [code] {"jupyter":{"outputs_hidden":false}}
discriminator.summary()

# %% [markdown]
# plot the training loss, psnr, ssim

# %% [code] {"jupyter":{"outputs_hidden":false}}
plot_loss(losses)
plot_psnr(psnr)
plot_ssim(ssim)

# %% [code] {"jupyter":{"outputs_hidden":false}}
generator.save_weights("/kaggle/working/srgan_generator.h5")
discriminator.save_weights("/kaggle/working/srgan_discriminator.h5")

# %% [code] {"jupyter":{"outputs_hidden":false}}


# %% [markdown]
# # Summary
#
# SRGANs were introduced in 2017 as State of the art for Super Image Resolution
#
# In 2018, Enhanced SRGAN (ESRGAN) were introduced. We'll learn about them in the upcoming notebooks.

# %% [markdown]
# # Note:- This can be replicated for TPU training. Stay tuned for TPU Notebook implementation. Will be shared soon.
