import tensorflow as tf
import imageio as io
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt

# Compute PSNR -> Peak Signal to noise ratio


def compute_psnr(original_image, generated_image):

    original_image = tf.convert_to_tensor(original_image, dtype=tf.float32)
    generated_image = tf.convert_to_tensor(generated_image, dtype=tf.float32)

    psnr = tf.image.psnr(original_image, generated_image, max_val=1.0)

    return tf.math.reduce_mean(psnr, axis=None, keepdims=False, name=None)


def plot_psnr(psnr):

    psnr_means = psnr['psnr_quality']
    plt.figure(figsize=(10, 8))

    plt.plot(psnr_means)
    plt.xlabel('Epochs')
    plt.ylabel('PSNR')
    plt.title('PSNR')
    plt.savefig('PSNR_means.png')

# Compute SSIM -> Structual similarity index


def compute_ssim(original_image, generated_image):

    original_image = tf.convert_to_tensor(original_image, dtype=tf.float32)
    generated_image = tf.convert_to_tensor(generated_image, dtype=tf.float32)

    ssim = tf.image.ssim(original_image, generated_image,
                         max_val=1.0, filter_size=11, filter_sigma=1.5, k1=0.01, )

    return tf.math.reduce_mean(ssim, axis=None, keepdims=False, name=None)


def plot_ssim(ssim):

    ssim_means = ssim['ssim_quality']

    plt.figure(figsize=(10, 8))
    plt.plot(ssim_means)
    plt.xlabel('Epochs')
    plt.ylabel('SSIM')
    plt.title('SSIM')
    plt.savefig('SSIM_means.png')


def plot_loss(losses):

    d_loss = losses['d_history']
    g_loss = losses['g_history']

    plt.figure(figsize=(10, 8))
    plt.plot(d_loss, label="Discriminator loss")
    plt.plot(g_loss, label="Generator Loss")
    plt.xlabel("Epochs")
    plt.ylabel('Loss')
    plt.title("Loss")
    plt.legend()


# Loading Dataset
def loadData(image_list, batch_size, high_resolution_shape, low_resolution_shape):

    images_batch = np.random.choice(image_list, size=batch_size)

    lr_images = []
    hr_images = []

    for img in images_batch:

        img1 = io.imread(img, as_gray=False, pilmode='RGB')
        img1 = img1.astype(np.float32)

        # change the size
        img1_high_resolution = io.imresize(img1, high_resolution_shape)
        img1_low_resolution = io.imresize(img1, low_resolution_shape)

        if np.random.random() < 0.5:
            img1_high_resolution = np.fliplr(img1_high_resolution)
            img1_low_resolution = np.fliplr(img1_low_resolution)

        hr_images.append(img1_high_resolution)
        lr_images.append(img1_low_resolution)

    return np.array(hr_images), np.array(lr_images)
