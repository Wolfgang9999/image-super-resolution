import tensorflow as tf
PATCH_SIZE = 96
LR_SCALE = 4
BATCH_SIZE = 16

buffer_size = 1024
patch_per_image = 128
LOG_STEP=1000
log_dir=logs\ESRGan
model_type='SRGAN_MSE'
FP16=False
image_dtype=tf.float32

use_div2k=True
use_div8k=False

blur_detection=True
MSE_after_bicubic=False
use_noise=True
progressive_training=False
espcn_growing=True
lr_reference=False

plot_PSNR=True
plot_LPIPS=True

init=tf.keras.initializers.GlorotUniform() # MSRA initilization 

if FP16:
  image_dtype=tf.float16
  tf.keras.mixed_precision.set_global_policy('mixed_float16') #<-- Not much benefit for Tesla T4 (7.5 TFLOP)