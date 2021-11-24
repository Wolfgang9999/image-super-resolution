import tensorflow as tf


def residual_block_gen(inp, ch=64, k_s=3, n_blocks=4):
    concat = inp
    for x in range(n_blocks):
        out = tf.keras.layers.Conv2D(ch, k_s, padding='same')(concat)
        out = tf.keras.layers.PReLU(shared_axes=[1, 2])(out)

        concat = tf.keras.layers.concatenate([concat, out])

    out = tf.keras.layers.Conv2D(ch, k_s, padding='same')(concat)
    return out


def Upsample_block(x, ch=256, k_s=3, st=1):
    x = tf.keras.layers.Conv2D(ch, k_s, strides=(st, st), padding='same')(x)
    x = tf.nn.depth_to_space(x, 2)  # Subpixel pixelshuffler
    x = tf.keras.layers.LeakyReLU()(x)
    return x


residual_scaling = 0.2

input_lr = tf.keras.layers.Input(shape=(None, None, 3))
input_conv = tf.keras.layers.Conv2D(64, 9, padding='same')(input_lr)
input_conv = tf.keras.layers.PReLU(shared_axes=[1, 2])(input_conv)

ESRRes = input_conv
for x in range(5):
    res_output = residual_block_gen(ESRRes)
    ESRRes = tf.keras.layers.Add()([ESRRes, res_output * residual_scaling])

ESRRes = tf.keras.layers.Conv2D(64, 3, padding='same')(ESRRes)
ESRRes = tf.keras.layers.BatchNormalization()(ESRRes)
ESRRes = tf.keras.layers.Add()([ESRRes, input_conv])

ESRRes = Upsample_block(ESRRes)
ESRRes = Upsample_block(ESRRes)

output_sr = tf.keras.layers.Conv2D(
    3, 9, activation='tanh', padding='same')(ESRRes)

ESRGAN = tf.keras.models.Model(input_lr, output_sr)

cross_entropy = tf.keras.losses.BinaryCrossentropy(from_logits=True)


def RAGAN_discriminator_loss(real_output, fake_output):
    real_loss = cross_entropy(tf.ones_like(
        real_output), real_output - fake_output)
    fake_loss = cross_entropy(tf.zeros_like(
        fake_output), fake_output - real_output)
    total_loss = real_loss + fake_loss
    return total_loss


def RAGAN_generator_loss(real_output, fake_output):
    return cross_entropy(tf.ones_like(fake_output), fake_output - real_output)


VGG19 = tf.keras.applications.VGG19(
    weights='imagenet', include_top=False, input_shape=(None, None, 3))

VGG_i, VGG_j = 5, 4


def VGG_partial(i_m=2, j_m=2):
    i, j = 1, 0
    accumulated_loss = 0.0
    for l in VGG19.layers:
        cl_name = l.__class__.__name__
        if cl_name == 'Conv2D':
            j += 1
        if cl_name == 'MaxPooling2D':
            i += 1
            j = 0
        if i == i_m and j == j_m and cl_name == 'Conv2D':
            before_act_output = tf.nn.convolution(
                l.input, l.weights[0], padding='SAME') + l.weights[1]
            return tf.keras.models.Model(VGG19.input, before_act_output)


partial_VGG = VGG_partial(VGG_i, VGG_j)
