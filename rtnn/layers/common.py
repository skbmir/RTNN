import tensorflow as tf


def conv2d_bn_relu(
        filters,
        padding='same',
        use_bias=True,
        kernel=(3, 3),
        strides=(1, 1),
        kernel_initializer='he_normal',
        trainable=True
):
    def layer(input):
        conv2d = tf.keras.layers.Conv2D(filters=filters, kernel_size=kernel, padding=padding, strides=strides,
                                        use_bias=use_bias,
                                        kernel_initializer=kernel_initializer,
                                        trainable=trainable)(input)
        conv2d_bn = tf.keras.layers.BatchNormalization()(conv2d)
        conv2d_bn_relu = tf.keras.layers.ReLU()(conv2d_bn)
        return conv2d_bn_relu

    return layer


def sep2d_bn_relu(
        filters,
        padding='same',
        use_bias=True,
        kernel=(3, 3),
        strides=(1, 1),
        kernel_initializer='he_normal',
        trainable=True
):
    def layer(input):
        conv2d = tf.keras.layers.SeparableConv2D(filters=filters, kernel_size=kernel, padding=padding, strides=strides,
                                                 use_bias=use_bias,
                                                 kernel_initializer=kernel_initializer,
                                                 trainable=trainable)(input)
        conv2d_bn = tf.keras.layers.BatchNormalization()(conv2d)
        conv2d_bn_relu = tf.keras.layers.ReLU()(conv2d_bn)
        return conv2d_bn_relu

    return layer
