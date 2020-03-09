import tensorflow.keras.layers as layers


def conv_bn_relu(
        filters, kernel_size,
        strides=(1, 1), padding='same',
        use_bias=True):
    def layer(input):
        conv = layers.Conv2D(
            filters=filters, kernel_size=kernel_size,
            strides=strides, padding=padding,
            use_bias=use_bias)(input)
        bn = layers.BatchNormalization()(conv)
        relu = layers.ReLU()(bn)
        return relu

    return layer
