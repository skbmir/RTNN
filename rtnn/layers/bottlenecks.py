import tensorflow as tf


def mobilenetv2_bottleneck(
        in_filters,
        out_filters,
        expansion_factor=1,
        dw_kernel=(3, 3),
        strides=(2, 2),
        padding='same',
        use_bias=True,
        kernel_initializer='he_normal',
        trainable=True
):
    def layer(input):
        conv = tf.keras.layers.Conv2D(filters=int(in_filters * expansion_factor), kernel_size=(1, 1), padding=padding,
                                      use_bias=use_bias,
                                      kernel_initializer=kernel_initializer,
                                      trainable=trainable,
                                      name='mobilenetv2_bottleneck_conv_input')(input)
        conv_relu = tf.nn.relu6(features=conv, name='mobilenetv2_bottleneck_relu6_1')
        depthwise = tf.keras.layers.DepthwiseConv2D(kernel_size=dw_kernel, strides=strides, padding=padding,
                                                    use_bias=use_bias, kernel_initializer=kernel_initializer,
                                                    trainable=trainable,
                                                    name='mobilenetv2_bottleneck_dwconv')(conv_relu)
        depthwise_relu6 = tf.nn.relu6(features=depthwise, name='mobilenetv2_bottleneck_relu6_2')
        conv = tf.keras.layers.Conv2D(filters=out_filters, kernel_size=(1, 1), padding=padding,
                                      use_bias=use_bias,
                                      kernel_initializer=kernel_initializer,
                                      trainable=trainable,
                                      name='mobilenetv2_bottleneck_conv_out')(depthwise_relu6)
        return conv

    return layer
