import tensorflow as tf
from rtnn.layers.bottlenecks import mobilenetv2_bottleneck
from rtnn.layers.common import conv2d_bn_relu, sep2d_bn_relu


class ContextNet:
    def __init__(self, in_shape=(1024, 2048, 3), n_classes=2, low_res_scale=0.25):
        self.in_shape = in_shape
        self.low_res_scale = int(1. / low_res_scale)

    def build(self):
        hr_input = tf.keras.layers.Input(shape=self.in_shape)

        #
        # High resolution branch (1024x2048 as default) contains:
        # conv2d_bn_relu - Conv2D + BatchNormalization + ReLU
        # sep2d_bn_relu x 3 - SeparableConv2D + BatchNormalization + ReLU
        hr_conv = conv2d_bn_relu(filters=32, kernel=(3, 3), strides=(2, 2), use_bias=False)(hr_input)
        hr_sep_1 = sep2d_bn_relu(filters=64, kernel=(3, 3), strides=(2, 2), use_bias=False)(hr_conv)
        hr_sep_2 = sep2d_bn_relu(filters=128, kernel=(3, 3), strides=(2, 2), use_bias=False)(hr_sep_1)
        hr_out = sep2d_bn_relu(filters=128, kernel=(3, 3), strides=(2, 2), use_bias=False)(hr_sep_2)

        #
        # Low resolution branch (256x512 as default) contains:
        # AveragePooling2D - 1028x2048 -> 256x512
        # conv2d_bn_relu - Conv2D + BatchNormalization + ReLU
        # mobilenetv2_bottleneck x 5 - Conv2D + ReLU6 + DepthwiseConv2D + ReLU6 + Conv2D
        lr_input = tf.keras.layers.AveragePooling2D(pool_size=(self.low_res_scale, self.low_res_scale))(hr_input)
        lr_conv1 = conv2d_bn_relu(filters=32, kernel=(3, 3), strides=(2, 2), use_bias=False)(lr_input)

        # bottleneck block #1
        lr_bottleneck1_1 = mobilenetv2_bottleneck(in_filters=32, expansion_factor=1, out_filters=32, dw_kernel=(3, 3),
                                                  strides=(1, 1))(lr_conv1)
        lr_bottleneck1_2 = mobilenetv2_bottleneck(in_filters=32, expansion_factor=6, out_filters=32, dw_kernel=(3, 3),
                                                  strides=(1, 1))(lr_bottleneck1_1)

        # bottleneck block #2
        lr_bottleneck2_1 = mobilenetv2_bottleneck(in_filters=32, expansion_factor=6, out_filters=48, dw_kernel=(3, 3),
                                                  strides=(2, 2))(lr_bottleneck1_2)
        lr_bottleneck2_2 = mobilenetv2_bottleneck(in_filters=48, expansion_factor=6, out_filters=48, dw_kernel=(3, 3),
                                                  strides=(2, 2))(lr_bottleneck2_1)
        lr_bottleneck2_3 = mobilenetv2_bottleneck(in_filters=48, expansion_factor=6, out_filters=48, dw_kernel=(3, 3),
                                                  strides=(2, 2))(lr_bottleneck2_2)

        # bottleneck block #3
        lr_bottleneck3_1 = mobilenetv2_bottleneck(in_filters=48, expansion_factor=6, out_filters=64, dw_kernel=(3, 3),
                                                  strides=(2, 2))(lr_bottleneck2_3)
        lr_bottleneck3_2 = mobilenetv2_bottleneck(in_filters=64, expansion_factor=6, out_filters=64, dw_kernel=(3, 3),
                                                  strides=(2, 2))(lr_bottleneck3_1)
        lr_bottleneck3_3 = mobilenetv2_bottleneck(in_filters=64, expansion_factor=6, out_filters=64, dw_kernel=(3, 3),
                                                  strides=(2, 2))(lr_bottleneck3_2)

        # bottleneck block #4
        lr_bottleneck4_1 = mobilenetv2_bottleneck(in_filters=64, expansion_factor=6, out_filters=96, dw_kernel=(3, 3),
                                                  strides=(1, 1))(lr_bottleneck3_3)
        lr_bottleneck4_2 = mobilenetv2_bottleneck(in_filters=96, expansion_factor=6, out_filters=96, dw_kernel=(3, 3),
                                                  strides=(1, 1))(lr_bottleneck4_1)
        # bottleneck block #5
        lr_bottleneck5_1 = mobilenetv2_bottleneck(in_filters=96, expansion_factor=6, out_filters=128, dw_kernel=(3, 3),
                                                  strides=(1, 1))(lr_bottleneck4_2)
        lr_bottleneck5_2 = mobilenetv2_bottleneck(in_filters=128, expansion_factor=6, out_filters=128, dw_kernel=(3, 3),
                                                  strides=(1, 1))(lr_bottleneck5_1)

        lr_out = conv2d_bn_relu(filters=128, kernel=(3, 3), use_bias=False)(lr_bottleneck5_2)

        #
        # Feature fusion block to concat
        ff_hr_conv = tf.keras.layers.Conv2D(filters=128, kernel_size=(1, 1), padding='same', use_bias=False)(hr_out)

        ff_lr_up = tf.keras.layers.UpSampling2D(size=(self.low_res_scale, self.low_res_scale))(lr_out)
        ff_lr_dw = tf.keras.layers.DepthwiseConv2D(kernel_size=(3, 3),
                                                   dilation_rate=(self.low_res_scale, self.low_res_scale),
                                                   padding='same',
                                                   use_bias=False,
                                                   depthwise_initializer='he_normal')(ff_lr_up)
        ff_lr_conv = tf.keras.layers.Conv2D(filters=128, kernel_size=(1, 1), padding='same', use_bias=False)(ff_lr_dw)

        #
        # Concat high and low resolution features and final
        add = tf.keras.layers.add([ff_hr_conv, ff_lr_conv])
        drop = tf.keras.layers.Dropout(rate=0.25)(add)
        conv = tf.keras.layers.Conv2D(filters=self.num_classes, kernel=(1, 1), strides=(1, 1), use_bias=False)(drop)
        softmax_out = tf.keras.layers.Softmax()(conv)

        return tf.keras.models.Model(inputs=hr_input, outputs=softmax_out)
