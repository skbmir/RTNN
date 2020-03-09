import numpy as np
import tensorflow.keras as K
import tensorflow.keras.layers as layers
from rtnn.layers.common import conv_bn_relu


class network(K.Model):
    def __init__(self, in_shape):
        super(network, self).__init__(in_shape)
        self.in_shape = in_shape

    def call(self, inputs, training=None, mask=None):
        input = layers.Input(shape=self.in_shape)

        # First block if convs
        conv_0 = conv_bn_relu(
            filters=16, kernel_size=(3, 3),
            strides=(2, 2))(input)
        conv_1 = conv_bn_relu(
            filters=24, kernel_size=(3, 3)
        )(conv_0)
        conv_2 = conv_bn_relu(
            filters=32, kernel_size=(3, 3),
            strides=(2, 2))(conv_1)
        conv_3 = conv_bn_relu(
            filters=48, kernel_size=(3, 3)
        )(conv_2)

        # Hard BlockX4
        hard_block_x4_1 = self.hard_block_x4(filters=(10, 18, 10, 28))(conv_3)
        conv_4 = conv_bn_relu(
            filters=64, kernel_size=(1, 1)
        )(hard_block_x4_1)
        pool_1 = layers.AveragePooling2D(pool_size=(2, 2), strides=(2, 2))(conv_4)

        # Hard BlockX4
        hard_block_x4_2 = self.hard_block_x4(filters=(16, 28, 16, 46))(pool_1)
        conv_5 = conv_bn_relu(
            filters=96, kernel_size=(1, 1)
        )(hard_block_x4_2)
        pool_2 = layers.AveragePooling2D(pool_size=(2, 2), strides=(2, 2))(conv_5)

        # Hard BlockX8
        hard_block_x8_1 = self.hard_block_x8(filters=(18, 30, 18, 52, 18, 30, 18, 88))(pool_2)
        conv_6 = conv_bn_relu(
            filters=160, kernel_size=(1, 1)
        )(hard_block_x8_1)
        pool_3 = layers.AveragePooling2D(pool_size=(2, 2), strides=(2, 2))(conv_6)

        # Hard BlockX8
        hard_block_x8_2 = self.hard_block_x8(filters=(24, 40, 24, 70, 24, 40, 24, 118))(pool_3)
        conv_7 = conv_bn_relu(
            filters=224, kernel_size=(1, 1)
        )(hard_block_x8_2)
        pool_4 = layers.AveragePooling2D(pool_size=(2, 2), strides=(2, 2))(conv_7)

        # Hard BlockX8
        hard_block_x8_3 = self.hard_block_x8(filters=(32, 54, 32, 92, 32, 54, 32, 158))(pool_4)
        conv_8 = conv_bn_relu(
            filters=320, kernel_size=(1, 1)
        )(hard_block_x8_3)

        # UpSample block
        trans_up_1 = layers.UpSampling2D(interpolation='bilinear', size=(2, 2))(conv_8)
        concat_1 = layers.concatenate([trans_up_1, hard_block_x8_2], axis=-1)
        conv_9 = conv_bn_relu(
            filters=267, kernel_size=(1, 1)
        )(concat_1)

        # UpSample block
        hard_block_x8_4 = self.hard_block_x8(filters=(24, 40, 24, 70, 24, 40, 24, 118))(conv_9)
        trans_up_2 = layers.UpSampling2D(interpolation='bilinear', size=(2, 2))(hard_block_x8_4)
        concat_2 = layers.concatenate([trans_up_2, hard_block_x8_1], axis=-1)
        conv_10 = conv_bn_relu(
            filters=187, kernel_size=(1, 1)
        )(concat_2)

        # UpSample block
        hard_block_x8_5 = self.hard_block_x8(filters=(18, 30, 18, 52, 18, 30, 18, 88))(conv_10)
        trans_up_3 = layers.UpSampling2D(interpolation='bilinear', size=(2, 2))(hard_block_x8_5)
        concat_3 = layers.concatenate([trans_up_3, hard_block_x4_2], axis=-1)
        conv_11 = conv_bn_relu(
            filters=119, kernel_size=(1, 1)
        )(concat_3)

        # UpSample block
        hard_block_x4_3 = self.hard_block_x4(filters=(16, 28, 16, 46))(conv_11)
        trans_up_4 = layers.UpSampling2D(interpolation='bilinear', size=(2, 2))(hard_block_x4_3)
        concat_4 = layers.concatenate([trans_up_4, hard_block_x4_1], axis=-1)
        conv_12 = conv_bn_relu(
            filters=63, kernel_size=(1, 1)
        )(concat_4)

        hard_block_x4_4 = self.hard_block_x4(filters=(10, 18, 10, 28))(conv_12)
        conv_13 = conv_bn_relu(
            filters=3, kernel_size=(1, 1)
        )(hard_block_x4_4)
        concat_5 = layers.concatenate([input, conv_13], axis=1)

        return concat_5

    def hard_block_x4(self, filters=()):
        def block(input):
            conv_0 = conv_bn_relu(filters=filters[0], kernel_size=(3, 3))(input)
            concat_0 = layers.concatenate([input, conv_0], axis=-1)
            conv_1 = conv_bn_relu(filters=filters[1], kernel_size=(3, 3))(concat_0)
            conv_2 = conv_bn_relu(filters=filters[2], kernel_size=(3, 3))(conv_1)
            concat_1 = layers.concatenate([input, conv_1, conv_2], axis=-1)
            conv_3 = conv_bn_relu(filters=filters[2], kernel_size=(3, 3))(concat_1)
            concat_2 = layers.concatenate([conv_0, conv_2, conv_3], axis=-1)
            return concat_2

        return block

    def hard_block_x8(self, filters=()):
        def block(input):
            conv_0 = conv_bn_relu(filters=filters[0], kernel_size=(3, 3))(input)
            concat_0 = layers.concatenate([input, conv_0], axis=-1)
            conv_1 = conv_bn_relu(filters=filters[1], kernel_size=(3, 3))(concat_0)
            conv_2 = conv_bn_relu(filters=filters[2], kernel_size=(3, 3))(conv_1)
            concat_1 = layers.concatenate([input, conv_1, conv_2], axis=-1)
            conv_3 = conv_bn_relu(filters=filters[3], kernel_size=(3, 3))(concat_1)
            conv_4 = conv_bn_relu(filters=filters[4], kernel_size=(3, 3))(conv_3)
            concat_2 = layers.concatenate([conv_3, conv_4], axis=-1)
            conv_5 = conv_bn_relu(filters=filters[5], kernel_size=(3, 3))(concat_2)
            conv_6 = conv_bn_relu(filters=filters[6], kernel_size=(3, 3))(conv_5)
            concat_3 = layers.concatenate([input, conv_3, conv_5, conv_6], axis=-1)
            conv_7 = conv_bn_relu(filters=filters[7], kernel_size=(3, 3))(concat_3)
            concat_4 = layers.concatenate([conv_0, conv_2, conv_4, conv_6, conv_7], axis=-1)
            return concat_4

        return block


if __name__ == '__main__':
    model = network(in_shape=(224, 224, 3))
    image = np.ones((224, 224, 3))
    out = model([image])
    print(model)
