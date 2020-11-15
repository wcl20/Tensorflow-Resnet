from tensorflow.keras import backend as K
from tensorflow.keras.layers import Activation
from tensorflow.keras.layers import add
from tensorflow.keras.layers import AveragePooling2D
from tensorflow.keras.layers import BatchNormalization
from tensorflow.keras.layers import Conv2D
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import Dropout
from tensorflow.keras.layers import Flatten
from tensorflow.keras.layers import Input
from tensorflow.keras.layers import MaxPooling2D
from tensorflow.keras.layers import ZeroPadding2D
from tensorflow.keras.models import Model
from tensorflow.keras.regularizers import l2


class ResNet:

    @staticmethod
    def residual_module(x, filters, strides, channel_dim, reduce=False, reg=0.0001, bn_epsilon=2e-5, bn_momentum=0.9):

        shortcut = x

        x = BatchNormalization(axis=channel_dim, epsilon=bn_epsilon, momentum=bn_momentum)(x)
        x = Activation("relu")(x)
        if reduce:
            shortcut = Conv2D(filters, (1, 1), strides=strides, use_bias=False, kernel_regularizer=l2(reg))(x)
        x = Conv2D(int(filters * 0.25), (1, 1), use_bias=False, kernel_regularizer=l2(reg))(x)

        x = BatchNormalization(axis=channel_dim, epsilon=bn_epsilon, momentum=bn_momentum)(x)
        x = Activation("relu")(x)
        x = Conv2D(int(filters * 0.25), (3, 3), strides=strides, padding="same", use_bias=False, kernel_regularizer=l2(reg))(x)

        x = BatchNormalization(axis=channel_dim, epsilon=bn_epsilon, momentum=bn_momentum)(x)
        x = Activation("relu")(x)
        x = Conv2D(filters, (1, 1), use_bias=False, kernel_regularizer=l2(reg))(x)


        x = add([x, shortcut])
        return x

    @staticmethod
    def build(height, width, channels, classes, stages, filters, reg=0.0001, bn_epsilon=2e-5, bn_momentum=0.9, dataset="cifar"):
        input_shape = (height, width, channels)
        channel_dim = -1
        if K.image_data_format() == "channels_first":
            input_shape = (channels, height, width)
            channel_dim = 1

        inputs = Input(shape=input_shape)
        x = BatchNormalization(axis=channel_dim, epsilon=bn_epsilon, momentum=bn_momentum)(inputs)

        if dataset == 'cifar':
            x = Conv2D(filters[0], (3, 3), use_bias=False, padding="same", kernel_regularizer=l2(reg))(x)
        elif dataset == 'tiny-imagenet':
            # Tiny imagenet has larger image, add additional layers to reduce dimensions
            x = Conv2D(filters[0], (5, 5), use_bias=False, padding="same", kernel_regularizer=l2(reg))(x)
            x = BatchNormalization(axis=channel_dim, epsilon=bn_epsilon, momentum=bn_momentum)(x)
            x = Activation("relu")(x)
            x = ZeroPadding2D((1, 1))(x)
            x = MaxPooling2D((3, 3), strides=(2, 2))(x)

        for i in range(len(stages)):
            # Perform downsampling in first module of every stage except the first stage
            strides = (1, 1) if i == 0 else (2, 2)
            x = ResNet.residual_module(x, filters[i + 1], strides, channel_dim, reduce=True, bn_epsilon=bn_epsilon, bn_momentum=bn_momentum)
            for _ in range(stages[i] - 1):
                x = ResNet.residual_module(x, filters[i + 1], (1, 1), channel_dim, bn_epsilon=bn_epsilon, bn_momentum=bn_momentum)

        x = BatchNormalization(axis=channel_dim, epsilon=bn_epsilon, momentum=bn_momentum)(x)
        x = Activation("relu")(x)
        x = AveragePooling2D((8, 8))(x)

        x = Flatten()(x)
        x = Dense(classes, kernel_regularizer=l2(reg), activation="softmax")(x)
        return Model(inputs, x, name="resnet")
