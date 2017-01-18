from keras.models import Model
from keras.layers import Input, Activation, merge, Dense, Flatten
from keras.layers.convolutional import Convolution2D, MaxPooling2D, AveragePooling2D
from keras.layers.normalization import BatchNormalization

# build a conv-bn-relu block
def _conv_bn_relu(nb_filter, nb_row, nb_col, subsample=(1,1)): # subsample=filter
    def f(input):
        conv = Convolution2D(nb_filter=nb_filter, nb_row=nb_row, nb_col=nb_col, subsample=subsample, init="he_normal",
                             border_mode="same")(input) # no padding and init method is adjustable
        norm = BatchNormalization(mode=0, axis=3)(conv) # featre-wise normalization
        return Activation("relu")(norm)
    return f

# build a bn-relu-conv block
def _bn_relu_conv(nb_filter, nb_row, nb_col, subsample=(1,1)):
    def f(input):
        norm = BatchNormalization(mode=0, axis=3)(input)
        activation = Activation("relu")(norm)
        return Convolution2D(nb_filter=nb_filter, nb_row=nb_row, nb_col=nb_col, subsample=subsample, init="he_normal",
                            border_mode="same")(activation)
    return f

# add a shortcut between input and residual block and merge them
def _shortcut(input, residual):
    # expand channels of shortcut to match residual
    stride_width = input._keras_shape[1] / residual._keras_shape[1]
    stride_height = input._keras_shape[2] / residual._keras_shape[2]
    equal_channels = input._keras_shape[3] == residual._keras_shape[3]
    
    shortcut = input
    
    if stride_width > 1 or stride_height > 1 or not equal_channels:
        shortcut = Convolution2D(nb_filter=residual._keras_shape[3], nb_row=1, nb_col=1,
                                 subsample=(stride_width, stride_height), init="he_normal", border_mode="valid")(input)
        # border mode does not matter
    
    return merge([shortcut, residual], mode="sum")

def _residual_block(block_function, nb_filter, repetitions, is_first_layer=False):
    def f(input):
        for i in range(repetitions):
            init_subsample = (1, 1)
            if i == 0 and not is_first_layer:
                init_subsample = (2, 2)
            input = block_function(nb_filter, init_subsample=init_subsample)(input)
        return input
    
    return f

# for layers < 34
def basic_block(nb_filter, init_subsample=(1, 1)):
    def f(input):
        conv1 = _bn_relu_conv(nb_filter, 3, 3, subsample=init_subsample)(input)
        residual = _bn_relu_conv(nb_filter, 3, 3)(conv1)
        return _shortcut(input, residual)
    return f

# for layers > 34, mentioned in paper
def bottleneck(nb_filter, init_subsample=(1, 1)):
    def f(input):
        conv_1_1 = _bn_relu_conv(nb_filter, 1, 1, subsample=init_subsample)(input)
        conv_3_3 = _bn_relu_conv(nb_filter, 3, 3)(conv_1_1)
        residual = _bn_relu_conv(nb_filter * 4, 1, 1)(conv_3_3)
        return _shortcut(input, residual)
    return f

class ResnetBuilder(object):
    @staticmethod
    def build(input_shape, num_outputs, block_fn, repetitions):
        """
        Builds a custom ResNet like architecture.
        :param input_shape: The input shape in the form (nb_channels, nb_rows, nb_cols)
        :param num_outputs: The number of outputs at final softmax layer
        :param block_fn: The block function to use. This is either :func:`basic_block` or :func:`bottleneck`.
        The original paper used basic_block for layers < 50
        :param repetitions: Number of repetitions of various block units.
        At each block unit, the number of filters are doubled and the input size is halved
        :return: The keras model.
        """
        if len(input_shape) != 3:
            raise Exception("Input shape should be a tuple (nb_rows, nb_cols, nb_channels)")

        input = Input(shape=input_shape)
        conv1 = _conv_bn_relu(nb_filter=64, nb_row=7, nb_col=7, subsample=(2, 2))(input)
        pool1 = MaxPooling2D(pool_size=(3, 3), strides=(2, 2), border_mode="same")(conv1)

        block = pool1
        nb_filter = 64
        for i, r in enumerate(repetitions):
            block = _residual_block(block_fn, nb_filter=nb_filter, repetitions=r, is_first_layer=i == 0)(block)
            nb_filter *= 2

        # Classifier block
        pool2 = AveragePooling2D(pool_size=(block._keras_shape[1],
                                            block._keras_shape[2]),
                                 strides=(1, 1))(block)
        flatten1 = Flatten()(pool2)
        dense = Dense(output_dim=num_outputs, init="he_normal", activation="softmax")(flatten1)

        model = Model(input=input, output=dense)
        return model

    @staticmethod
    def build_resnet_18(input_shape, num_outputs):
        return ResnetBuilder.build(input_shape, num_outputs, basic_block, [2, 2, 2, 2])

    @staticmethod
    def build_resnet_34(input_shape, num_outputs):
        return ResnetBuilder.build(input_shape, num_outputs, basic_block, [3, 4, 6, 3])

    @staticmethod
    def build_resnet_50(input_shape, num_outputs):
        return ResnetBuilder.build(input_shape, num_outputs, bottleneck, [3, 4, 6, 3])

    @staticmethod
    def build_resnet_101(input_shape, num_outputs):
        return ResnetBuilder.build(input_shape, num_outputs, bottleneck, [3, 4, 23, 3])

    @staticmethod
    def build_resnet_152(input_shape, num_outputs):
        return ResnetBuilder.build(input_shape, num_outputs, bottleneck, [3, 8, 36, 3])


def main():
    model = ResnetBuilder.build_resnet_18((3, 224, 224), 1000)
    model.compile(loss="categorical_crossentropy", optimizer="sgd")
    model.summary()


if __name__ == '__main__':
    main()