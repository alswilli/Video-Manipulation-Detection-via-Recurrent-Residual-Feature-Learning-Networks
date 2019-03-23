from keras.layers import Dense, Flatten, Dropout, ZeroPadding3D
from keras.layers.recurrent import LSTM
from keras.models import Sequential, load_model
from keras.optimizers import Adam, RMSprop
from keras.layers.wrappers import TimeDistributed
from keras.layers.convolutional import (Conv2D, MaxPooling3D, Conv3D,
    MaxPooling2D)
# from collections import deque
import config
import sys

from keras.models import Model
from keras.layers import Input, Lambda, Conv2D, MaxPooling2D, BatchNormalization, ELU, Reshape, Concatenate, Activation, Flatten, Dense, ZeroPadding2D, AveragePooling2D, GlobalAveragePooling2D, GlobalMaxPooling2D, Dropout
from keras.regularizers import l2
from keras import layers
import keras.backend as K

class TestModels():
    def __init__(self, nclasses, model, seq_length=config.MIN_SEQ_LENGTH, saved_model=None):
        self.nclasses = nclasses
        self.saved_model = saved_model

        if self.saved_model is not None:
            self.model = load_model(self.saved_model)
        elif model == 'lrcn':
            self.input_shape = (None, config.IMG_WIDTH, config.IMG_HEIGHT, config.IMG_CHANNELS)
            self.model = self.lrcn()
        elif model == 'lrcn_resnet':
            self.input_shape = (None, config.IMG_WIDTH, config.IMG_HEIGHT, config.IMG_CHANNELS)
            self.model = self.lrcn_resnet()
        else:
            print("No such network configuration: {0}" % model)
            sys.exit()
        
        metrics = ['accuracy']
        if self.nclasses >= 10:
            metrics.append('top_k_categorical_accuracy')

        optimizer = Adam(lr=1e-5, decay=1e-6)
        if self.nclasses>2:
            self.model.compile(loss='categorical_crossentropy', optimizer=optimizer, metrics=metrics)
        else:
            self.model.compile(loss='binary_crossentropy', optimizer=optimizer, metrics=metrics)

    def lrcn(self):
        """Build a CNN into RNN.
        Starting version from:
            https://github.com/udacity/self-driving-car/blob/master/
                steering-models/community-models/chauffeur/models.py
        Heavily influenced by VGG-16:
            https://arxiv.org/abs/1409.1556
        Also known as an LRCN:
            https://arxiv.org/pdf/1411.4389.pdf
        """


        model = Sequential()

        model.add(TimeDistributed(Conv2D(32, (7, 7), strides=(2, 2),
            activation='relu', padding='same'), input_shape=self.input_shape))
        model.add(TimeDistributed(Conv2D(32, (3,3),
            kernel_initializer="he_normal", activation='relu')))
        model.add(TimeDistributed(MaxPooling2D((2, 2), strides=(2, 2))))

        model.add(TimeDistributed(Conv2D(64, (3,3),
            padding='same', activation='relu')))
        model.add(TimeDistributed(Conv2D(64, (3,3),
            padding='same', activation='relu')))
        model.add(TimeDistributed(MaxPooling2D((2, 2), strides=(2, 2))))

        # model.add(TimeDistributed(Conv2D(128, (3,3),
        #     padding='same', activation='relu')))
        # model.add(TimeDistributed(Conv2D(128, (3,3),
        #     padding='same', activation='relu')))
        # model.add(TimeDistributed(MaxPooling2D((2, 2), strides=(2, 2))))

        # model.add(TimeDistributed(Conv2D(256, (3,3),
        #     padding='same', activation='relu')))
        # model.add(TimeDistributed(Conv2D(256, (3,3),
        #     padding='same', activation='relu')))
        # model.add(TimeDistributed(MaxPooling2D((2, 2), strides=(2, 2))))
        
        # model.add(TimeDistributed(Conv2D(512, (3,3),
        #     padding='same', activation='relu')))
        # model.add(TimeDistributed(Conv2D(512, (3,3),
        #     padding='same', activation='relu')))
        # model.add(TimeDistributed(MaxPooling2D((2, 2), strides=(2, 2))))

        model.add(TimeDistributed(Flatten()))

        model.add(Dropout(0.5))
        model.add(LSTM(32, return_sequences=False, dropout=0.5))
        model.add(Dense(self.nclasses, activation='softmax'))

        return model
    
    def lrcn_resnet(self):
        """Build a CNN into RNN.
        Starting version from:
            https://github.com/udacity/self-driving-car/blob/master/
                steering-models/community-models/chauffeur/models.py
        Heavily influenced by VGG-16:
            https://arxiv.org/abs/1409.1556
        Also known as an LRCN:
            https://arxiv.org/pdf/1411.4389.pdf
        """


        # model = Sequential()

        img_input = Input(shape=self.input_shape)
        bn_axis = 3

        x = TimeDistributed(ZeroPadding2D((3, 3)))(img_input)
        x = TimeDistributed(Conv2D(64, (7, 7), strides=(2, 2), name='conv1'))(x)
        x = TimeDistributed(BatchNormalization(axis=bn_axis, name='bn_conv1'))(x)
        x = TimeDistributed(Activation('relu'))(x)
        x = TimeDistributed(MaxPooling2D((3, 3), strides=(2, 2)))(x)

        x = conv_block(x, 3, [64, 64, 256], stage=2, block='a', strides=(1, 1))
        x = identity_block(x, 3, [64, 64, 256], stage=2, block='b')
        x = identity_block(x, 3, [64, 64, 256], stage=2, block='c')

        x = conv_block(x, 3, [128, 128, 512], stage=3, block='a')
        x = identity_block(x, 3, [128, 128, 512], stage=3, block='b')
        x = identity_block(x, 3, [128, 128, 512], stage=3, block='c')
        x = identity_block(x, 3, [128, 128, 512], stage=3, block='d')

        x = TimeDistributed(AveragePooling2D((7, 7), name='avg_pool'))(x)

        x = TimeDistributed(Flatten())(x)

        x = Dropout(0.5)(x)
        x = LSTM(32, return_sequences=False, dropout=0.5)(x)
        predictions = Dense(self.nclasses, activation='softmax')(x)

        model = Model(inputs=img_input, outputs = predictions)

        return model

def identity_block(input_tensor, kernel_size, filters, stage, block):
    """The identity block is the block that has no conv layer at shortcut.
    # Arguments
        input_tensor: input tensor
        kernel_size: defualt 3, the kernel size of middle conv layer at main path
        filters: list of integers, the filterss of 3 conv layer at main path
        stage: integer, current stage label, used for generating layer names
        block: 'a','b'..., current block label, used for generating layer names
    # Returns
        Output tensor for the block.
    """
    filters1, filters2, filters3 = filters
    if K.image_data_format() == 'channels_last':
        bn_axis = 3
    else:
        bn_axis = 1
    conv_name_base = 'res' + str(stage) + block + '_branch'
    bn_name_base = 'bn' + str(stage) + block + '_branch'

    x = TimeDistributed(Conv2D(filters1, (1, 1), name=conv_name_base + '2a'))(input_tensor)
    x = TimeDistributed(BatchNormalization(axis=bn_axis, name=bn_name_base + '2a'))(x)
    x = TimeDistributed(Activation('relu'))(x)

    x = TimeDistributed(Conv2D(filters2, kernel_size,
            padding='same', name=conv_name_base + '2b'))(x)
    x = TimeDistributed(BatchNormalization(axis=bn_axis, name=bn_name_base + '2b'))(x)
    x = TimeDistributed(Activation('relu'))(x)

    x = TimeDistributed(Conv2D(filters3, (1, 1), name=conv_name_base + '2c'))(x)
    x = TimeDistributed(BatchNormalization(axis=bn_axis, name=bn_name_base + '2c'))(x)

    x = layers.add([x, input_tensor])
    x = TimeDistributed(Activation('relu'))(x)
    return x


def conv_block(input_tensor, kernel_size, filters, stage, block, strides=(2, 2)):
    """conv_block is the block that has a conv layer at shortcut
    # Arguments
        input_tensor: input tensor
        kernel_size: defualt 3, the kernel size of middle conv layer at main path
        filters: list of integers, the filterss of 3 conv layer at main path
        stage: integer, current stage label, used for generating layer names
        block: 'a','b'..., current block label, used for generating layer names
    # Returns
        Output tensor for the block.
    Note that from stage 3, the first conv layer at main path is with strides=(2,2)
    And the shortcut should have strides=(2,2) as well
    """
    filters1, filters2, filters3 = filters
    if K.image_data_format() == 'channels_last':
        bn_axis = 3
    else:
        bn_axis = 1
    conv_name_base = 'res' + str(stage) + block + '_branch'
    bn_name_base = 'bn' + str(stage) + block + '_branch'

    x = TimeDistributed(Conv2D(filters1, (1, 1), strides=strides,
            name=conv_name_base + '2a'))(input_tensor)
    x = TimeDistributed(BatchNormalization(axis=bn_axis, name=bn_name_base + '2a'))(x)
    x = TimeDistributed(Activation('relu'))(x)

    x = TimeDistributed(Conv2D(filters2, kernel_size, padding='same',
            name=conv_name_base + '2b'))(x)
    x = TimeDistributed(BatchNormalization(axis=bn_axis, name=bn_name_base + '2b'))(x)
    x = TimeDistributed(Activation('relu'))(x)

    x = TimeDistributed(Conv2D(filters3, (1, 1), name=conv_name_base + '2c'))(x)
    x = TimeDistributed(BatchNormalization(axis=bn_axis, name=bn_name_base + '2c'))(x)

    shortcut = TimeDistributed(Conv2D(filters3, (1, 1), strides=strides,
                    name=conv_name_base + '1'))(input_tensor)
    shortcut = TimeDistributed(BatchNormalization(axis=bn_axis, name=bn_name_base + '1'))(shortcut)

    x = layers.add([x, shortcut])
    x = TimeDistributed(Activation('relu'))(x)
    return x