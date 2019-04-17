import os

from keras.layers import Dense, Flatten, Dropout, ZeroPadding3D
from keras.layers.recurrent import LSTM, GRU
from keras.models import Sequential, load_model
from keras.optimizers import Adam, RMSprop
from keras.layers.wrappers import TimeDistributed
from keras.layers.convolutional import (Conv2D, MaxPooling3D, Conv3D,
    MaxPooling2D)
# from collections import deque
import config
import sys
from keras.layers.core import *
from keras.layers import multiply
from keras.models import *
from keras.layers import concatenate

import tensorflow as tf
from keras import backend as K

def auc(y_true, y_pred):
    auc = tf.metrics.auc(y_true, y_pred)[1]
    K.get_session().run(tf.local_variables_initializer())
    return auc

from keras.models import Model
from keras.layers import Input, Lambda, Conv2D, MaxPooling2D, BatchNormalization, ELU, Reshape, Concatenate, Activation, Flatten, Dense, ZeroPadding2D, AveragePooling2D, GlobalAveragePooling2D, GlobalMaxPooling2D, Dropout, ConvLSTM2D, Bidirectional
from keras.regularizers import l2
from keras import layers
import keras.backend as K
SINGLE_ATTENTION_VECTOR = False

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
        elif model == 'lstm':
            self.model = self.lstm()
        elif model == 'c3d':
            self.input_shape = (None, config.IMG_WIDTH, config.IMG_HEIGHT, config.IMG_CHANNELS)
            self.model = self.c3d()
        elif model == 'conv_lstm':
            self.input_shape = (20, config.IMG_WIDTH, config.IMG_HEIGHT, config.IMG_CHANNELS)
            self.model = self.conv_lstm()
        elif 'conv_lstm' in model:
            self.input_shape = (20, config.IMG_WIDTH, config.IMG_HEIGHT, config.IMG_CHANNELS)
            self.model = eval('self.'+model+'()')
        elif 'lrcn' in model:
            self.input_shape = (20, config.IMG_WIDTH, config.IMG_HEIGHT, config.IMG_CHANNELS)
            self.model = eval('self.'+model+'()')
        else:
            print("No such network configuration: {0}" % model)
            sys.exit()
        
        metrics = ['accuracy']
        if self.nclasses >= 10:
            metrics.append('top_k_categorical_accuracy')

        optimizer = Adam(lr=1e-4, decay=1e-6)
        # optimizer = 'rmsprop'
        if self.nclasses>2:
            self.model.compile(loss='categorical_crossentropy', optimizer=optimizer, metrics=metrics)
        else:
            self.model.compile(loss='binary_crossentropy', optimizer=optimizer, metrics=metrics)
    
    def attention_3d_block(self, inputs):
        # inputs.shape = (batch_size, time_steps, input_dim)
        print(inputs.shape)
        TIME_STEPS = 20
        print(TIME_STEPS)
        # input_dim = inputs.shape[2:]
        a = Permute((2,1))(inputs)
        # a = Reshape((input_dim))(a) # this line is not useful. It's just to know which dimension is what.
        a = Dense(int(TIME_STEPS), activation='softmax')(a)
        a_probs = Permute((2, 1), name='attention_vec')(a)
        # output_attention_mul = concatenate([inputs, a_probs], name='attention_mul', mode='mul')
        output_attention_mul = multiply([inputs, a_probs], name='attention_mul')
        return output_attention_mul

    #LRCN with attention
    def lrcn2(self):
        inputs = Input(shape=self.input_shape)
        # attention_mul = self.attention_3d_block(inputs)
        # flat1 = TimeDistributed(Flatten())(inputs)
        # lstm = LSTM(10, return_sequences=True, dropout=0)(flat1)
        c1 = TimeDistributed(Conv2D(32, (7,7), strides=(2,2), activation='relu', padding='valid'))(inputs)
        # c1 = TimeDistributed(MaxPooling2D())(c1)
        c2 = TimeDistributed(Conv2D(64, (5,5), strides=(2,2), activation='relu', padding='valid'))(c1)
        # c1 = TimeDistributed(MaxPooling2D())(c1)
        c3 = TimeDistributed(Conv2D(128, (3,3), strides=(2,2), activation='relu', padding='valid'))(c2)
        c4 = TimeDistributed(Conv2D(256, (1,1), strides=(2,2), activation='relu', padding='valid'))(c3)
        c5 = TimeDistributed(Conv2D(512, (1,1), strides=(2,2), activation='relu', padding='valid'))(c4)
        # c6 = TimeDistributed(Conv2D(1024, (1,1), strides=(2,2), activation='relu', padding='valid'))(c5)
        # c1 = TimeDistributed(Conv2D(128, (3,3), strides=(2,2), activation='relu', padding='valid'))(c1)
        # c1 = TimeDistributed(MaxPooling2D())(c1)
        # norm = BatchNormalization(axis=-1)(c1)
        # c1 = TimeDistributed(Conv2D(32, (3,3), strides=(2,2), activation='relu', padding='valid'))(c1)
        # c1 = TimeDistributed(Conv2D(32, (3,3), strides=(2,2), activation='relu', padding='valid'))(c1)
        # pool = TimeDistributed(MaxPooling2D())(c1)
        # c2 = TimeDistributed(Conv2D(64, (3,3), activation='relu', padding='valid'), input_shape=self.input_shape)(c1)
        
        # flat = TimeDistributed(Flatten())(c5)
        flat = TimeDistributed(GlobalMaxPooling2D())(c5)
        
        # attention_mul = self.attention_3d_block(flat)

        # dense = Dense(64, activation='relu')(flat)
        drop = Dropout(0.25)(flat)
        # attention_mul = self.attention_3d_block(drop)
        # lstm = LSTM(10, return_sequences=True,dropout=0.25)(drop)
        # attention_mul = self.attention_3d_block(lstm)
        # dense1 = TimeDistributed(Dense(self.nclasses, activation='softmax'))(drop)
        lstm = LSTM(100, return_sequences=True)(drop)
        outputs = TimeDistributed(Dense(self.nclasses, activation='softmax'))(lstm)

        model = Model(inputs=inputs, outputs=outputs)
        return model


       

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

        # model.add(TimeDistributed(Conv2D(64, (3,3),
        #     padding='same', activation='relu')))
        # model.add(TimeDistributed(Conv2D(64, (3,3),
        #     padding='same', activation='relu')))
        # model.add(TimeDistributed(MaxPooling2D((2, 2), strides=(2, 2))))

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

        # model.add(TimeDistributed(Conv2D(1024, (3,3),
        #     padding='same', activation='relu')))
        # model.add(TimeDistributed(Conv2D(1024, (3,3),
        #     padding='same', activation='relu')))
        # model.add(TimeDistributed(MaxPooling2D((2, 2), strides=(2, 2))))

        model.add(TimeDistributed(Flatten()))
        model.add(Dense(128, activation='relu', name='fc1'))
        model.add(Dropout(0.5))
        
        model.add(LSTM(32, return_sequences=True, dropout=0.5))
        model.add(TimeDistributed(Dense(self.nclasses, activation='softmax')))

        return model

    def conv_lstm(self):
        inputs = Input(shape=self.input_shape)
        # attention_mul = self.attention_3d_block(inputs)
        conv = ConvLSTM2D(32, (3,3), return_sequences=True)(inputs)
        conv2 = ConvLSTM2D(32, (1,1), return_sequences=True)(conv)
        flat = TimeDistributed(Flatten())(conv2)
        outputs = TimeDistributed(Dense(self.nclasses, activation='softmax'))(flat)
        model = Model(inputs = inputs, outputs = outputs)
        return model

    def conv_lstm_att1(self):
        inputs = Input(shape=self.input_shape)
        attention_mul = self.attention_3d_block(inputs)
        conv = ConvLSTM2D(32, (3,3), return_sequences=True)(attention_mul)
        conv2 = ConvLSTM2D(32, (1,1), return_sequences=True)(conv)
        flat = TimeDistributed(Flatten())(conv2)
        outputs = TimeDistributed(Dense(self.nclasses, activation='softmax'))(flat)
        model = Model(inputs = inputs, outputs = outputs)
        return model

    def conv_lstm_att2(self):
        inputs = Input(shape=self.input_shape)
        
        conv = ConvLSTM2D(32, (3,3), return_sequences=True)(inputs)
        conv2 = ConvLSTM2D(32, (1,1), return_sequences=True)(conv)
        attention_mul = self.attention_3d_block(conv2)
        flat = TimeDistributed(Flatten())(attention_mul)
        outputs = TimeDistributed(Dense(self.nclasses, activation='softmax'))(flat)
        model = Model(inputs = inputs, outputs = outputs)
        return model

    def conv_lstm1(self):
        """ Conv Layers in front of ConvLSTM
        """
        inputs = Input(shape=self.input_shape)
        pconv = TimeDistributed(Conv2D(32, (7,7), strides=(2,2), activation='relu'))(inputs)
        pconv = TimeDistributed(Conv2D(64, (3,3), strides=(2,2), activation='relu'))(pconv)
        conv = ConvLSTM2D(32, (3,3), return_sequences=True)(pconv)
        conv2 = ConvLSTM2D(32, (1,1), return_sequences=True)(conv)
        flat = TimeDistributed(Flatten())(conv2)
        outputs = TimeDistributed(Dense(self.nclasses, activation='softmax'))(flat)
        model = Model(inputs = inputs, outputs = outputs)
        return model
    
    def conv_lstm2(self):
        """ Conv Layers after ConvLSTM
        """
        inputs = Input(shape=self.input_shape)
        conv = ConvLSTM2D(32, (3,3), return_sequences=True)(inputs)
        conv2 = ConvLSTM2D(32, (1,1), return_sequences=True)(conv)
        pconv = TimeDistributed(Conv2D(32, (7,7), strides=(2,2), activation='relu'))(conv2)
        pconv = TimeDistributed(Conv2D(64, (3,3), strides=(2,2), activation='relu'))(pconv)

        flat = TimeDistributed(Flatten())(pconv)
        outputs = TimeDistributed(Dense(self.nclasses, activation='softmax'))(flat)
        model = Model(inputs = inputs, outputs = outputs)
        return model

    def conv_lstm3(self):
        """ Extra Dense layer at the end. 
        """
        inputs = Input(shape=self.input_shape)
        conv = ConvLSTM2D(32, (3,3), return_sequences=True)(inputs)
        conv2 = ConvLSTM2D(32, (1,1), return_sequences=True)(conv)
        flat = TimeDistributed(Flatten())(conv2)
        fc1 = TimeDistributed(Dense(512, activation='relu'))(flat)
        outputs = TimeDistributed(Dense(self.nclasses, activation='softmax'))(fc1)
        model = Model(inputs = inputs, outputs = outputs)
        return model

    def conv_lstm4(self):
        """ 3 conv-lstms
        """
        inputs = Input(shape=self.input_shape)
        conv = ConvLSTM2D(32, (3,3), return_sequences=True)(inputs)
        conv2 = ConvLSTM2D(32, (1,1), return_sequences=True)(conv)
        conv3 = ConvLSTM2D(32, (1,1), return_sequences=True)(conv2)
        flat = TimeDistributed(Flatten())(conv3)
        outputs = TimeDistributed(Dense(self.nclasses, activation='softmax'))(flat)
        model = Model(inputs = inputs, outputs = outputs)
        return model

    def conv_lstm5(self):
        """ ConvLSTM with average pooling
        """
        inputs = Input(shape=self.input_shape)
        conv = ConvLSTM2D(32, (3,3), return_sequences=True)(inputs)
        conv2 = ConvLSTM2D(32, (1,1), return_sequences=True)(conv)
        avgpool = TimeDistributed(AveragePooling2D((7, 7), name='avg_pool'))(conv2)
        flat = TimeDistributed(Flatten())(avgpool)
        
        outputs = TimeDistributed(Dense(self.nclasses, activation='softmax'))(flat)
        model = Model(inputs = inputs, outputs = outputs)
        return model
    
    def conv_lstm6(self):
        """ ConvLSTM with Dropout before dense
        """
        inputs = Input(shape=self.input_shape)
        conv = ConvLSTM2D(32, (3,3), return_sequences=True)(inputs)
        conv2 = ConvLSTM2D(32, (1,1), return_sequences=True)(conv)
        flat = TimeDistributed(Flatten())(conv2)
        drop = Dropout(0.5)(flat)
        outputs = TimeDistributed(Dense(self.nclasses, activation='softmax'))(drop)
        model = Model(inputs = inputs, outputs = outputs)
        return model
    
    def conv_lstm7(self):
        """ ConvLSTM with Dropout in Convs
        """
        inputs = Input(shape=self.input_shape)
        conv = ConvLSTM2D(32, (3,3), return_sequences=True, dropout=0.5)(inputs)
        conv2 = ConvLSTM2D(32, (1,1), return_sequences=True, dropout=0.5)(conv)
        flat = TimeDistributed(Flatten())(conv2)
        outputs = TimeDistributed(Dense(self.nclasses, activation='softmax'))(flat)
        model = Model(inputs = inputs, outputs = outputs)
        return model

    def c3d(self):

        inputs = Input(shape=(20, config.IMG_WIDTH, config.IMG_HEIGHT, config.IMG_CHANNELS))
        pconv = TimeDistributed(Conv2D(32, (3,3), activation))
        conv = Conv3D(5, (1,1,1))(inputs)
        conv2 = Conv3D(5, (1,1,1))(conv)
        flat = Flatten()(conv2)
        dense = Dense(64, activation='relu')(flat)
        outputs = Dense(self.nclasses, activation='softmax')(dense)
        model = Model(inputs=inputs, outputs=outputs)
        return model

    
    def lstm(self):
        #batch_shape = (# of elements in batch (None means arbitrary), sequence length, size of features (i.e. size of dense layer that feeds into this.))
        input_features = Input(batch_shape=(None,20,128,))
        input_norm = BatchNormalization()(input_features)
        input_drop = Dropout(0.5)(input_norm)
        lstm = LSTM(32, return_sequences=True, stateful=False, name='lstm')(input_drop)
        output = TimeDistributed(Dense(self.nclasses, activation='softmax'), name='fc')(lstm)
        model = Model(inputs=input_features, outputs=output)
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

        x = conv_block(x, 3, [256, 256, 1024], stage=4, block='a')
        x = identity_block(x, 3, [256, 256, 1024], stage=4, block='b')
        x = identity_block(x, 3, [256, 256, 1024], stage=4, block='c')
        # x = identity_block(x, 3, [256, 256, 1024], stage=4, block='d')
        # x = identity_block(x, 3, [256, 256, 1024], stage=4, block='e')
        # x = identity_block(x, 3, [256, 256, 1024], stage=4, block='f')

        # x = conv_block(x, 3, [512, 512, 2048], stage=5, block='a')
        # x = identity_block(x, 3, [512, 512, 2048], stage=5, block='b')
        # x = identity_block(x, 3, [512, 512, 2048], stage=5, block='c')

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
