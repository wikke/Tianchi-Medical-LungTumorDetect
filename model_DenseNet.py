from keras.models import Model
from keras.layers import Input, Conv3D, Dense, BatchNormalization, Concatenate, Dropout, AveragePooling3D, GlobalAveragePooling3D, Activation
from keras.optimizers import Adam
from config import *

def bn_relu_conv(x, filters, kernel_size=(3, 3, 3)):
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    x = Conv3D(filters=filters, kernel_size=kernel_size, padding='same')(x)

    return x

def dense_block(x):
    print('dense block')
    print(x.get_shape())

    for _ in range(DENSE_NET_BLOCK_LAYERS):
        y = x

        if DENSE_NET_ENABLE_BOTTLENETCK:
            y = bn_relu_conv(y, filters=DENSE_NET_GROWTH_RATE, kernel_size=(1, 1, 1))

        y = bn_relu_conv(y, filters=DENSE_NET_GROWTH_RATE, kernel_size=(3, 3, 3))
        x = Concatenate(axis=4)([x, y])
        print(x.get_shape())

    return x

def transition_block(x):
    print('transition block')
    print(x.get_shape())

    filters = x.get_shape()[4].value
    filters = int(filters * DENSE_NET_TRANSITION_COMPRESSION)

    x = Conv3D(filters=filters, kernel_size=(1, 1, 1), padding='same')(x)
    x = AveragePooling3D(pool_size=(2, 2, 2), padding='same')(x)
    print(x.get_shape())

    return x

def get_DenseNet_classifier():
    inputs = Input((CLASSIFY_INPUT_WIDTH, CLASSIFY_INPUT_HEIGHT, CLASSIFY_INPUT_DEPTH, CLASSIFY_INPUT_CHANNEL))
    x = Conv3D(DENSE_NET_INITIAL_CONV_DIM, (3, 3, 3), padding='same')(inputs)
    print('input')
    print(x.get_shape())

    for i in range(DENSE_NET_BLOCKS):
        x = dense_block(x)
        if i != DENSE_NET_BLOCKS - 1:
            x = transition_block(x)

    print('top')
    x = GlobalAveragePooling3D()(x)
    print(x.get_shape())

    if DENSE_NET_ENABLE_DROPOUT:
        x = Dropout(DENSE_NET_DROPOUT)(x)

    x = Dense(2, activation='softmax')(x)
    print(x.get_shape())

    model = Model(inputs=inputs, outputs=x)
    model.compile(optimizer=Adam(lr=TRAIN_CLASSIFY_LEARNING_RATE), loss='binary_crossentropy', metrics=['accuracy'])

    return model
