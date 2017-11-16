from keras.models import Model
from keras.layers import Input, Conv3D, MaxPooling3D, Dense, GlobalMaxPooling3D, Dropout, Flatten, BatchNormalization, \
    Concatenate, AveragePooling3D, GlobalAveragePooling3D, Activation
from keras.optimizers import Adam
from config import *

def get_InceptionV4_classifier():
    inputs = Input((CLASSIFY_INPUT_WIDTH, CLASSIFY_INPUT_HEIGHT, CLASSIFY_INPUT_DEPTH, CLASSIFY_INPUT_CHANNEL))
    x = inputs
    print('inputs')
    print(x.get_shape())

    def conv_bn(x, filters, kernel_size=(3, 3, 3), strides=(1, 1, 1), padding='same'):
        x = Conv3D(filters, kernel_size=kernel_size, strides=strides, padding=padding)(x)
        x = BatchNormalization()(x)
        x = Activation('relu')(x)
        return x

    def inception_v4_base(x):
        x = conv_bn(x, filters=32)
        x = conv_bn(x, filters=32)
        x = conv_bn(x, filters=64)

        b0 = MaxPooling3D(pool_size=(2, 2, 2))(x)
        b1 = conv_bn(x, 64, strides=(2, 2, 2))
        x = Concatenate(axis=4)([b0, b1])

        print('inception_v4_base')
        print(b0.get_shape())
        print(b1.get_shape())
        print(x.get_shape())

        return x

    def inception_block(x, filters=256):
        shrinkaged_filters = int(filters * INCEPTION_V4_ENABLE_DEPTHWISE_SEPARABLE_CONV_SHRINKAGE)
        b0 = conv_bn(x, filters=filters, kernel_size=(1, 1, 1))

        b1 = conv_bn(x, filters=shrinkaged_filters, kernel_size=(1, 1, 1))
        b1 = conv_bn(b1, filters=filters, kernel_size=(3, 3, 3))

        b2 = conv_bn(x, filters=shrinkaged_filters, kernel_size=(1, 1, 1))
        b2 = conv_bn(b2, filters=filters, kernel_size=(3, 3, 3))
        b2 = conv_bn(b2, filters=filters, kernel_size=(3, 3, 3))

        b3 = AveragePooling3D(pool_size=(3, 3, 3), strides=(1, 1, 1), padding='same')(x)
        b3 = conv_bn(b3, filters=filters, kernel_size=(1, 1, 1))

        bs = [b0, b1, b2, b3]

        print('inception_block')
        print(b0.get_shape())
        print(b1.get_shape())
        print(b2.get_shape())
        print(b3.get_shape())

        if INCEPTION_V4_ENABLE_SPATIAL_SEPARABLE_CONV:
            b4 = conv_bn(x, filters=shrinkaged_filters, kernel_size=(1, 1, 1))
            b4 = conv_bn(b4, filters=filters, kernel_size=(5, 1, 1))
            b4 = conv_bn(b4, filters=filters, kernel_size=(1, 5, 1))
            b4 = conv_bn(b4, filters=filters, kernel_size=(1, 1, 5))
            bs.append(b4)
            print(b4.get_shape())

        x = Concatenate(axis=4)(bs)
        print(x.get_shape())

        return x

    # reduce by strides
    def reduction_block(x, filters=256):
        b0 = conv_bn(x, filters=filters, kernel_size=(3, 3, 3), strides=(2, 2, 2), padding='same')

        b1 = conv_bn(x, filters=filters, kernel_size=(1, 1, 1))
        b1 = conv_bn(b1, filters=filters, kernel_size=(3, 3, 3))
        b1 = conv_bn(b1, filters=filters, kernel_size=(3, 3, 3), strides=(2, 2, 2), padding='same')

        b2 = MaxPooling3D(pool_size=(3, 3, 3), strides=(2, 2, 2), padding='same')(x)
        # I added
        b2 = conv_bn(b2, filters=filters, kernel_size=(1, 1, 1))

        bs = [b0, b1, b2]

        print('reduction_block')
        print(b0.get_shape())
        print(b1.get_shape())
        print(b2.get_shape())

        if INCEPTION_V4_ENABLE_SPATIAL_SEPARABLE_CONV:
            b3 = conv_bn(x, filters=filters, kernel_size=(1, 1, 1))
            b3 = conv_bn(b3, filters=filters, kernel_size=(5, 1, 1))
            b3 = conv_bn(b3, filters=filters, kernel_size=(1, 5, 1))
            b3 = conv_bn(b3, filters=filters, kernel_size=(1, 1, 5))
            b3 = conv_bn(b3, filters=filters, kernel_size=(3, 3, 3), strides=(2, 2, 2), padding='same')
            bs.append(b3)
            print(b3.get_shape())

        x = Concatenate(axis=4)(bs)
        print(x.get_shape())

        return x

    # Make inception base
    x = inception_v4_base(x)

    for i in range(INCEPTION_V4_BLOCKS):
        x = inception_block(x, filters=INCEPTION_V4_KEEP_FILTERS)

        if (i + 1) % INCEPTION_V4_REDUCTION_STEPS == 0 and i != INCEPTION_V4_BLOCKS - 1:
            x = reduction_block(x, filters=INCEPTION_V4_KEEP_FILTERS // 2)

    print('top')
    x = GlobalMaxPooling3D()(x)
    print(x.get_shape())
    x = Dropout(INCEPTION_V4_DROPOUT)(x)
    x = Dense(2, activation='softmax')(x)
    print(x.get_shape())

    model = Model(inputs=inputs, outputs=x)
    model.compile(optimizer=Adam(lr=TRAIN_CLASSIFY_LEARNING_RATE), loss='binary_crossentropy', metrics=['accuracy'])

    return model

def get_DenseNet_classifier():
    inputs = Input((CLASSIFY_INPUT_WIDTH, CLASSIFY_INPUT_HEIGHT, CLASSIFY_INPUT_DEPTH, CLASSIFY_INPUT_CHANNEL))
    x = Conv3D(DENSE_NET_INITIAL_CONV_DIM, (3, 3, 3), padding='same')(inputs)

    for i in range(DENSE_NET_BLOCKS):
        y = BatchNormalization()(x)
        y = Activation('relu')(y)
        y = Conv3D(DENSE_NET_DIM_INCR, (3, 3, 3), padding='same')(y)
        x = Concatenate(axis=4)([x, y])

        if (i + 1) % DENSE_NET_SHRINKAGE_STEPS == 0 and i != DENSE_NET_BLOCKS - 1:
            x = BatchNormalization()(x)
            x = Activation('relu')(x)
            x = Conv3D(x.get_shape()[4].value, (3, 3, 3), padding='same')(x)
            x = AveragePooling3D(pool_size=(2, 2, 2), padding='same')(x)

    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    x = GlobalAveragePooling3D()(x)
    x = Dense(2, activation='softmax')(x)

    model = Model(inputs=inputs, outputs=x)
    model.compile(optimizer=Adam(lr=TRAIN_CLASSIFY_LEARNING_RATE), loss='binary_crossentropy', metrics=['accuracy'])

    return model

def get_VGG_classifier():
    return get_simplified_VGG_classifier() if USE_SIMPLIFIED_VGG else get_full_VGG_classifier()

def get_simplified_VGG_classifier():
    inputs = Input((CLASSIFY_INPUT_WIDTH, CLASSIFY_INPUT_HEIGHT, CLASSIFY_INPUT_DEPTH, CLASSIFY_INPUT_CHANNEL))

    x = Conv3D(16, (3, 3, 3), padding='same', activation='relu')(inputs)
    x = Conv3D(16, (3, 3, 3), padding='same', activation='relu')(x)
    x = MaxPooling3D(pool_size=(2, 2, 2))(x)

    if TRAIN_CLASSIFY_USE_BN:
        x = BatchNormalization()(x)

    x = Conv3D(32, (3, 3, 3), padding='same', activation='relu')(x)
    x = Conv3D(32, (3, 3, 3), padding='same', activation='relu')(x)
    x = MaxPooling3D(pool_size=(2, 2, 2))(x)

    if TRAIN_CLASSIFY_USE_BN:
        x = BatchNormalization()(x)

    x = Conv3D(64, (3, 3, 3), padding='same', activation='relu')(x)
    x = Conv3D(64, (3, 3, 3), padding='same', activation='relu')(x)
    x = Conv3D(64, (3, 3, 3), padding='same', activation='relu')(x)
    x = MaxPooling3D(pool_size=(2, 2, 2))(x)

    if TRAIN_CLASSIFY_USE_BN:
        x = BatchNormalization()(x)

    x = Conv3D(128, (3, 3, 3), padding='same', activation='relu')(x)
    x = Conv3D(128, (3, 3, 3), padding='same', activation='relu')(x)
    x = Conv3D(128, (3, 3, 3), padding='same', activation='relu')(x)
    # x = MaxPooling3D(pool_size=(2, 2, 2))(x)
    #
    # x = Conv3D(512, (3, 3, 3), padding='same', activation='relu')(x)
    # x = Conv3D(512, (3, 3, 3), padding='same', activation='relu')(x)
    # x = Conv3D(512, (3, 3, 3), padding='same', activation='relu')(x)
    x = GlobalMaxPooling3D()(x)
    # x = Flatten()(x)

    x = Dense(32, activation='relu')(x)
    x = Dropout(0.5)(x)
    x = Dense(2, activation='softmax')(x)

    model = Model(inputs=inputs, outputs=x)
    model.compile(optimizer=Adam(lr=TRAIN_CLASSIFY_LEARNING_RATE), loss='binary_crossentropy', metrics=['accuracy'])

    return model

def get_full_VGG_classifier():
    inputs = Input((CLASSIFY_INPUT_WIDTH, CLASSIFY_INPUT_HEIGHT, CLASSIFY_INPUT_DEPTH, CLASSIFY_INPUT_CHANNEL))
    x = inputs


    x = Conv3D(32, (3, 3, 3), padding='same', activation='relu')(x)
    x = Conv3D(32, (3, 3, 3), padding='same', activation='relu')(x)
    x = MaxPooling3D(pool_size=(2, 2, 2))(x)

    if TRAIN_CLASSIFY_USE_BN:
        x = BatchNormalization()(x)

    x = Conv3D(64, (3, 3, 3), padding='same', activation='relu')(x)
    x = Conv3D(64, (3, 3, 3), padding='same', activation='relu')(x)
    x = MaxPooling3D(pool_size=(2, 2, 2))(x)

    if TRAIN_CLASSIFY_USE_BN:
        x = BatchNormalization()(x)

    x = Conv3D(128, (3, 3, 3), padding='same', activation='relu')(x)
    x = Conv3D(128, (3, 3, 3), padding='same', activation='relu')(x)
    x = Conv3D(128, (3, 3, 3), padding='same', activation='relu')(x)
    x = MaxPooling3D(pool_size=(2, 2, 2))(x)

    if TRAIN_CLASSIFY_USE_BN:
        x = BatchNormalization()(x)

    x = Conv3D(256, (3, 3, 3), padding='same', activation='relu')(x)
    x = Conv3D(256, (3, 3, 3), padding='same', activation='relu')(x)
    x = Conv3D(256, (3, 3, 3), padding='same', activation='relu')(x)
    x = MaxPooling3D(pool_size=(2, 2, 2))(x)

    if TRAIN_CLASSIFY_USE_BN:
        x = BatchNormalization()(x)

    x = Conv3D(512, (3, 3, 3), padding='same', activation='relu')(x)
    x = Conv3D(512, (3, 3, 3), padding='same', activation='relu')(x)
    x = Conv3D(512, (3, 3, 3), padding='same', activation='relu')(x)
    x = GlobalMaxPooling3D()(x)

    x = Dense(32, activation='relu')(x)
    x = Dropout(0.5)(x)
    x = Dense(2, activation='softmax')(x)

    model = Model(inputs=inputs, outputs=x)
    model.compile(optimizer=Adam(lr=TRAIN_CLASSIFY_LEARNING_RATE), loss='binary_crossentropy', metrics=['accuracy'])

    return model

