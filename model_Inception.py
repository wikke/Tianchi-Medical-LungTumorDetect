from keras.models import Model
from keras.layers import Input, Conv3D, MaxPooling3D, Dense, GlobalMaxPooling3D, Dropout, BatchNormalization, \
    Concatenate, AveragePooling3D, Activation
from keras.optimizers import Adam
from config import *

def conv_bn_relu(x, filters, kernel_size=(3, 3, 3), strides=(1, 1, 1), padding='same'):
    x = Conv3D(filters, kernel_size=kernel_size, strides=strides, padding=padding)(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    return x

def inception_base(x):
    x = conv_bn_relu(x, filters=32)
    x = conv_bn_relu(x, filters=32)
    x = conv_bn_relu(x, filters=64)

    b0 = MaxPooling3D(pool_size=(2, 2, 2))(x)
    b1 = conv_bn_relu(x, 64, strides=(2, 2, 2))
    x = Concatenate(axis=4)([b0, b1])

    print('inception_base')
    print(b0.get_shape())
    print(b1.get_shape())
    print(x.get_shape())

    return x

def inception_block(x, filters=256):
    shrinkaged_filters = int(filters * INCEPTION_ENABLE_DEPTHWISE_SEPARABLE_CONV_SHRINKAGE)
    b0 = conv_bn_relu(x, filters=filters, kernel_size=(1, 1, 1))

    b1 = conv_bn_relu(x, filters=shrinkaged_filters, kernel_size=(1, 1, 1))
    b1 = conv_bn_relu(b1, filters=filters, kernel_size=(3, 3, 3))

    b2 = conv_bn_relu(x, filters=shrinkaged_filters, kernel_size=(1, 1, 1))
    b2 = conv_bn_relu(b2, filters=filters, kernel_size=(3, 3, 3))
    b2 = conv_bn_relu(b2, filters=filters, kernel_size=(3, 3, 3))

    b3 = AveragePooling3D(pool_size=(3, 3, 3), strides=(1, 1, 1), padding='same')(x)
    b3 = conv_bn_relu(b3, filters=filters, kernel_size=(1, 1, 1))

    bs = [b0, b1, b2, b3]

    print('inception_block')
    print(b0.get_shape())
    print(b1.get_shape())
    print(b2.get_shape())
    print(b3.get_shape())

    if INCEPTION_ENABLE_SPATIAL_SEPARABLE_CONV:
        b4 = conv_bn_relu(x, filters=shrinkaged_filters, kernel_size=(1, 1, 1))
        b4 = conv_bn_relu(b4, filters=filters, kernel_size=(5, 1, 1))
        b4 = conv_bn_relu(b4, filters=filters, kernel_size=(1, 5, 1))
        b4 = conv_bn_relu(b4, filters=filters, kernel_size=(1, 1, 5))
        bs.append(b4)
        print(b4.get_shape())

    x = Concatenate(axis=4)(bs)
    print(x.get_shape())

    return x

def reduction_block(x, filters=256):
    b0 = conv_bn_relu(x, filters=filters, kernel_size=(3, 3, 3), strides=(2, 2, 2), padding='same')

    b1 = conv_bn_relu(x, filters=filters, kernel_size=(1, 1, 1))
    b1 = conv_bn_relu(b1, filters=filters, kernel_size=(3, 3, 3))
    b1 = conv_bn_relu(b1, filters=filters, kernel_size=(3, 3, 3), strides=(2, 2, 2), padding='same')

    b2 = MaxPooling3D(pool_size=(3, 3, 3), strides=(2, 2, 2), padding='same')(x)
    b2 = conv_bn_relu(b2, filters=filters, kernel_size=(1, 1, 1))

    bs = [b0, b1, b2]

    print('reduction_block')
    print(b0.get_shape())
    print(b1.get_shape())
    print(b2.get_shape())

    if INCEPTION_ENABLE_SPATIAL_SEPARABLE_CONV:
        b3 = conv_bn_relu(x, filters=filters, kernel_size=(1, 1, 1))
        b3 = conv_bn_relu(b3, filters=filters, kernel_size=(5, 1, 1))
        b3 = conv_bn_relu(b3, filters=filters, kernel_size=(1, 5, 1))
        b3 = conv_bn_relu(b3, filters=filters, kernel_size=(1, 1, 5))
        b3 = conv_bn_relu(b3, filters=filters, kernel_size=(3, 3, 3), strides=(2, 2, 2), padding='same')
        bs.append(b3)
        print(b3.get_shape())

    x = Concatenate(axis=4)(bs)
    print(x.get_shape())

    return x

def get_Inception_classifier():
    inputs = Input((CLASSIFY_INPUT_WIDTH, CLASSIFY_INPUT_HEIGHT, CLASSIFY_INPUT_DEPTH, CLASSIFY_INPUT_CHANNEL))
    print('inputs')
    print(inputs.get_shape())

    # Make inception base
    x = inception_base(inputs)

    for i in range(INCEPTION_BLOCKS):
        x = inception_block(x, filters=INCEPTION_KEEP_FILTERS)

        if (i + 1) % INCEPTION_REDUCTION_STEPS == 0 and i != INCEPTION_BLOCKS - 1:
            x = reduction_block(x, filters=INCEPTION_KEEP_FILTERS // 2)

    print('top')
    x = GlobalMaxPooling3D()(x)
    print(x.get_shape())
    x = Dropout(INCEPTION_DROPOUT)(x)
    x = Dense(2, activation='softmax')(x)
    print(x.get_shape())

    model = Model(inputs=inputs, outputs=x)
    model.compile(optimizer=Adam(lr=TRAIN_CLASSIFY_LEARNING_RATE), loss='binary_crossentropy', metrics=['accuracy'])

    return model
