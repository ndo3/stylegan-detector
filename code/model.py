import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers


def make_block_name(num, subnum, is_separable):
    if is_separable:
        mid_val = 'sepconv'
    else:
        mid_val = 'conv'
    return f'block{num}_{mid_val}{subnum}'


def create_act(num, subnum, is_separable):
    name = f'{make_block_name(num, subnum, is_separable)}_act'
    return layers.Activation('relu', name=name)


def create_conv_or_sepconv(is_separable, is_residual, num, subnum, filters, kernel_size, strides, act):
    name = f'residual{num}' if is_residual else make_block_name(
        num, subnum, is_separable)
    conv_layer = layers.SeparableConv2D if is_separable else layers.Conv2D
    channel_axis = 1 if keras.backend.image_data_format() == 'channels_first' else -1
    ret = [
        conv_layer(filters, kernel_size, strides=strides,
                   padding='same', name=name),
        layers.BatchNormalization(axis=channel_axis, name=f'{name}_bn')
    ]
    if act == 'before':
        return [create_act(num, subnum, is_separable), *ret]
    elif act == 'after':
        return [*ret, create_act(num, subnum, is_separable)]
    else:
        return ret


def create_conv(num, subnum, filters, kernel_size, strides=(1, 1), act='after'):
    return create_conv_or_sepconv(False, False, num, subnum, filters, kernel_size, strides, act)


def create_residual(num, filters):
    return create_conv_or_sepconv(False, True, num, None, filters, (1, 1), (2, 2), None)


def create_sepconv(num, subnum, filters, act='before'):
    return create_conv_or_sepconv(True, False, num, subnum, filters, (3, 3), (1, 1), act)


def create_maxpool(num):
    return layers.MaxPooling2D(pool_size=(3, 3), strides=(2, 2), padding='same', name=f'block{num}_pool')


def create_entry_block(num, filters, start_with_act=True):
    return [
        *create_sepconv(num, 0, filters,
                        act='before' if start_with_act else 'after'),
        *create_sepconv(num, 1, filters),
        create_maxpool(num)
    ]


def create_middle_block(num):
    return [
        *create_sepconv(num, 0, 728),
        *create_sepconv(num, 1, 728),
        *create_sepconv(num, 2, 728)
    ]


def create_model(truncate_block_num, img_size, batch_size):
    """Creates and returns a possibly truncated custom version of the Xception
    model using the Keras Functional API. We could not use the Sequential API 
    because the residuals of the model mean it is non-sequential. Also, we 
    created all the layers and then called them, which is untraditional but 
    helps us control the truncation location that can be customized at any 
    block. The model is based on:
     - https://arxiv.org/pdf/1610.02357.pdf
     - https://github.com/keras-team/keras-applications/blob/master/keras_applications/xception.py
     - https://github.com/chail/patch-forensics/blob/3697d6658477f9ed5eef4d1ab826a6d15b3daab3/models/networks/xception.py
     - https://github.com/chail/patch-forensics/blob/3697d6658477f9ed5eef4d1ab826a6d15b3daab3/models/networks/customnet.py#L26

    Args:
        truncate_block_num (int): None if untruncated, else the block number to 
        truncate after.
        img_size (int): The number of pixels in height and width of each image.
        batch_size (int): The number of images per batch for the model.

    Returns:
        keras.model.Model: The model, ready to be compiled and trained on.
    """

    #####################
    ###    STEP 1:    ###
    ### Create layers ###
    #####################

    is_trunc = truncate_block_num != None
    inputs = layers.Input(shape=(img_size, img_size, 3), batch_size=batch_size)

    blocks = {}
    residuals = {}

    blocks[0] = [
        *create_conv(0, 0, 32, (3, 3), (2, 2)),
        *create_conv(0, 1, 64, (3, 3))
    ]

    for i, filters in zip([1, 2, 3], [128, 256, 728]):
        residuals[i] = create_residual(i, filters)
        blocks[i] = create_entry_block(i, filters, start_with_act=i != 1)

    for i in range(4, 12):
        blocks[i] = create_middle_block(i)

    residuals[12] = create_residual(12, 1024)
    blocks[12] = [
        *create_sepconv(12, 0, 728),
        *create_sepconv(12, 1, 1024),
        create_maxpool(12)
    ]

    blocks[13] = [
        *create_sepconv(13, 0, 1536, act='after'),
        *create_sepconv(13, 1, 2048, act='after'),
        layers.GlobalAveragePooling2D(name='block13_globalpool'),
        layers.Dense(2, activation='softmax', name='block13_dense')
    ]

    blocks['trunc'] = [
        layers.Flatten(name='truncated_end_flatten'),
        layers.Dense(2, activation='sigmoid', name='truncated_end_dense')
    ]

    #######################
    ###     STEP 2:     ###
    ### Assemble layers ###
    #######################

    x = inputs
    for i in range(truncate_block_num + 1 if is_trunc else 14):
        if i in [0, 13]:
            for l in blocks[i]:
                x = l(x)
        elif i in [1, 2, 3, 12]:
            r = tf.identity(x)
            for l in residuals[i]:
                r = l(r)
            for l in blocks[i]:
                x = l(x)
            x = layers.add([x, r])
        else:  # i is in range(4, 12)
            for l in blocks[i]:
                x = l(x)
    if is_trunc:
        for l in blocks['trunc']:
            x = l(x)
    predictions = x

    model = keras.models.Model(inputs=inputs, outputs=predictions)

    return model
