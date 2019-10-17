import os
import random
import datetime
import re
import math
import logging


import numpy as np
import tensorflow as tf

import keras
import keras.backend as K
import keras.layers as KL
import keras.engine as KE
import keras.models as KM

# import config as sconfig
# from keras.utils.vis_utils import  plot_model
import srcnn_utils as Mutils

from tensorflow.compat.v1 import ConfigProto
from tensorflow.compat.v1 import InteractiveSession


config = ConfigProto()
config.gpu_options.allow_growth = True
session = InteractiveSession(config=config)

##########################################################################
#  some small functions
##########################################################################
def log(text, array=None):
    """Prints a text message. And, optionally, if a Numpy array is provided it
    prints it's shape, min, and max values.
    """
    if array is not None:
        text = text.ljust(25)
        text += ("shape: {:20}  ".format(str(array.shape)))
        if array.size:
            text += ("min: {:10.5f}  max: {:10.5f}".format(array.min(),array.max()))
        else:
            text += ("min: {:10}  max: {:10}".format("",""))
        text += "  {}".format(array.dtype)
    print(text)

def index_intersection(a, b):
    # a: [N] b:[N]
    # get the intersetion of a , b   the rank of a and b is 1
    a= tf.unique(a)[0]
    b = tf.unique(b)[0]
    find_match = tf.abs(tf.expand_dims(a, 1) - tf.expand_dims(b, 0))  # find_match [length_a, length_b]
    index_a = tf.transpose(tf.where(tf.equal(find_match, tf.zeros_like(find_match))))[0]
    index_a = tf.unique(index_a)[0]
    return tf.unique(tf.gather(a, index_a))[0]

def index_union(a,b):
    # get the union of a , b   the rank of a and b is 1
    a = tf.unique(a)[0]
    b = tf.unique(b)[0]
    find_match = tf.abs(tf.expand_dims(a, 1) - tf.expand_dims(b, 0))  # find_match [length_a, length_b]
    index_a_insert = tf.where(tf.equal(find_match, tf.zeros_like(find_match)))[:,0]
    index_a_no_insert =  tf.where(tf.not_equal(find_match, tf.zeros_like(find_match)))[:,0]
    index_b_no_insert  = tf.where(tf.not_equal(find_match, tf.zeros_like(find_match)))[:,1]

    insert = tf.gather(a,index_a_insert)
    no_insert_a = tf.gather(a,index_a_no_insert)
    no_insert_b = tf.gather(b, index_b_no_insert)
    return tf.unique(tf.concat([ insert,no_insert_a,no_insert_b],axis=0))[0]


def batch_pack_graph(x, counts, num_rows):
    """Picks different number of values from each row
    in x depending on the values in counts.
    """
    outputs = []
    for i in range(num_rows):
        outputs.append(x[i, :counts[i]])
    return tf.concat(outputs, axis=0)


def compose_image_meta(image_id, original_image_shape_group, image_shape_group,
                       window_group, scale_group, active_class_ids):
    """Takes attributes of an image and puts them in one 1D array.

    image_id: An int ID of the image. Useful for debugging.
    original_image_shape_group: [left_shape,right_shape] where left_shape si  [H, W, C] before resizing or padding.
    image_shape_group: [left_shape,right_shape]  where left_shape [H, W, C]  after resizing and padding
    window_group: [left_window,right_window ] where left_window is[y1, x1, y2, x2 ]in pixels. The area of the image where the real
            image is (excluding the padding)
    scale_group: [scale_left,scale_right] The scaling factor applied to the original image (float32)
    active_class_ids: List of class_ids available in the dataset from which
        the image came. Useful if training on images from multiple datasets
        where not all classes are present in all datasets.
    """
    meta = np.array(
        [image_id] +                  # size=1
        list(original_image_shape_group[0]) +  # size=3 left
        list(original_image_shape_group[1]) +  # size=3 right
        list(image_shape_group[0]) +           # size=3   left
        list( image_shape_group[1]) +          # size=3 right
        list( window_group[0])  +                # size=4     left
        list(window_group[1] ) +                # size=4     right                 [left_window,right_window] where left_window :(y1, x1, y2, x2) in image cooredinates
        [scale_group[0]] +                     # size=1
        [scale_group[1]] +                     # size =1
        list(active_class_ids)        # size=num_classes
    )
    return meta

def parse_image_meta_graph(meta):
    """Parses a tensor that contains image attributes to its components.
    See compose_image_meta() for more details.

    meta: [meta length] where meta length depends on NUM_CLASSES   I remove batch_size

    Returns a dict of the parsed tensors.
    """
    image_id = meta[:,0]                       #1
    original_image_shape_left = meta[:,1:4]   #3
    original_image_shape_right = meta[:,4:7]  #3

    image_shape_left = meta[:,7:10]           #3
    image_shape_right  = meta[:,10:13]         # 3

    # (y1, x1, y2, x2) window of image in in pixels
    window_left = meta[:,13:17]                   #4
    window_right = meta[:,17:21]                   #4

    scale_left = meta[:,21]                       #1
    scale_right = meta[:,22]                     #1
    active_class_ids = meta[:, 23:]

    return {
        "image_id": image_id,
        "original_image_shape_left": original_image_shape_left,
        "original_image_shape_right": original_image_shape_right,

        "image_shape_left": image_shape_left,
        "image_shape_right": image_shape_right,

        "window_left": window_left,
        "window_left": window_right,


        "scale_left": scale_left,
        "scale_right": scale_right,
        "active_class_ids": active_class_ids,
    }
def mold_image(images, config):
    """Expects an RGB image (or array of images) and subtracts
    the mean pixel and converts it to float. Expects image
    colors in RGB order.
    """
    return images.astype(np.float32) - config.MEAN_PIXEL


def unmold_image(normalized_images, config):
    """Takes a image normalized with mold() and returns the original."""
    return (normalized_images + config.MEAN_PIXEL).astype(np.uint8)
##############################################################
# FPN Feature map graph
###############################################################
def build_fpn_feature_map_model (channels,config,name):
    '''
    :param channels the channels of input stages [C2,C3,C4,C5...]  [channels1, channels2, channels3 , channels4,...]
    :param config: config of the model
    :return: feature pyramid networks  [P2,P3,P4,P5,P6]  P6 is used only for fpn
    '''

    C2 = KL.Input([None,None,channels[0]],name='fpn_c2_input')
    C3 = KL.Input([None, None, channels[1]], name='fpn_c3_input')
    C4 = KL.Input([None,None,channels[2]],name='fpn_c4_input')
    C5 = KL.Input([None,None,channels[3]],name='fpn_c5_input')

    def paddingC(C,name):
        '''
        padding to ensure that the C H and W is both 偶数， 因为 可能出现 上采样之后不匹配报错的情况
        :param C:
        :return:
        '''
        shape = C.get_shape().as_list()
        h = shape[1]
        w = shape[2]
        padding_top = 0
        padding_left = 0
        if h :
            if h %2 != 0:
                padding_top =1
        if w:
            if w%2 !=0:
                padding_left =1
        return KL.ZeroPadding2D(((padding_top,0),(padding_left,0)),name = name)(C)
    # print(C4.get_shape().as_list(),'\n')
    # print( KL.UpSampling2D(size=(2, 2))(C4).get_shape().as_list(), '\n')
    # their is some problem with the KL.UpSampling2D
    # P5 = KL.Conv2D(config.TOP_DOWN_PYRAMID_SIZE, (1, 1), name='fpn_c5p5_shared')(C5)
    # P4 =KL.Add(name="fpn_p4add")([
    #     KL.Lambda(lambda t: tf.image.resize_images(t,tf.shape(C4)[1:3]), name="fpn_p5upsampled_shared")(P5),
    #     KL.Conv2D(config.TOP_DOWN_PYRAMID_SIZE, (1, 1), name='fpn_c4p4_shared')(C4)
    # ]
    # )
    # P3 = KL.Add(name="fpn_p3add")([
    #     KL.Lambda(lambda t: tf.image.resize_images(t, tf.shape(C3)[1:3]), name="fpn_p4upsampled_shared")(P4),
    #     KL.Conv2D(config.TOP_DOWN_PYRAMID_SIZE, (1, 1), name='fpn_c3p3_shared')(C3)
    # ]
    # )
    # P2 = KL.Add(name="fpn_p2add")([
    #     KL.Lambda(lambda t: tf.image.resize_images(t, tf.shape(C2)[1:3]), name="fpn_p3upsampled_shared")(P3),
    #     KL.Conv2D(config.TOP_DOWN_PYRAMID_SIZE, (1, 1), name='fpn_c2p2_shared')(C2)
    # ]
    # )
    P5 = KL.Conv2D(config.TOP_DOWN_PYRAMID_SIZE, (1, 1), name='fpn_c5p5_shared')(C5)
    # # PADDING


    P4 = KL.Add(name="fpn_p4add")([
        KL.UpSampling2D(size=(2, 2), name="fpn_p5upsampled_shared")(P5),
        KL.Conv2D(config.TOP_DOWN_PYRAMID_SIZE, (1, 1), name='fpn_c4p4_shared')(C4)])


    P3 = KL.Add(name="fpn_p3add")([
        KL.UpSampling2D(size=(2, 2), name="fpn_p4upsampled_shared")(P4),
        KL.Conv2D(config.TOP_DOWN_PYRAMID_SIZE, (1, 1), name='fpn_c3p3_shared')(C3)])


    P2 = KL.Add(name="fpn_p2add")([
        KL.UpSampling2D(size=(2, 2), name="fpn_p3upsampled_shared")(P3),
        KL.Conv2D(config.TOP_DOWN_PYRAMID_SIZE, (1, 1), name='fpn_c2p2_shared')(C2)])

    # Attach 3x3 conv to all P layers to get the final feature maps.
    P2 = KL.Conv2D(config.TOP_DOWN_PYRAMID_SIZE, (3, 3), padding="SAME", name="fpn_p2_shared")(P2)
    P3 = KL.Conv2D(config.TOP_DOWN_PYRAMID_SIZE, (3, 3), padding="SAME", name="fpn_p3_shared")(P3)
    P4 = KL.Conv2D(config.TOP_DOWN_PYRAMID_SIZE, (3, 3), padding="SAME", name="fpn_p4_shared")(P4)
    P5 = KL.Conv2D(config.TOP_DOWN_PYRAMID_SIZE, (3, 3), padding="SAME", name="fpn_p5_shared")(P5)
    # P6 is used for the 5th anchor scale in RPN. Generated by
    # subsampling from P5 with stride of 2.
    P6 = KL.MaxPooling2D(pool_size=(1, 1), strides=2, name="fpn_p6_shared")(P5)
    output=[P2,P3,P4,P5,P6]
    # Note that P6 is used in RPN, but not in the classifier heads.
    return  KM.Model([C2,C3,C4,C5],output,name='fpn_model_'+name)

############################################################
#  Resnet model
############################################################

# Code adopted from:
# https://github.com/fchollet/deep-learning-models/blob/master/resnet50.py
# https://github.com/chonepieceyb/Mask_RCNN/blob/master/mrcnn/model.py

def identity_block(input_tensor, kernel_size, filters, stage, block,
                   use_bias=True, train_bn=True):
    """The identity_block is the block that has no conv layer at shortcut
    # Arguments
        input_tensor: input tensor
        kernel_size: default 3, the kernel size of middle conv layer at main path
        filters: list of integers, the nb_filters of 3 conv layer at main path
        stage: integer, current stage label, used for generating layer names
        block: 'a','b'..., current block label, used for generating layer names
        use_bias: Boolean. To use or not use a bias in conv layers.
        train_bn: Boolean. Train or freeze Batch Norm layers
    """
    nb_filter1, nb_filter2, nb_filter3 = filters
    conv_name_base = 'res' + str(stage) + block + '_branch'
    bn_name_base = 'bn' + str(stage) + block + '_branch'

    x = KL.Conv2D(nb_filter1, (1, 1), name=conv_name_base + '2a',
                  use_bias=use_bias)(input_tensor)
    x = BatchNorm(name=bn_name_base + '2a')(x, training=train_bn)
    x = KL.Activation('relu')(x)

    x = KL.Conv2D(nb_filter2, (kernel_size, kernel_size), padding='same',
                  name=conv_name_base + '2b', use_bias=use_bias)(x)
    x = BatchNorm(name=bn_name_base + '2b')(x, training=train_bn)
    x = KL.Activation('relu')(x)

    x = KL.Conv2D(nb_filter3, (1, 1), name=conv_name_base + '2c',
                  use_bias=use_bias)(x)
    x = BatchNorm(name=bn_name_base + '2c')(x, training=train_bn)

    x = KL.Add()([x, input_tensor])
    x = KL.Activation('relu', name='res' + str(stage) + block + '_out')(x)
    return x


def conv_block(input_tensor, kernel_size, filters, stage, block,
               strides=(2, 2), use_bias=True, train_bn=True):
    """conv_block is the block that has a conv layer at shortcut
    # Arguments
        input_tensor: input tensor
        kernel_size: default 3, the kernel size of middle conv layer at main path
        filters: list of integers, the nb_filters of 3 conv layer at main path
        stage: integer, current stage label, used for generating layer names
        block: 'a','b'..., current block label, used for generating layer names
        use_bias: Boolean. To use or not use a bias in conv layers.
        train_bn: Boolean. Train or freeze Batch Norm layers
    Note that from stage 3, the first conv layer at main path is with subsample=(2,2)
    And the shortcut should have subsample=(2,2) as well
    """
    nb_filter1, nb_filter2, nb_filter3 = filters
    conv_name_base = 'res' + str(stage) + block + '_branch'
    bn_name_base = 'bn' + str(stage) + block + '_branch'

    x = KL.Conv2D(nb_filter1, (1, 1), strides=strides,
                  name=conv_name_base + '2a', use_bias=use_bias)(input_tensor)
    x = BatchNorm(name=bn_name_base + '2a')(x, training=train_bn)
    x = KL.Activation('relu')(x)

    x = KL.Conv2D(nb_filter2, (kernel_size, kernel_size), padding='same',
                  name=conv_name_base + '2b', use_bias=use_bias)(x)
    x = BatchNorm(name=bn_name_base + '2b')(x, training=train_bn)
    x = KL.Activation('relu')(x)

    x = KL.Conv2D(nb_filter3, (1, 1), name=conv_name_base +
                  '2c', use_bias=use_bias)(x)
    x = BatchNorm(name=bn_name_base + '2c')(x, training=train_bn)

    shortcut = KL.Conv2D(nb_filter3, (1, 1), strides=strides,
                         name=conv_name_base + '1', use_bias=use_bias)(input_tensor)
    shortcut = BatchNorm(name=bn_name_base + '1')(shortcut, training=train_bn)

    x = KL.Add()([x, shortcut])
    x = KL.Activation('relu', name='res' + str(stage) + block + '_out')(x)
    return x


def build_resnet( architecture, name=None,stage5=False, train_bn=True,channel=3):
    """Build a ResNet keras model.
        architecture: Can be resnet50 or resnet101
        stage5: Boolean. If False, stage5 of the network is not created
        train_bn: Boolean. Train or freeze Batch Norm layers
    """
    assert architecture in ["resnet50", "resnet101"]
    # Stage 1
    input_image = KL.Input([None,None,channel])
    x = KL.ZeroPadding2D((3, 3))(input_image)
    x = KL.Conv2D(64, (7, 7), strides=(2, 2), name='conv1', use_bias=True)(x)
    x = BatchNorm(name='bn_conv1')(x, training=train_bn)
    x = KL.Activation('relu')(x)
    C1 = x = KL.MaxPooling2D((3, 3), strides=(2, 2), padding="same")(x)
    # Stage 2
    x = conv_block(x, 3, [64, 64, 256], stage=2, block='a', strides=(1, 1), train_bn=train_bn)
    x = identity_block(x, 3, [64, 64, 256], stage=2, block='b', train_bn=train_bn)
    C2 = x = identity_block(x, 3, [64, 64, 256], stage=2, block='c', train_bn=train_bn)
    # Stage 3
    x = conv_block(x, 3, [128, 128, 512], stage=3, block='a', train_bn=train_bn)
    x = identity_block(x, 3, [128, 128, 512], stage=3, block='b', train_bn=train_bn)
    x = identity_block(x, 3, [128, 128, 512], stage=3, block='c', train_bn=train_bn)
    C3 = x = identity_block(x, 3, [128, 128, 512], stage=3, block='d', train_bn=train_bn)
    # Stage 4
    x = conv_block(x, 3, [256, 256, 1024], stage=4, block='a', train_bn=train_bn)
    block_count = {"resnet50": 5, "resnet101": 22}[architecture]
    for i in range(block_count):
        x = identity_block(x, 3, [256, 256, 1024], stage=4, block=chr(98 + i), train_bn=train_bn)
    C4 = x
    # Stage 5
    if stage5:
        x = conv_block(x, 3, [512, 512, 2048], stage=5, block='a', train_bn=train_bn)
        x = identity_block(x, 3, [512, 512, 2048], stage=5, block='b', train_bn=train_bn)
        C5 = x = identity_block(x, 3, [512, 512, 2048], stage=5, block='c', train_bn=train_bn)
    else:
        C5 = None
    return  KM.Model(input_image,[C1,C2,C3,C4,C5],name=architecture+'_shared')

class BatchNorm(KL.BatchNormalization):
    """Extends the Keras BatchNormalization class to allow a central place
    to make changes if needed.

    Batch normalization has a negative effect on training if batches are small
    so this layer is often frozen (via setting in Config class) and functions
    as linear layer.
    """
    def call(self, inputs, training=None):
        """
        Note about training values:
            None: Train BN layers. This is the normal mode
            False: Freeze BN layers. Good when batch size is small
            True: (don't use). Set layer in training mode even when making inferences
        """
        return super(self.__class__, self).call(inputs, training=training)

def compute_backbone_shapes(config, image_shape):
    """Computes the width and height of each stage of the backbone network.

    Returns:
        [N, (height, width)]. Where N is the number of stages
    """
    if callable(config.BACKBONE):
        return config.COMPUTE_BACKBONE_SHAPE(image_shape)

    # Currently supports ResNet only
    assert config.BACKBONE in ["resnet50", "resnet101"]
    return np.array(
        [[int(math.ceil(image_shape[0] / stride)),
            int(math.ceil(image_shape[1] / stride))]
            for stride in config.BACKBONE_STRIDES])
############################################################
#  S-RPN model
############################################################

def build_srpn_model(anchor_stride, anchors_per_location, channels):

    feature_map = KL.Input([None,None,channels],name ='srpn_input')             # the feature map of the backbone

    # because we concatenate the left and right feature_map, the channels here is 1024 rather than 512
    shared_map =  KL.Conv2D(1024,(3,3), padding='same' , strides=anchor_stride, name='srpn_conv_shared',activation='relu')(feature_map)

    # class_logist
    srpn_class_logist = KL.Conv2D(2*anchors_per_location,(1,1), padding='valid',activation='linear',name='srpn_class_raw')(shared_map)
    # reshape to the shape [batch,anchorsNum , 2]
    srpn_class_logist = KL.Lambda(
        lambda t: tf.reshape(t, [tf.shape(t)[0], -1, 2]))( srpn_class_logist)
    # apply softmax      [batch,anchornum , [BG,FG]]
    srpn_class_prob = KL.Activation(
        "softmax", name="srpn_class_xxx")( srpn_class_logist )

    # box regresser    [ batch, H,W , anchors_per_location *6 ] the 6 is [dx0 log(w0) dx1 log(w1) dy log(dh)]  the 0 refer to the left image  and 1 refer to the right image
    srpn_bbox = KL.Conv2D(6*anchors_per_location,(1,1),padding='valid',name='srpn_bbox_pred',activation='linear')(shared_map)
    # reshape to [batch, anchorsnum , [dx0 log(w0) dx1 log(w1) dy log(dh)]]
    srpn_bbox = KL.Lambda(lambda t: tf.reshape(t, [tf.shape(t)[0], -1, 6]))(srpn_bbox)

    return KM.Model(feature_map,[srpn_class_logist,srpn_class_prob,srpn_bbox],name='srpn')


############################################################
#  S-RPN Layer
############################################################
# adopt the code from : https://github.com/chonepieceyb/Mask_RCNN/blob/master/mrcnn/model.py
def apply_box_deltas_graph(boxes, deltas):

    """Applies the given deltas to the given boxes.
    boxes: [N, (y1, x1, y2, x2)] boxes to update
    deltas: [N, (dy, dx, log(dh), log(dw))] refinements to apply
    """
    # Convert to y, x, h, w
    height = boxes[:, 2] - boxes[:, 0]
    width = boxes[:, 3] - boxes[:, 1]
    center_y = boxes[:, 0] + 0.5 * height
    center_x = boxes[:, 1] + 0.5 * width
    # Apply deltas
    center_y += deltas[:, 0] * height
    center_x += deltas[:, 1] * width
    height *= tf.exp(deltas[:, 2])
    width *= tf.exp(deltas[:, 3])
    # Convert back to y1, x1, y2, x2
    y1 = center_y - 0.5 * height
    x1 = center_x - 0.5 * width
    y2 = y1 + height
    x2 = x1 + width
    result = tf.stack([y1, x1, y2, x2], axis=1, name="apply_box_deltas_out")
    return result

# adopt the code from : https://github.com/chonepieceyb/Mask_RCNN/blob/master/mrcnn/model.py
def clip_boxes_graph(boxes, window):
    """
    boxes: [N, (y1, x1, y2, x2)]
    window: [4] in the form y1, x1, y2, x2
    """
    # Split
    wy1, wx1, wy2, wx2 = tf.split(window, 4)
    y1, x1, y2, x2 = tf.split(boxes, 4, axis=1)   # y1 = [ [y11],[y12],[y13] ......] shape=(N,1)
    # Clip
    y1 = tf.maximum(tf.minimum(y1, wy2), wy1)          #y1 >0 且<1 把 windows( 长宽均为1）外的box拉到windows的边界上
    x1 = tf.maximum(tf.minimum(x1, wx2), wx1)
    y2 = tf.maximum(tf.minimum(y2, wy2), wy1)
    x2 = tf.maximum(tf.minimum(x2, wx2), wx1)
    clipped = tf.concat([y1, x1, y2, x2], axis=1, name="clipped_boxes")       # concat和 stack不一样，按照我的理解stack应该会增加一个维度 而 concat是拼起来
    clipped.set_shape((clipped.shape[0], 4))
    return clipped

class SProposalLayer(KE.Layer):
    """Receives anchor scores and selects a subset to pass as proposals
    to the second stage. Filtering is done based on anchor scores and
    non-max suppression to remove overlaps. It also applies bounding
    box refinement deltas to anchors.

    Inputs:
        0 srpn_class_probs: [batch,num_anchors, (bg prob, fg prob)]
        1 srpn_bbox: [batch,num_anchors, (dx0 ,log(w0), dx1, log(w1), dy, log(dh)]
        2 anchors: [batch,num_anchors, (y1, x1, y2, x2)]               left and right use the same original anchors

    Returns:
        Proposals in normalized coordinates [batch, rois, (y1, x1, y2, x2)]  proposal_left,proposal_right
    """

    def __init__(self, proposal_count, nms_threshold, config=None, **kwargs):
        super(SProposalLayer, self).__init__(**kwargs)
        self.config = config
        self.proposal_count = proposal_count
        self.nms_threshold = nms_threshold

    def call(self, inputs):

        # Box Scores. Use the foreground class confidence. [batch, num_rois, 1]
        scores = inputs[0][:,:,1]
        #scores=KL.Lambda(lambda input: inputs[0][:,:,1])(inputs)

        # Box regress info  [batch,num_rois, 6]
        deltas_group = inputs[1]          #[batch, anchorsnum , [dx0 log(w0) dx1 log(w1) dy log(dh)]]

        deltas_group = deltas_group * np.reshape(self.config.RPN_BBOX_STD_DEV, [ 1, 6])
        # get x0,x1 ....
        dx0,log_dw0, dx1,log_dw1, dy,log_dh =tf.split(deltas_group,6,axis=-1)

        #left and right Anchors         [batch,anchor_count, (y1, x1, y2, x2)]
        anchors_original = inputs[2]

        proposals_index_group=[]      # the index of the box of left and right images after nms    [left_proposals,right_propsals]
        boxes_group = []
        # process left and right resp ectively
        for i in range(2):
            # seperate deltas        convert to : [N, (dy, dx, log(dh), log(dw))]
            deltas = None
            if i==0:                # left
                 deltas = tf.concat([dy,dx0,log_dh ,log_dw0],axis=-1)  # 待测试
            elif i==1:
                 deltas = tf.concat([dy,dx1,log_dh ,log_dw1],axis=-1)  # 待测试
            anchors = anchors_original

            # Improve performance by trimming to top anchors by score   先过滤掉一部分分数低的 anchor 选出 最高的  PRE_NMS_LIMIT 个
            # and doing the rest on the smaller subset.
            pre_nms_limit = tf.minimum(self.config.PRE_NMS_LIMIT, tf.shape(anchors)[1])

            ix = tf.nn.top_k(scores, pre_nms_limit, sorted=True,  # 应该是挑选出分数最高的k个anchor
                             name="top_anchors").indices
            # filter

            scores = Mutils.batch_slice([scores, ix], lambda x, y: tf.gather(x, y),
                                       self.config.IMAGES_PER_GPU)
            deltas = Mutils.batch_slice([deltas, ix],lambda x, y: tf.gather(x, y),
                                       self.config.IMAGES_PER_GPU)           #deltas are different for left and rignt  scores and anchors are the same for left and rignt
            anchors = Mutils.batch_slice([anchors, ix], lambda x, y: tf.gather(x, y),
                                        self.config.IMAGES_PER_GPU,names=['anchors_pre_nms'])

            # Apply deltas to anchors to get refined anchors.  get left_boxes and right_boxes
            # [batch, N, (y1, x1, y2, x2)]

            boxes = Mutils.batch_slice([anchors,deltas],lambda x,y: apply_box_deltas_graph(x,y),
                                       self.config.IMAGES_PER_GPU,names=["refined_anchors"])
            # Clip to image boundaries. Since we're in normalized coordinates,
            # clip to 0..1 range. [batch, N, (y1, x1, y2, x2)]
            window = np.array([0, 0, 1, 1], dtype=np.float32)

            # 将 box弄到window下   box [batch , n ,4]
            boxes = Mutils.batch_slice([boxes],lambda x:  clip_boxes_graph(x,window),
                                       self.config.IMAGES_PER_GPU,names=['refined_anchors_clipped'])
            boxes_group.append(boxes)
            # Non-max suppression

            name=None
            if i==0:
                name = 'rpn_non_max_suppression_left'
            elif i==1:
                name = 'rpn_non_max_suppression_right'
            # box [ batch,N,4]
            index = Mutils.batch_slice([boxes,scores],
                                       lambda boxes_one,scores_one:tf.image.non_max_suppression(
                                           boxes_one,scores_one,  self.proposal_count ,self.nms_threshold
                                       ),self.config.IMAGES_PER_GPU,names=[name]
                                       )
            # index [batch, N]
            proposals_index_group.append(index)

        # choose the proposals that both in left and right after nms
        # proposals_index_group shape: [batch ,N ]
        proposals_index =Mutils.batch_slice([proposals_index_group[0], proposals_index_group[1]], lambda x, y:  index_intersection(x, y),
                                        self.config.IMAGES_PER_GPU,names=['proposals_index_intersection'])

        scores = Mutils.batch_slice([scores,  proposals_index], lambda x, y: tf.gather(x, y),
                                    self.config.IMAGES_PER_GPU,names=['scores_intersection'])
        #choose the top  proposal_count porposals
        # scores : [batch,N,1]



        proposals_left = Mutils.batch_slice([boxes_group[0],proposals_index], lambda x,y:tf.gather(x,y),
                                            self.config.IMAGES_PER_GPU,names=['left_proposals']
                                            )
        proposals_right = Mutils.batch_slice([boxes_group[1], proposals_index], lambda x, y: tf.gather(x, y),
                                            self.config.IMAGES_PER_GPU, names=['right_proposals']
                                            )

        # padding if needed     proposals_left : [batch ,num ,4]
        padding = tf.maximum(self.proposal_count - tf.shape(proposals_left)[1], 0)
        proposals_left = tf.pad(proposals_left , [(0,0), (0, padding), (0, 0)], name ='proposals_left_after_padding' )
        proposals_right = tf.pad(proposals_right, [(0, 0), (0, padding), (0, 0)], name='proposals_right_after_padding')

        return [proposals_left,proposals_right]


    def compute_output_shape(self, input_shape):
        assert isinstance(input_shape, list)
        return [( None, self.proposal_count, 4),( None, self.proposal_count, 4)]

# adopt the code from : https://github.com/chonepieceyb/Mask_RCNN/blob/master/mrcnn/model.py
def norm_boxes_graph(boxes, shape):
    """Converts boxes from pixel coordinates to normalized coordinates.
    boxes: [..., (y1, x1, y2, x2)] in pixel coordinates
    shape: [..., (height, width)] in pixels

    Note: In pixel coordinates (y2, x2) is outside the box. But in normalized
    coordinates it's inside the box.

    Returns:
        [..., (y1, x1, y2, x2)] in normalized coordinates
    """
    h, w = tf.split(tf.cast(shape, tf.float32), 2)
    scale = tf.concat([h, w, h, w], axis=-1) - tf.constant(1.0)
    shift = tf.constant([0., 0., 1., 1.])
    return tf.divide(boxes - shift, scale)

###############################################################################
#   Detection Target Layer
##############################################################################
# I adopt and modified the code from :https://github.com/chonepieceyb/Mask_RCNN/blob/master/mrcnn/model.py
def trim_zeros_graph(boxes, name='trim_zeros'):
    """Often boxes are represented with matrices of shape [N, 4] and
    are padded with zeros. This removes zero boxes.

    boxes: [N, 4] matrix of boxes.
    non_zeros: [N] a 1D boolean mask identifying the rows to keep
    """

    non_zeros = tf.cast(tf.reduce_sum(tf.abs(boxes), axis=1), tf.bool)
    boxes = tf.boolean_mask(boxes, non_zeros, name=name)

    return boxes, non_zeros

def overlaps_graph(boxes1, boxes2):
    """Computes IoU overlaps between two sets of boxes.
    boxes1, boxes2: [N, (y1, x1, y2, x2)].
    """
    # 1. Tile boxes2 and repeat boxes1. This allows us to compare
    # every boxes1 against every boxes2 without loops.
    # TF doesn't have an equivalent to np.repeat() so simulate it
    # using tf.tile() and tf.reshape.
    b1 = tf.reshape(tf.tile(tf.expand_dims(boxes1, 1),
                            [1, 1, tf.shape(boxes2)[0]]), [-1, 4])
    b2 = tf.tile(boxes2, [tf.shape(boxes1)[0], 1])
    # 2. Compute intersections
    b1_y1, b1_x1, b1_y2, b1_x2 = tf.split(b1, 4, axis=1)
    b2_y1, b2_x1, b2_y2, b2_x2 = tf.split(b2, 4, axis=1)
    y1 = tf.maximum(b1_y1, b2_y1)
    x1 = tf.maximum(b1_x1, b2_x1)
    y2 = tf.minimum(b1_y2, b2_y2)
    x2 = tf.minimum(b1_x2, b2_x2)
    intersection = tf.maximum(x2 - x1, 0) * tf.maximum(y2 - y1, 0)
    # 3. Compute unions
    b1_area = (b1_y2 - b1_y1) * (b1_x2 - b1_x1)
    b2_area = (b2_y2 - b2_y1) * (b2_x2 - b2_x1)
    union = b1_area + b2_area - intersection
    # 4. Compute IoU and reshape to [boxes1, boxes2]
    iou = intersection / union
    overlaps = tf.reshape(iou, [tf.shape(boxes1)[0], tf.shape(boxes2)[0]])
    return overlaps



class DetectionTargetLayer(KE.Layer):
    """Subsamples proposals and generates target box refinement, class_ids,
    and masks for each.

    Inputs:
    0  proposals_left: proppsal_left  are :[batch, N, (y1, x1, y2, x2)] in normalized coordinates. Might
               be zero padded if there are not enough proposals.
    1  proposals_right:[batch, N, (y1, x1, y2, x2)] in normalized coordinates. Might
               be zero padded if there are not enough proposals.
    2  gt_class_ids: [batch,MAX_GT_INSTANCES] Integer class IDs.
    3  gt_boxes_left :    gt_boxes_left are [ batch, MAX_GT_INSTANCES, (y1, x1, y2, x2)] in normalized
              coordinates.
    4  gt_boxes_right:    gt_boxes _right are [ batch, MAX_GT_INSTANCES, (y1, x1, y2, x2)] in normalized
              coordinates.

    Returns: Target left and right ROIs and corresponding class IDs, left and right  bounding box shifts,

    0  rois_group: [rois_left,rois_right]  rois_left and rois_right: [ batch, TRAIN_ROIS_PER_IMAGE, (y1, x1, y2, x2)] in normalized
          coordinates
    2  target_class_ids: [batch, TRAIN_ROIS_PER_IMAGE]. Integer class IDs.
    3  target_deltas_group: [target_deltas_left,target_deltas_right]  where target_deltas : [ batch,TRAIN_ROIS_PER_IMAGE, (dy, dx, log(dh), log(dw)]
    4  target_gt_boxes_group


    Note: Returned arrays might be zero padded if not enough target ROIs.
    """

    def __init__(self, config, **kwargs):
        super(DetectionTargetLayer, self).__init__(**kwargs)
        self.config = config

    def call(self, inputs):
        proposals_left = inputs[0]
        proposals_right = inputs[1]
        proposals_group = [proposals_left,proposals_right]
        gt_class_ids = inputs[2]
        gt_boxes_left = inputs[3]
        gt_boxes_right = inputs[4]
        gt_boxes_group = [ gt_boxes_left,gt_boxes_right]

        # Assertions
        asserts = [
            tf.Assert(tf.greater(tf.shape(proposals_group[0])[1], 0), [proposals_group[0],proposals_group[1]],
                      name="roi_assertion"),
        ]
        with tf.control_dependencies(asserts):
            proposals_group[0] = tf.identity(proposals_group[0],name='proposals_identity_left')
            proposals_group[1] = tf.identity(proposals_group[1],name='proposals_identity_right')


        # Remove zero padding
        proposals_group = [ Mutils.batch_slice([proposals],
            lambda x: trim_zeros_graph(x)[0],self.config.IMAGES_PER_GPU, names=['trim_proposals'+ str(i)]
        ) for i, proposals in enumerate(proposals_group)]

        non_zeros = None


        for i in range(2):
            gt_boxes = gt_boxes_group[i]
            if i==0:
                gt_boxes =Mutils.batch_slice([gt_boxes], lambda x: trim_zeros_graph(x)[0] ,
                self.config.IMAGES_PER_GPU, names=["trim_gt_boxes"+'left'])
                non_zeros = Mutils.batch_slice([gt_boxes], lambda x: trim_zeros_graph(x)[1],
                                              self.config.IMAGES_PER_GPU, names=["trim_gt_boxes" + 'left'])

            elif i==1:
                gt_boxes= Mutils.batch_slice([gt_boxes], lambda x: trim_zeros_graph(x)[0],
                self.config.IMAGES_PER_GPU, names=["trim_gt_boxes" + 'right'])

            gt_boxes_group[i] = gt_boxes

        # 这里选择以第一张图的 non_zeros 为主 因为 左右两张是一一对应的 （所以 zeros pad 完全相同 （理论上 不出bug))
        gt_class_ids =  Mutils.batch_slice([gt_class_ids, non_zeros],lambda x,y:tf.boolean_mask(x,y),
                        self.config.IMAGES_PER_GPU, names=["trim_gt_class_ids"])


        indices_positive_group = []      #the indices of left and right proposals after subsamping
        indices_negative_group = []
        overlaps_group=[]
        for i in range(2):
            # subsample the proposals of left and right  0:left 1:right
            proposals = proposals_group[i]
            gt_boxes = gt_boxes_group [i]

            # Compute overlaps matrix [proposals, gt_boxes]
            overlaps = Mutils.batch_slice([proposals, gt_boxes],lambda x,y:overlaps_graph(x,y),
                        self.config.IMAGES_PER_GPU,names=['cal_overlaps'+str(i)])     #output [batch, len(proposals), len(gt_boxes)]
            overlaps_group.append(overlaps)

            # Determine positive and negative ROIs
            roi_iou_max = tf.reduce_max(overlaps, axis=2,name ='redece_max'+str(i))       # calculate the max overlap with any gt_box  of the proposal ,output shape [batch,N]

            # 1. Positive ROIs are those with >= 0.5 IoU with a GT box
            positive_indices = Mutils.batch_slice([roi_iou_max],lambda x :tf.where(x >= 0.5)[:, 0],
                        self.config.IMAGES_PER_GPU,names=['positive_indices_1_'+str(i)])

            # 2. Negative ROIs are those with < 0.5 with every GT box. Skip crowds.
            negative_indices = Mutils.batch_slice([roi_iou_max], lambda x: tf.where(tf.logical_and(x < 0.5,x>=0.1))[:, 0],
                                                  self.config.IMAGES_PER_GPU,names=['negative_indices_1_'+str(i)])

            indices_positive_group.append(positive_indices)
            indices_negative_group.append(negative_indices)

        # cal the intersion of positive_indices of left and right
        positive_indices = Mutils.batch_slice([indices_positive_group[0],indices_positive_group[1]],
                                              lambda x,y :index_intersection(x,y),
                            self.config.IMAGES_PER_GPU,names=['positive_indices_2'])      # shape [ batch, N]


        # cal the union of negative_indices of left and right
        negative_indices = Mutils.batch_slice([ indices_negative_group[0], indices_negative_group[1]],
                                              lambda x, y: index_union(x, y),
                         self.config.IMAGES_PER_GPU,names=['negative_indices_2'])# shape [ batch, N]

        # Assign positive ROIs to GT boxes.
        positive_overlaps_group = [
            Mutils.batch_slice([ overlaps , positive_indices],
                               lambda x, y:tf.gather(x, y),
            self.config.IMAGES_PER_GPU, names=['positive_overlaps'+str(i)]) for overlaps in overlaps_group
        ]   # [positive_overlaps_left,positive_overlaps_right]  where the positive_overlaps_left are [batch, len(proposal),len(gt)]



        roi_gt_box_assignment_group = [tf.cond(
            tf.greater(tf.shape(positive_overlaps)[1], 0),
            true_fn=lambda: tf.argmax(positive_overlaps, axis=2),
            false_fn=lambda: tf.cast(tf.constant([]), tf.int64) ,name='tf_cond'          #这里可能出错
        )  for positive_overlaps in positive_overlaps_group]             # if true_fn the output is (batch,len(proposal))

        # filter those left and right gt_box_assignment dose not equal
        positive_indices =  Mutils.batch_slice([ positive_indices , roi_gt_box_assignment_group[0],roi_gt_box_assignment_group[1]],
                               lambda x, y,z :tf.gather(x,tf.where(tf.equal(y,z))[:,0]),
            self.config.IMAGES_PER_GPU,names=['positive_indices_3'])

        # Subsample ROIs. Aim for 33% positive
        # Positive ROIs
        positive_count = int(self.config.TRAIN_ROIS_PER_IMAGE *
                             self.config.ROI_POSITIVE_RATIO)

        positive_indices = Mutils.batch_slice([ positive_indices],
                               lambda x :tf.random_shuffle(x)[:positive_count],
            self.config.IMAGES_PER_GPU, names=['positive_indices_shuffle'])

        positive_count = tf.shape(positive_indices)[0]
        # Negative ROIs. Add enough to maintain positive:negative ratio.
        r = 1.0 / self.config.ROI_POSITIVE_RATIO
        negative_count = tf.cast(r * tf.cast(positive_count, tf.float32), tf.int32) - positive_count

        negative_indices = Mutils.batch_slice([ negative_indices],
                                              lambda x: tf.random_shuffle(x)[0:negative_count],
                                              self.config.IMAGES_PER_GPU, names=['negative_indices_shuffle'])



        # Gather selected ROIs
        positive_rois_group = [
            Mutils.batch_slice([ proposals, positive_indices],
                               lambda x,y : tf.gather(x,y),
                               self.config.IMAGES_PER_GPU, names=['positive_rois'+str(i)])
            for i,proposals in enumerate(proposals_group)
        ]
        negative_rois_group = [
            Mutils.batch_slice([proposals, negative_indices],
                               lambda x, y: tf.gather(x, y),
                               self.config.IMAGES_PER_GPU, names=['negative_rois' + str(i)])
            for i, proposals in enumerate(proposals_group)
        ]

        # select gt_box_assignment by positive indices
        roi_gt_box_assignment_group = [
            Mutils.batch_slice([roi_gt_box_assignment, positive_indices],
                               lambda x, y: tf.gather(x, y),
                               self.config.IMAGES_PER_GPU, names=['roi_gt_box_assignment' + str(i)])
            for i, roi_gt_box_assignment in enumerate(roi_gt_box_assignment_group)
        ]

        roi_gt_boxes_group = [
            Mutils.batch_slice([x[0], x[1]],
                               lambda x, y: tf.gather(x, y),
                               self.config.IMAGES_PER_GPU, names=['roi_gt_boxes' + str(i)])
            for i,x in enumerate(list(zip(gt_boxes_group ,roi_gt_box_assignment_group)))
        ]

        # 选择 左边的，其实左边和右边的应该是一一对应的,上一步已经去掉了那些不同的
        roi_gt_class_ids = Mutils.batch_slice([gt_class_ids, roi_gt_box_assignment_group[0]],
                                              lambda x,y:  tf.gather(x,y),
                                              self.config.IMAGES_PER_GPU, names=['roi_gt_class_ids'])


        # Compute bbox refinement for positive ROIs
        deltas_group = [
            Mutils.batch_slice([positive_rois, roi_gt_boxes],
                               lambda x, y: Mutils.box_refinement_graph(x,y),
                               self.config.IMAGES_PER_GPU,names=['deltas'+str(i)])
            for positive_rois, roi_gt_boxes in list(zip(positive_rois_group, roi_gt_boxes_group))
        ]

        # deltas_group : [2,batch,N,6]
        for i in range(2):
            deltas_group[i] /= self.config.BBOX_STD_DEV     # 可能有错误

        # Append negative ROIs_group and pad bbox deltas and masks that
        # are not used for negative ROIs with zeros.
        # positive_rois_group shape [2,batch,N,4]


        rois_group =[tf.concat([p, n], axis=1) for p,n in list(zip(positive_rois_group,negative_rois_group))]
        N = tf.shape(negative_rois_group[0])[1]
        P = tf.maximum(self.config.TRAIN_ROIS_PER_IMAGE - tf.shape(rois_group[0])[1], 0)

        #padding
        rois_group = [
            tf.pad(rois, [(0,0),(0, P), (0, 0)],name='rois_group'+str(i))
            for i, rois in enumerate(rois_group)
        ]
        roi_gt_boxes_group = [tf.pad(roi_gt_boxes, [(0,0),(0, N + P), (0, 0)],name='roi_gt_boxes'+str(i)) for i, roi_gt_boxes in enumerate(roi_gt_boxes_group)]
        roi_gt_class_ids = tf.pad(roi_gt_class_ids, [(0,0),(0, N + P)],name='roi_gt_class_ids')
        deltas_group = [tf.pad(deltas, [(0,0),(0, N + P), (0, 0)], name = 'rois_bbox'+str(i)) for i, deltas in enumerate(deltas_group)]

        # combine the deltas_left and deltas_right to the shape of [dx0,log(dw0),dx1,log(d(w1)), dy, log(dh)] ,
        # the deltas_left are [ batch,TRAIN_ROIS_PER_IMAGE, (dy, dx, log(dh), log(dw)]
        # and because the left and right y should be the same ( we design it use the same y gt and the same y regression item)
        # use the y and h of left_deltas as the y and h of deltas_union
        dy,dx0,log_dh,log_dw0 = tf.split(deltas_group[0],4,axis=-1)
        _, dx1, _, log_dw1 = tf.split(deltas_group[1], 4, axis=-1)
        deltas_union = tf.concat([dx0,log_dw0,dx1,log_dw1,dy,log_dh],axis=-1)

        output = [
            KL.Lambda(lambda x:x,name ='rois_left')(rois_group[0]),
            KL.Lambda(lambda x: x, name='rois_right')( rois_group[1]),
            KL.Lambda(lambda x: x, name='rois_gt_class_ids')( roi_gt_class_ids),
            KL.Lambda(lambda x: x, name='deltas_union')(deltas_union)
            ]

        return output

    def compute_output_shape(self, input_shape):
        return [
            (None, self.config.TRAIN_ROIS_PER_IMAGE, 4),  # rois_group
            (None, self.config.TRAIN_ROIS_PER_IMAGE,4),  # class_ids
            (None, self.config.TRAIN_ROIS_PER_IMAGE),  # deltas_group
            (None, self.config.TRAIN_ROIS_PER_IMAGE,6)    # gt_boxes_group
        ]

#####################################################################################
# fpn_srcnn_model
#####################################################################################
def log2_graph(x):
    """Implementation of Log2. TF doesn't have a native implementation."""
    return tf.log(x) / tf.log(2.0)

# I adopt the code and modify it from : https://github.com/chonepieceyb/Mask_RCNN/blob/master/mrcnn/model.py
class SPyramidROIAlign(KE.Layer):
    """Implements ROI Pooling on multiple levels of the feature pyramid.

    Params:
    - pool_shape: [batch, pool_height, pool_width] of the output pooled regions. Usually [7, 7]
    - img_type: left or right image
    Inputs:
    - boxes: [batch,num_boxes, (y1, x1, y2, x2)] in normalized
             coordinates. Possibly padded with zeros if not enough
             boxes to fill the array.
    - image_meta: [batch,(meta data)] Image details. See compose_image_meta()
    - feature_maps: List of feature maps from different levels of the pyramid.
                    Each is [batch, height, width, channels]

    Output:
    Pooled regions in the shape: [batch, num_boxes, pool_height, pool_width, channels].
    The width and height are those specific in the pool_shape in the layer
    constructor.
    """

    def __init__(self, pool_shape,img_type, **kwargs):
        super(SPyramidROIAlign, self).__init__(**kwargs)
        assert( img_type in ['left','right'])
        self.img_type = img_type
        self.pool_shape = tuple(pool_shape)

    def call(self, inputs):
        # Crop boxes [batch, num_boxes, (y1, x1, y2, x2)] in normalized coords
        boxes = inputs[0]

        # Image meta
        # Holds details about the image. See compose_image_meta()
        image_meta = inputs[1]

        # Feature Maps. List of feature maps from different level of the
        # feature pyramid. Each is [batch, height, width, channels]
        feature_maps = inputs[2:]

        # Assign each ROI to a level in the pyramid based on the ROI area.
        y1, x1, y2, x2 = tf.split(boxes, 4, axis=2)
        h = y2 - y1
        w = x2 - x1
        # Use shape of first image. Images in a batch must have the same size.  image_shape : [batch, 4]
        image_shape = None
        if self.img_type =='left':
            image_shape = parse_image_meta_graph(image_meta)['image_shape_left'][0]
        elif self.img_type == 'right':
            image_shape = parse_image_meta_graph(image_meta)['image_shape_right'][0]

        # Equation 1 in the Feature Pyramid Networks paper. Account for
        # the fact that our coordinates are normalized here.
        # e.g. a 224x224 ROI (in pixels) maps to P4


        image_area = tf.cast(image_shape[0] * image_shape[1], tf.float32)
        roi_level = log2_graph(tf.sqrt(h * w) / (224.0 / tf.sqrt(image_area)))
        roi_level = tf.minimum(5, tf.maximum(
            2, 4 + tf.cast(tf.round(roi_level), tf.int32)))
        roi_level = tf.squeeze(roi_level, 2)

        # Loop through levels and apply ROI pooling to each. P2 to P5.
        pooled = []
        box_to_level = []
        for i, level in enumerate(range(2, 6)):
            # roi_level  [batch,box_num]
            ix = tf.where(tf.equal(roi_level, level))
            level_boxes = tf.gather_nd(boxes, ix)

            # Box indices for crop_and_resize.  track the image_ind which is batch id
            box_indices = tf.cast(ix[:, 0], tf.int32)

            # Keep track of which box is mapped to which level
            box_to_level.append(ix)

            # Stop gradient propogation to ROI proposals
            level_boxes = tf.stop_gradient(level_boxes)
            box_indices = tf.stop_gradient(box_indices)

            # Crop and Resize
            # From Mask R-CNN paper: "We sample four regular locations, so
            # that we can evaluate either max or average pooling. In fact,
            # interpolating only a single value at each bin center (without
            # pooling) is nearly as effective."
            #
            # Here we use the simplified approach of a single value per bin,
            # which is how it's done in tf.crop_and_resize()
            # Result: [batch * num_boxes, pool_height, pool_width, channels]
            pooled.append(tf.image.crop_and_resize(
                feature_maps[i], level_boxes, box_indices, self.pool_shape,
                method="bilinear"))

        # Pack pooled features into one tensor
        pooled = tf.concat(pooled, axis=0)

        # Pack box_to_level mapping into one array and add another
        # column representing the order of pooled boxes
        box_to_level = tf.concat(box_to_level, axis=0)
        box_range = tf.expand_dims(tf.range(tf.shape(box_to_level)[0]), 1)
        box_to_level = tf.concat([tf.cast(box_to_level, tf.int32), box_range],
                                 axis=1)

        # Rearrange pooled features to match the order of the original boxes
        # Sort box_to_level by batch then box index
        # TF doesn't have a way to sort by two columns, so merge them and sort.
        sorting_tensor = box_to_level[:, 0] * 100000 + box_to_level[:, 1]       # batch 排序, 把混在一起的feature 按 batch 分开
        ix = tf.nn.top_k(sorting_tensor, k=tf.shape(
            box_to_level)[0]).indices[::-1]  # 倒叙
        ix = tf.gather(box_to_level[:, 2], ix)
        pooled = tf.gather(pooled, ix)

        # Re-add the batch dimension
        shape = tf.concat([tf.shape(boxes)[:2], tf.shape(pooled)[1:]], axis=0)
        pooled = KL.Lambda(lambda x : tf.reshape(x, shape))(pooled)
        return pooled


    def compute_output_shape(self, input_shape):
        return input_shape[0][:2] + self.pool_shape + (input_shape[2][-1], )

def fpn_classifier_graph(rois_group, feature_maps_group, image_meta,
                         pool_size, num_classes, train_bn=True,
                         fc_layers_size=1024):
    """Builds the computation graph of the feature pyramid network classifier
    and regressor heads.

    rois_group : [rois_left,rois_right] where rois_left: [batch, num_rois, (y1, x1, y2, x2)] Proposal boxes in normalized
          coordinates.
    feature_maps_group : [ feature_maps_left, feature_maps_right]  feature_maps_left:List of feature maps from different layers of the pyramid,
                  [P2, P3, P4, P5]. Each has a different resolution.
    image_meta: [batch, (meta data)] Image details. See compose_image_meta()
    pool_size: The width of the square feature map generated from ROI Pooling.
    num_classes: number of classes, which determines the depth of the results
    train_bn: Boolean. Train or freeze Batch Norm layers
    fc_layers_size: Size of the 2 FC layers

    Returns:
        logits: [batch, num_rois, NUM_CLASSES] classifier logits (before softmax)
        probs: [batch, num_rois, NUM_CLASSES] classifier probabilities
        bbox_deltas: [batch, num_rois, NUM_CLASSES, (dx0,logdw0,dx1,logdw1,dy,logdh)] Deltas to apply to
                     left and right proposal boxes
    """
    # ROI Pooling
    # Shape: [batch, num_rois, POOL_SIZE, POOL_SIZE, channels]
    # aplay left and right roi_aligh respectively
    left =SPyramidROIAlign([pool_size, pool_size],'left',
                        name="roi_align_classifier_left")([rois_group[0], image_meta] + feature_maps_group[0])
    right = SPyramidROIAlign([pool_size, pool_size],'right',
                        name="roi_align_classifier")([rois_group[1], image_meta] + feature_maps_group[1])
    # concat left and right feature map
    # left shape: [batch,N,H,W,C]  right :[batch,N,H,W,C]  -> [batch,N,H,W,2C]
    x= KL.Concatenate(axis=-1,name='shared_fpn_features')([left,right])
    #x = KL.Lambda(lambda x,y : tf.concat([x,y],axis=-1),name ='shared_fpn_features')([left,right])
    # Two 1024 FC layers (implemented with Conv2D for consistency)
    x = KL.TimeDistributed(KL.Conv2D(fc_layers_size, (pool_size, pool_size), padding="valid"),
                           name="srcnn_class_conv1")(x)
    x = KL.TimeDistributed(BatchNorm(), name='srcnn_class_bn1')(x, training=train_bn)
    x = KL.Activation('relu')(x)
    x = KL.TimeDistributed(KL.Conv2D(fc_layers_size, (1, 1)),
                           name="srcnn_class_conv2")(x)
    x = KL.TimeDistributed(BatchNorm(), name='srcnn_class_bn2')(x, training=train_bn)
    x = KL.Activation('relu')(x)

    shared = KL.Lambda(lambda x: K.squeeze(K.squeeze(x, 3), 2),
                       name="pool_squeeze")(x)

    # Classifier head
    srcnn_class_logits = KL.TimeDistributed(KL.Dense(num_classes),
                                            name='srcnn_class_logits')(shared)
    srcnn_probs = KL.TimeDistributed(KL.Activation("softmax"),       #这里区分开使用 TimeDistributed来应该是要和之前的区分开
                                     name="srcnn_class")(srcnn_class_logits)

    # BBox head
    # [batch, num_rois, NUM_CLASSES * (dx0,logdw0,dx1,logdw1,dy,logdh)]
    x = KL.TimeDistributed(KL.Dense(num_classes * 6, activation='linear'),
                           name='srcnn_bbox_fc')(shared)
    # Reshape to [batch, num_rois, NUM_CLASSES, (dx0,logdw0,dx1,logdw1,dy,logdh)]

    s = K.int_shape(x)
    num_rosi = s[1]
    if num_rosi==None:
        num_rosi=-1
    srcnn_bbox = KL.Reshape((num_rosi, num_classes, 6), name="srcnn_bbox")(x)
    return srcnn_class_logits, srcnn_probs, srcnn_bbox

######################################################################################
# loss function
#####################################################################################
#######################################
# SRPN LOSS
#######################################

# I adopt this code and modified it from :https://github.com/chonepieceyb/Mask_RCNN/blob/master/mrcnn/model.py
def smooth_l1_loss(y_true, y_pred, inside_weight =1.0, outside_wight=1.0):
    '''

    :param y_true:  ture_bbox [N, ()]
    :param y_pred:  predict_bbox [N.()]
    :param inside_weight:
    :param outside_wight:
    :return:  loss
    '''
    diff = (y_true - y_pred)
    diff = tf.multiply(tf.cast(diff,'float32'),inside_weight)
    diff = K.abs(diff)

    less_than_one = K.cast(K.less(diff,  1.0), "float32")
    loss = (  less_than_one  * 0.5 * diff** 2) + (1 - less_than_one) * (diff - 0.5)

    loss = tf.multiply(loss,outside_wight)
    return loss

def srpn_class_loss_graph(rpn_match, rpn_class_logits):
    """RPN anchor classifier loss.

    rpn_match: [batch, anchors, 1]. Anchor match type. 1=positive,
               -1=negative, 0=neutral anchor.
    rpn_class_logits: [batch, anchors, 2]. RPN classifier logits for BG/FG.
    """
    # Squeeze last dim to simplify
    rpn_match = tf.squeeze(rpn_match, -1)
    # Get anchor classes. Convert the -1/+1 match to 0/1 values.
    anchor_class = K.cast(K.equal(rpn_match, 1), tf.int32)
    # Positive and Negative anchors contribute to the loss,
    # but neutral anchors (match value = 0) don't.
    indices = tf.where(K.not_equal(rpn_match, 0))
    # Pick rows that contribute to the loss and filter out the rest.
    rpn_class_logits = tf.gather_nd(rpn_class_logits, indices)
    anchor_class = tf.gather_nd(anchor_class, indices)
    # Cross entropy loss
    loss = K.sparse_categorical_crossentropy(target=anchor_class,
                                             output=rpn_class_logits,
                                             from_logits=True)
    loss = K.switch(tf.size(loss) > 0, K.mean(loss), tf.constant(0.0))
    return loss


def srpn_bbox_loss_graph(config, target_bbox, rpn_match, rpn_bbox):
    """Return the RPN bounding box loss graph.

    config: the model config object.
    target_bbox: [batch, max positive anchors, (dy, dx, log(dh), log(dw))].
        Uses 0 padding to fill in unsed bbox deltas.
    rpn_match: [batch, anchors, 1]. Anchor match type. 1=positive,
               -1=negative, 0=neutral anchor.
    rpn_bbox: [batch, anchors, (dy, dx, log(dh), log(dw))]
    """
    # Positive anchors contribute to the loss, but negative and
    # neutral anchors (match value of 0 or -1) don't.
    rpn_match = K.squeeze(rpn_match, -1)
    indices = tf.where(K.equal(rpn_match, 1))

    # Pick bbox deltas that contribute to the loss
    rpn_bbox = tf.gather_nd(rpn_bbox, indices)

    # Trim target bounding box deltas to the same length as rpn_bbox.
    batch_counts = K.sum(K.cast(K.equal(rpn_match, 1), tf.int32), axis=1)
    target_bbox = batch_pack_graph(target_bbox, batch_counts,
                                   config.IMAGES_PER_GPU)

    loss = smooth_l1_loss(target_bbox, rpn_bbox)

    loss = K.switch(tf.size(loss) > 0, K.mean(loss), tf.constant(0.0))
    return loss

def srcnn_class_loss_graph(target_class_ids, pred_class_logits):
    """Loss for the classifier head of S RCNN.

    target_class_ids: [batch, num_rois]. Integer class IDs. Uses zero
        padding to fill in the array.
    pred_class_logits: [batch, num_rois, num_classes]

    """
    # During model building, Keras calls this function with
    # target_class_ids of type float32. Unclear why. Cast it
    # to int to get around it.
    target_class_ids = tf.cast(target_class_ids, 'int64')

    # Find predictions of classes that are not in the dataset.

    # Loss
    loss = tf.nn.sparse_softmax_cross_entropy_with_logits(
        labels=target_class_ids, logits=pred_class_logits)


    # Computer loss mean.    divided num_class

    loss = tf.reduce_sum(loss) / pred_class_logits.get_shape().as_list()[2]
    return loss


def srcnn_bbox_loss_graph(target_bbox, target_class_ids, pred_bbox):
    """Loss for Mask R-CNN bounding box refinement.

    target_bbox: [batch, num_rois, (dy, dx, log(dh), log(dw))]
    target_class_ids: [batch, num_rois]. Integer class IDs.
    pred_bbox: [batch, num_rois, num_classes, (dy, dx, log(dh), log(dw))]
    """
    # Reshape to merge batch and roi dimensions for simplicity.
    target_class_ids = K.reshape(target_class_ids, (-1,))
    target_bbox = K.reshape(target_bbox, (-1, 4))
    pred_bbox = K.reshape(pred_bbox, (-1, K.int_shape(pred_bbox)[2], 4))

    # Only positive ROIs contribute to the loss. And only
    # the right class_id of each ROI. Get their indices.
    positive_roi_ix = tf.where(target_class_ids > 0)[:, 0]
    positive_roi_class_ids = tf.cast(
        tf.gather(target_class_ids, positive_roi_ix), tf.int64)
    indices = tf.stack([positive_roi_ix, positive_roi_class_ids], axis=1)

    # Gather the deltas (predicted and true) that contribute to loss
    target_bbox = tf.gather(target_bbox, positive_roi_ix)
    pred_bbox = tf.gather_nd(pred_bbox, indices)

    # Smooth-L1 Loss
    loss = K.switch(tf.size(target_bbox) > 0,
                    smooth_l1_loss(y_true=target_bbox, y_pred=pred_bbox),
                    tf.constant(0.0))
    loss = K.mean(loss)
    return loss

###########################################################################
#DATA
###########################################################################
# input_left_image、input_right_image、input_image_meta、input_srpn_match、 input_srpn_bbox_union、input_gt_class_ids、input_gt_boxes_left、input_gt_boxes_right

def load_image_gt(dataset,config,image_id):
    '''
    load the gt of one image of  which id is image_id
    :param dataset:
    :return:left_image、right_image、image_meta、gt_class_ids、gt_boxes_left、gt_boxes_right,gt_boxes_union
    '''

    image_left,image_right = dataset.load_image(image_id)
    original_shape_left = image_left.shape
    original_shape_right = image_left.shape
    # resize the image
    image_left,window_left,scale_left,padding_left,_= Mutils.resize_image(image_left,min_dim= config.IMAGE_MIN_DIM,max_dim=config.IMAGE_MAX_DIM,mode=config.IMAGE_RESIZE_MODE)
    image_right,window_right,scale_right,padding_right,_ = Mutils.resize_image(image_right,min_dim= config.IMAGE_MIN_DIM,max_dim=config.IMAGE_MAX_DIM,mode=config.IMAGE_RESIZE_MODE)
    assert(image_left.shape==image_right.shape)
    #  :return:  class_ids , boxes_left:[N,(y0,x0,y1,x1)] , boxes_right , boxes_union
    gt_class_ids,gt_boxes_left,gt_boxes_right,gt_boxes_union = dataset.load_image_gt_info(image_id)
    # image_id, original_image_shape_group, image_shape_group,
    #                        window_group, scale_group, active_class_ids):
    activity_class_id = np.arange(dataset.class_num)
    image_meta  =  compose_image_meta(image_id,
                                      [original_shape_left,original_shape_right],
                                      [image_left.shape,image_right.shape],
                                      [window_left,window_right],
                                      [scale_left,scale_right],
                                      activity_class_id
                                      )
    return image_left,image_right,image_meta,gt_class_ids,gt_boxes_left,gt_boxes_right,gt_boxes_union



# I adopt the code and modify it  from : https://github.com/chonepieceyb/Mask_RCNN/blob/master/mrcnn/model.py
def build_srpn_targets( anchors, gt_class_ids, gt_boxes_union, gt_boxes_left,gt_boxes_right,config):
    """Given the anchors and GT boxes, compute overlaps and identify positive
    anchors and deltas to refine them to match their corresponding GT boxes.

    anchors: [num_anchors, (y1, x1, y2, x2)]
    gt_class_ids: [num_gt_boxes] Integer class IDs.
    gt_boxes_union : [num_gt_boxes, (y1, x1, y2, x2)] of the union box of left and right box
    gt_boxes_left : [num_gt_boxes, (y1, x1, y2, x2)] of the the letf box used to calculate offset
    gt_boxes_right : [num_gt_boxes, (y1, x1, y2, x2)] of the right box used to calculate offset
    Returns:
    rpn_match: [N] (int32) matches between anchors and GT boxes.
               1 = positive anchor, -1 = negative anchor, 0 = neutral
    rpn_bbox: [N, (dy, dx, log(dh), log(dw))] Anchor bbox deltas.
    """

    # SRPN Match: 1 = positive anchor, -1 = negative anchor, 0 = neutral
    srpn_match = np.zeros([anchors.shape[0]], dtype=np.int32)
    # SRPN bounding boxes: [max anchors per image, (dx0,log(dw0),dw1,log(dw1),dy,log(dy)]  only for positive, the negative and netual are zero
    srpn_bbox_union = np.zeros((config.RPN_TRAIN_ANCHORS_PER_IMAGE, 6))

    # Compute overlaps [num_anchors, num_gt_boxes]
    overlaps = Mutils.compute_overlaps(anchors, gt_boxes_union)

    # Match anchors to GT Boxes
    # If an anchor overlaps a GT_union box with IoU >= 0.7 then it's positive.
    # If an anchor overlaps any GT_union box with IoU < 0.3 then it's negative.
    # Neutral anchors are those that don't match the conditions above,
    # and they don't influence the loss function.
    # However, don't keep any GT box unmatched (rare, but happens). Instead,
    # match it to the closest anchor (even if its max IoU is < 0.3).
    #
    # 1. Set negative anchors first. They get overwritten below if a GT_union box is
    # matched to them. Skip boxes in crowd areas.
    anchor_iou_argmax = np.argmax(overlaps, axis=1)  #行和列的overlaps矩阵   这里先获得下标  [N] N是anchor的个数
    anchor_iou_max = overlaps[np.arange(overlaps.shape[0]), anchor_iou_argmax]    #这里通过下标获取值 [N]
    srpn_match[anchor_iou_max < 0.3] = -1
    # 2. Set an anchor for each GT box (regardless of IoU value).
    # If multiple anchors have the same IoU match all of them

    gt_iou_argmax = np.argwhere(overlaps == np.max(overlaps, axis=0))[:,0]    # 按照 gt_box 把每一个gt_box的最高的 overlabs 挑出来, 并且给所有等于这些overlap的 anchor 赋值为 1
    srpn_match[gt_iou_argmax] = 1                                             # 这里是为了确保每一个gt_box都有 match对象，要用到每一个gt_box
    # 3. Set anchors with high overlap as positive.
    srpn_match[anchor_iou_max >= 0.7] = 1

    # Subsample to balance positive and negative anchors
    # Don't let positives be more than half the anchors
    ids = np.where(srpn_match == 1)[0]
    extra = len(ids) - (config.RPN_TRAIN_ANCHORS_PER_IMAGE // 2)
    if extra > 0:
        # Reset the extra ones to neutral
        ids = np.random.choice(ids, extra, replace=False)
        srpn_match[ids] = 0
    # Same for negative proposals
    ids = np.where(srpn_match == -1)[0]
    extra = len(ids) - (config.RPN_TRAIN_ANCHORS_PER_IMAGE -
                        np.sum(srpn_match == 1))
    if extra > 0:
        # Rest the extra ones to neutral
        ids = np.random.choice(ids, extra, replace=False)
        srpn_match[ids] = 0

    # For positive anchors, compute shift and scale needed to transform them
    # to match the corresponding GT boxes.
    ids = np.where(srpn_match == 1)[0]
    ix = 0  # index into rpn_bbox
    # TODO: use box_refinement() rather than duplicating the code here
    for i, a in zip(ids, anchors[ids]):
        # cal the offset of closest of left and right box to get regression item
        # Closest gt box (it might have IoU < 0.7 of gt_box_union to ensure every gt_union_box matches an anchor )
        gt_left = gt_boxes_left[anchor_iou_argmax[i]]
        gt_right = gt_boxes_right[anchor_iou_argmax[i]]

        # Convert coordinates to center plus width/height.
        # GT left Box
        gt_left_h =  gt_left[2] -  gt_left[0]
        gt_left_w = gt_left[3] - gt_left[1]
        gt_left_center_y = gt_left[0] + 0.5 * gt_left_h
        gt_left_center_x = gt_left[1] + 0.5 * gt_left_w

        # GT Right Box  use calculate w and center_x because left and right use the same h and y
        gt_right_w = gt_right[3] - gt_right[1]
        gt_right_center_x = gt_right[1] + 0.5 * gt_right_w

        # Anchor
        a_h = a[2] - a[0]
        a_w = a[3] - a[1]
        a_center_y = a[0] + 0.5 * a_h
        a_center_x = a[1] + 0.5 * a_w

        # Compute the bbox refinement that the RPN should predict.
        # the shape of srpn_bbox_union is  [ x0, log(dw0) , x1 , log(dw1), y0,log(dh)]   0 refer to left and 1 refer to right, h and y is the same for left and right
        srpn_bbox_union[ix] = [
            (gt_left_center_x-a_center_x)/a_w,                 # dx0
            np.log(gt_left_w/a_w),                             # log(dw0)
            (gt_right_center_x - a_center_x) / a_w,            # dx1
            np.log(gt_right_w / a_w),                          # log(dw1)
            (gt_left_center_y - a_center_y) / a_h,             # dy
            np.log(gt_left_h / a_h),                           # log(dy)
        ]
        # Normalize
        srpn_bbox_union[ix] /= config.RPN_BBOX_STD_DEV
        ix += 1

    return srpn_match, srpn_bbox_union

def data_generator(dataset, config, shuffle=True,batch_size=1):
    """A generator that returns images and corresponding target class ids,
    bounding box deltas,

    dataset: The Dataset object to pick data from
    config: The model config object
    shuffle: I f True, shuffles the samples before every epoch
    batch_size: How many images to return in each call

    Returns a Python generator. Upon calling next() on it, the
    generator returns two lists, inputs and outputs. The contents
    of the lists differs depending on the received arguments:
    inputs list:
    - images_left: [batch, H, W, C]
    - images_right：[batch, H, W, C]
    - image_meta: [batch, (meta data)] Image details. See compose_image_meta()
    - srpn_match: [batch, N] Integer (1=positive anchor, -1=negative, 0=neutral)
    - srpn_bbox: [batch, N, (dx0,log(dw0), dx1, log(dw1), y, log(dh))]       Anchor bbox deltas.
    - gt_class_ids: [batch, MAX_GT_INSTANCES] Integer class IDs
    - gt_boxes_left: [batch, MAX_GT_INSTANCES, (y1, x1, y2, x2)]
    - gt_boxes_right:[batch, MAX_GT_INSTANCES, (y1, x1, y2, x2)]
    -

    outputs list: Usually empty in regular training. But if detection_targets
        is True then the outputs list contains target class_ids, bbox deltas,
        and masks.
    """
    b = 0  # batch item index
    image_index = -1
    image_ids = np.copy(dataset.image_ids)
    error_count = 0

    # Anchors
    # [anchor_count, (y1, x1, y2, x2)]
    backbone_shapes = compute_backbone_shapes(config, config.IMAGE_SHAPE)
    anchors = Mutils.generate_pyramid_anchors(config.RPN_ANCHOR_SCALES,
                                             config.RPN_ANCHOR_RATIOS,
                                             backbone_shapes,
                                             config.BACKBONE_STRIDES,
                                             config.RPN_ANCHOR_STRIDE)

    # Keras requires a generator to run indefinitely.
    while True:
        try:
            # Increment index to pick next image. Shuffle if at the start of an epoch.
            image_index = (image_index + 1) % len(image_ids)
            if shuffle and image_index == 0:
                np.random.shuffle(image_ids)

            # Get GT bounding boxes and masks for image.
            image_id = image_ids[image_index]

            # If the image source is not to be augmented pass None as augmentation
            image_left, image_right, image_meta, gt_class_ids, gt_boxes_left, gt_boxes_right, gt_boxes_union = \
            load_image_gt(dataset, config, image_id)

            # Skip images that have no instances. This can happen in cases
            # where we train on a subset of classes and the image doesn't
            # have any of the classes we care about.
            gt_class_ids = np.array(gt_class_ids ,dtype=np.int32)
            if not np.any(gt_class_ids > 0):
                continue

            # RPN Targets
            srpn_match, srpn_bbox = build_srpn_targets( anchors, gt_class_ids, gt_boxes_union, gt_boxes_left,gt_boxes_right,config)

            # Init batch arrays
            if b == 0:
                batch_images_left = np.zeros(
                    (batch_size,) + image_left.shape, dtype=np.float32)
                batch_images_right = np.zeros(
                    (batch_size,) + image_right.shape, dtype=np.float32)
                batch_image_meta = np.zeros(
                    (batch_size,) + image_meta.shape, dtype=image_meta.dtype)
                batch_srpn_match = np.zeros(
                    [batch_size, anchors.shape[0], 1], dtype=srpn_match.dtype)
                batch_srpn_bboxes = np.zeros(
                    [batch_size, config.RPN_TRAIN_ANCHORS_PER_IMAGE, 6], dtype=srpn_bbox.dtype)
                batch_gt_class_ids = np.zeros(
                    (batch_size, config.MAX_GT_INSTANCES), dtype=np.int32)
                batch_gt_boxes_left = np.zeros(
                    (batch_size, config.MAX_GT_INSTANCES, 4), dtype=np.int32)
                batch_gt_boxes_right = np.zeros(
                    (batch_size, config.MAX_GT_INSTANCES, 4), dtype=np.int32)

            # If more instances than fits in the array, sub-sample from them.
            # left and right must match
            assert(gt_boxes_left.shape[0]== gt_boxes_right.shape[0])
            if gt_boxes_left.shape[0] > config.MAX_GT_INSTANCES:
                ids = np.random.choice(
                    np.arange(gt_boxes_left.shape[0]), config.MAX_GT_INSTANCES, replace=False)
                gt_class_ids = gt_class_ids[ids]
                gt_boxes_left = gt_boxes_left[ids]
                gt_boxes_right = gt_boxes_right[ids]


            # Add to batch
            batch_images_left[b] = mold_image(image_left.astype(np.float32), config)
            batch_images_right[b] = mold_image(image_right.astype(np.float32), config)
            batch_image_meta[b] = image_meta
            batch_srpn_match[b] = srpn_match[:, np.newaxis]
            batch_srpn_bboxes[b] = srpn_bbox
            batch_gt_class_ids[b, :gt_class_ids.shape[0]] = gt_class_ids
            batch_gt_boxes_left[b, :gt_boxes_left.shape[0]] = gt_boxes_left
            batch_gt_boxes_right[b, :gt_boxes_right.shape[0]] = gt_boxes_right
            b += 1

            # Batch full?
            if b >= batch_size:
                inputs = [batch_images_left,batch_images_right, batch_image_meta, batch_srpn_match, batch_srpn_bboxes,
                          batch_gt_class_ids, batch_gt_boxes_left, batch_gt_boxes_right]
                outputs = []

                yield inputs, outputs

                # start a new batch
                b = 0
        except (GeneratorExit, KeyboardInterrupt):
            raise
        except:
            # Log it and skip the image
            logging.exception("Error processing image {}".format(
                dataset.image_infos[image_id]))
            error_count += 1
            if error_count > 5:
                raise


class SRCNN():
    """
    以Mask RCNN为基础用Keras实现 Stereo RCNN论文网络
    """
    def __init__(self,mode,model_dir,config):
        assert mode in ['training', 'inference']
        self.mode = mode
        self.model_dir = model_dir
        self.config = config
        self.set_log_dir()
        self.keras_model = self.build()

    def get_anchors(self, image_shape):
        """Returns anchor pyramid for the given image size."""
        backbone_shapes = compute_backbone_shapes(self.config, image_shape)
        # Cache anchors and reuse if image shape is the same
        if not hasattr(self, "_anchor_cache"):
            self._anchor_cache = {}
        if not tuple(image_shape) in self._anchor_cache:
            # Generate Anchors
            a = Mutils.generate_pyramid_anchors(
                self.config.RPN_ANCHOR_SCALES,
                self.config.RPN_ANCHOR_RATIOS,
                backbone_shapes,
                self.config.BACKBONE_STRIDES,
                self.config.RPN_ANCHOR_STRIDE)
            # Keep a copy of the latest anchors in pixel coordinates because
            # it's used in inspect_model notebooks.
            # TODO: Remove this after the notebook are refactored to not use it
            self.anchors = a         # 作者保留下来的 anchors
            # Normalize coordinates
            self._anchor_cache[tuple(image_shape)] = Mutils.norm_boxes(a, image_shape[:2])
        return self._anchor_cache[tuple(image_shape)]

    def build(self):
        mode = self.mode
        config = self.config
        #构建SRCNN模型的函数
        assert mode in ['training','inference']
        heihgt,width = config.IMAGE_SHAPE[0:2]           # shape of the image is calculate in the config.py
        if heihgt / 2 ** 6 != int(heihgt / 2 ** 6) or width / 2 ** 6 != int(width / 2 ** 6):
            raise Exception("Image size must be dividable by 2 at least 6 times "
                            "to avoid fractions when downscaling and upscaling."
                            "For example, use 256, 320, 384, 448, 512, ... etc. ")

        input_left_image = KL.Input([heihgt,width,config.IMAGE_CHANNEL_COUNT],name='left_images_input')     # input of the left images
        input_right_image = KL.Input([heihgt,width,config.IMAGE_CHANNEL_COUNT],name='right_images_input')
        input_image_meta = KL.Input(shape=[config.IMAGE_META_SIZE], name="input_image_meta")

        # inputs
        if mode == "training":
            # SRPN GT
            input_srpn_match = KL.Input(
                shape=[None, 1], name="input_srpn_match", dtype=tf.int32)
            input_srpn_bbox_union = KL.Input(
                shape=[None, 6], name="input_srpn_bbox_union", dtype=tf.float32)

            # S-RCNN GT (class IDs, bounding boxes_left, bounding boxes_right, dimensions)
            # 1. GT Class IDs (zero padded)
            input_gt_class_ids = KL.Input(
                shape=[None], name="input_gt_class_ids", dtype=tf.int32)

            # 2. GT Boxes in pixels (zero padded)
            # [MAX_GT_INSTANCES, (y1, x1, y2, x2)] in image coordinates

            input_gt_boxes_left = KL.Input(
                shape=[None, 4], name="input_gt_boxes_left", dtype=tf.float32)
            input_gt_boxes_right = KL.Input(
                shape=[None, 4], name="input_gt_boxes_right", dtype=tf.float32)

            # Normalize coordinates
            gt_boxes_left  = KL.Lambda(lambda x: norm_boxes_graph(
                x, K.shape(input_left_image)[1:3]), name ='gt_boxes_left')(input_gt_boxes_left)
            gt_boxes_right = KL.Lambda(lambda x: norm_boxes_graph(
                x, K.shape(input_right_image)[1:3]), name='gt_boxes_right')(input_gt_boxes_right)



        # build a shared resnet101
        shared_resnet = build_resnet(config.BACKBONE,name='shared',stage5=True, train_bn=config.TRAIN_BN)
        C_left = shared_resnet(input_left_image)[1:5]
        C_right = shared_resnet(input_right_image)[1:5]
        C_union=[]
        # mearge the chanels of C_left and C_right   [5,H,W,C]
        # about the shape of te C_shared  the width and height is as well as C_left and C_right but double the chanels
        for C_1,C_2 in list(zip(C_left,C_right)):
            C_u = KL.Concatenate(axis=-1)([C_1,C_2])
            C_union.append(C_u)
        # fpn net work

        # fpn for left_right_union
        C_channels_union = [ C.get_shape().as_list()[-1] for C in C_union ]
        shared_fpn_feature_maps = build_fpn_feature_map_model( C_channels_union,config,'shared')
        rpn_feature_maps_shared = shared_fpn_feature_maps(C_union)


        #fpn for left and right ,shared
        C_channels_left = [C.get_shape().as_list()[-1] for C in C_left]
        left_fpn_feature_maps = build_fpn_feature_map_model(C_channels_left , config,'left')
        srpn_feature_maps_left = left_fpn_feature_maps(C_left)[0:4]   # get P2,P3,P4,P5
        srpn_feature_maps_right = left_fpn_feature_maps(C_right)[0:4]

        # S-RPN network
        if mode == 'training':
            anchors = self.get_anchors(config.IMAGE_SHAPE)
            # A hack to get around Keras's bad support for constants
            # Duplicate across the batch dimension because Keras requires it
            # TODO: can this be optimized to avoid duplicating the anchors?
            anchors = np.broadcast_to(anchors, (config.BATCH_SIZE,) + anchors.shape)
            anchors = KL.Lambda(lambda x: tf.Variable(anchors), name="anchors_left")(input_left_image)  # 将anchors作为变量？

        srpn_model = build_srpn_model(config.RPN_ANCHOR_STRIDE, len(config.RPN_ANCHOR_RATIOS),config.TOP_DOWN_PYRAMID_SIZE)
        srpn_outputs=[]
        # applay srpn_model to fpn
        '''
        srpn_outputs = [
        [srpn_class_logist,srpn_class_prob,srpn_bbox],            #where the srpn_class_logist: [batch_size,anchors,2] ... [batch_size,anchors,2], [anchors,6]
        [ ...],
        ]
        '''

        for P in rpn_feature_maps_shared:
            #P =KL.Lambda(lambda x:x )(P)
            srpn_outputs.append(srpn_model(P))
        # convert the shape to the [ [srpn_class_logist1, srpn_class_logist12 ], [srpn_class_prob1, srpn_class_prob2]..]
        output_name = ['srpn_class_logist','srpn_class_prob','srpn_bbox']

        srpn_outputs = list(zip(*srpn_outputs))                             #convert to [ [a1,a2,a3],[b1,b2,b3]]

        srpn_outputs = [
            KL.Concatenate(axis=1,name=name)(list(output))                        # concatenat the a1 a2 a3
            for output, name in zip(srpn_outputs,output_name)   #[ ( [a1,a2,a3],name), .....]
        ]

        srpn_class_logisit = srpn_outputs[0]
        srpn_class_prob =srpn_outputs[1]

        srpn_bbox = srpn_outputs[2]

        # now the srpn_outputs is
        '''
        [
         [anchors,2]         # note that the anchors is the num of all anchors(after aplay srpn to fpn feature map )
         [anchors,2]
         [ahchors,6]
        ]
        '''
        spn_class_logisit, spn_class_prob, spn_bbox = srpn_outputs

        # SRPN Layer appaly nms and other functions to get rois  get rois num
        proposal_count = config.POST_NMS_ROIS_TRAINING if mode == "training"\
            else config.POST_NMS_ROIS_INFERENCE

        # get the rois  srpn_rois_left:[ rois,(y1,x1,y2,x2)] in  normalized coordinates [0,1]
        srpn_rois_left,srpn_rois_right = SProposalLayer(  # shape  [batch, rois, (y1, x1, y2, x2)]   rois:经过非极大值抑制之后的roi, here the y of left and right should be the same
            proposal_count=proposal_count,
            nms_threshold=config.RPN_NMS_THRESHOLD,
            name="SRCNN_ROI_Layer",
            config=config)([srpn_class_prob, srpn_bbox, anchors])


        if mode=='training':
            # get target
            target_rois_left,target_rois_right, target_gt_class_ids, target_bbox_union=\
                DetectionTargetLayer(self.config,name='srcnn_detection_layer')([srpn_rois_left,srpn_rois_right,input_gt_class_ids,gt_boxes_left,gt_boxes_right])


            # network headers
            srcnn_class_logist,srcnn_class_prob,srcnn_bbox = \
                fpn_classifier_graph([target_rois_left,target_rois_right],[srpn_feature_maps_left,srpn_feature_maps_right],input_image_meta,
                                     config.POOL_SIZE, config.NUM_CLASSES,
                                     train_bn=config.TRAIN_BN,
                                     fc_layers_size=config.FPN_CLASSIF_FC_LAYERS_SIZE)

            # TODO: clean up (use tf.identify if necessary)
            target_rois_left = KL.Lambda(lambda x: x * 1, name="output_rois_left")( target_rois_left)
            targte_rois_right = KL.Lambda(lambda x: x * 1, name="output_rois_right")( target_rois_right)

            #loss
            # Losses
            srpn_class_loss = KL.Lambda(lambda x: srpn_class_loss_graph(*x), name="srpn_class_loss")(
                [input_srpn_match,  srpn_class_logisit])
            srpn_bbox_loss = KL.Lambda(lambda x: srpn_bbox_loss_graph(config, *x), name="srpn_bbox_loss")(
                [input_srpn_bbox_union, input_srpn_match, srpn_bbox])
            srcnn_class_loss = KL.Lambda(lambda x: srcnn_class_loss_graph(*x), name="srcnn_class_loss")(
                [ target_gt_class_ids, srcnn_class_logist])
            srcnn_bbox_loss = KL.Lambda(lambda x: srcnn_bbox_loss_graph(*x), name="srcnn_bbox_loss")(
                [target_bbox_union, target_gt_class_ids, srcnn_bbox])

            # Model
            inputs = [input_left_image, input_right_image, input_image_meta,
                      input_srpn_match, input_srpn_bbox_union, input_gt_class_ids, input_gt_boxes_left, input_gt_boxes_right]

            outputs = [srpn_class_logisit, srpn_class_prob, srpn_bbox,
                       srcnn_class_logist, srcnn_class_prob, srcnn_bbox,
                       srpn_rois_left, srpn_rois_right,  target_rois_left,targte_rois_right,
                       srpn_class_loss, srpn_bbox_loss,  srcnn_class_loss,  srcnn_bbox_loss ]

            model = KM.Model(inputs, outputs, name='srcnn')

            return model


    def find_last(self):
        """Finds the last checkpoint file of the last trained model in the
        model directory.
        Returns:
            The path of the last checkpoint file
        """
        # Get directory names. Each directory corresponds to a model
        dir_names = next(os.walk(self.model_dir))[1]
        key = self.config.NAME.lower()
        dir_names = filter(lambda f: f.startswith(key), dir_names)
        dir_names = sorted(dir_names)
        if not dir_names:
            import errno
            raise FileNotFoundError(
                errno.ENOENT,
                "Could not find model directory under {}".format(self.model_dir))
        # Pick last directory
        dir_name = os.path.join(self.model_dir, dir_names[-1])
        # Find the last checkpoint
        checkpoints = next(os.walk(dir_name))[2]
        checkpoints = filter(lambda f: f.startswith("srcnn"), checkpoints)
        checkpoints = sorted(checkpoints)
        if not checkpoints:
            import errno
            raise FileNotFoundError(
                errno.ENOENT, "Could not find weight files in {}".format(dir_name))
        checkpoint = os.path.join(dir_name, checkpoints[-1])
        return checkpoint

    def load_weights(self, filepath, by_name=False, exclude=None):
        """Modified version of the corresponding Keras function with
        the addition of multi-GPU support and the ability to exclude
        some layers from loading.
        exclude: list of layer names to exclude
        """
        import h5py
        # Conditional import to support versions of Keras before 2.2
        # TODO: remove in about 6 months (end of 2018)
        try:
            from keras.engine import saving
        except ImportError:
            # Keras before 2.2 used the 'topology' namespace.
            from keras.engine import topology as saving

        if exclude:
            by_name = True

        if h5py is None:
            raise ImportError('`load_weights    ` requires h5py.')
        f = h5py.File(filepath, mode='r')
        if 'layer_names' not in f.attrs and 'model_weights' in f:
            f = f['model_weights']

        # In multi-GPU training, we wrap the model. Get layers
        # of the inner model because they have the weights.
        keras_model = self.keras_model
        layers = keras_model.inner_model.layers if hasattr(keras_model, "inner_model")\
            else keras_model.layers

        # Exclude some layers
        if exclude:
            layers = filter(lambda l: l.name not in exclude, layers)

        if by_name:
            saving.load_weights_from_hdf5_group_by_name(f, layers)
        else:
            saving.load_weights_from_hdf5_group(f, layers)
        if hasattr(f, 'close'):
            f.close()

        # Update the log directory
        self.set_log_dir(filepath)

    def compile(self, learning_rate, momentum):
        """Gets the model ready for training. Adds losses, regularization, and
        metrics. Then calls the Keras compile() function.
        """
        # Optimizer object
        optimizer = keras.optimizers.SGD(
            lr=learning_rate, momentum=momentum,
            clipnorm=self.config.GRADIENT_CLIP_NORM)
        # Add Losses
        # First, clear previously set losses to avoid duplication
        self.keras_model._losses = []
        self.keras_model._per_input_losses = {}
        loss_names = [
            "srpn_class_loss", "srpn_bbox_loss",
            "srcnn_class_loss", "srcnn_bbox_loss"]
        for name in loss_names:
            layer = self.keras_model.get_layer(name)
            if layer.output in self.keras_model.losses:
                continue
            loss = (
                    tf.reduce_mean(layer.output, keepdims=True)
                    * self.config.LOSS_WEIGHTS.get(name, 1.))
            self.keras_model.add_loss(loss)

        # Add L2 Regularization
        # Skip gamma and beta weights of batch normalization layers.
        reg_losses = [
            keras.regularizers.l2(self.config.WEIGHT_DECAY)(w) / tf.cast(tf.size(w), tf.float32)       # 考虑了权重矩阵的大小
            for w in self.keras_model.trainable_weights
            if 'gamma' not in w.name and 'beta' not in w.name]
        self.keras_model.add_loss(tf.add_n(reg_losses))

        # Compile
        self.keras_model.compile(
            optimizer=optimizer,
            loss=[None] * len(self.keras_model.outputs))

        # Add metrics for losses
        for name in loss_names:
            if name in self.keras_model.metrics_names:
                continue
            layer = self.keras_model.get_layer(name)
            self.keras_model.metrics_names.append(name)
            loss = (
                    tf.reduce_mean(layer.output, keepdims=True)
                    * self.config.LOSS_WEIGHTS.get(name, 1.))
            self.keras_model.metrics_tensors.append(loss)

    def set_trainable(self, layer_regex, keras_model=None, indent=0, verbose=1):
        """Sets model layers as trainable if their names match
        the given regular expression.
        """
        # Print message on the first call (but not on recursive calls)
        if verbose > 0 and keras_model is None:
            log("Selecting layers to train")

        keras_model = keras_model or self.keras_model

        # In multi-GPU training, we wrap the model. Get layers
        # of the inner model because they have the weights.
        layers = keras_model.inner_model.layers if hasattr(keras_model, "inner_model") \
            else keras_model.layers

        for layer in layers:
            # Is the layer a model?
            if layer.__class__.__name__ == 'Model':
                print("In model: ", layer.name)
                self.set_trainable(
                    layer_regex, keras_model=layer, indent=indent + 4)
                continue

            if not layer.weights:
                continue
            # Is it trainable?
            trainable = bool(re.fullmatch(layer_regex, layer.name))
            # Update layer. If layer is a container, update inner layer.
            if layer.__class__.__name__ == 'TimeDistributed':
                layer.layer.trainable = trainable
            else:
                layer.trainable = trainable
            # Print trainable layer names
            if trainable and verbose > 0:
                log("{}{:20}   ({})".format(" " * indent, layer.name,
                                            layer.__class__.__name__))

    def set_log_dir(self, model_path=None):
        """Sets the model log directory and epoch counter.

        model_path: If None, or a format different from what this code uses
            then set a new log directory and start epochs from 0. Otherwise,
            extract the log directory and the epoch counter from the file
            name.
        """
        # Set date and epoch counter as if starting a new model
        self.epoch = 0
        now = datetime.datetime.now()

        # If we have a model path with date and epochs use them
        if model_path:
            # Continue from we left of. Get epoch and date from the file name
            # A sample model path might look like:
            # \path\to\logs\coco20171029T2315\mask_rcnn_coco_0001.h5 (Windows)
            # /path/to/logs/coco20171029T2315/mask_rcnn_coco_0001.h5 (Linux)
            regex = r".*[/\\][\w-]+(\d{4})(\d{2})(\d{2})T(\d{2})(\d{2})[/\\]srcnn\_[\w-]+(\d{4})\.h5"
            m = re.match(regex, model_path)
            if m:
                now = datetime.datetime(int(m.group(1)), int(m.group(2)), int(m.group(3)),
                                        int(m.group(4)), int(m.group(5)))
                # Epoch number in file is 1-based, and in Keras code it's 0-based.
                # So, adjust for that then increment by one to start from the next epoch
                self.epoch = int(m.group(6)) - 1 + 1
                print('Re-starting from epoch %d' % self.epoch)

        # Directory for training logs
        self.log_dir = os.path.join(self.model_dir, "{}{:%Y%m%dT%H%M}".format(
            self.config.NAME.lower(), now))

        # Path to save after each epoch. Include placeholders that get filled by Keras.
        self.checkpoint_path = os.path.join(self.log_dir, "srcnn_{}_*epoch*.h5".format(
            self.config.NAME.lower()))
        self.checkpoint_path = self.checkpoint_path.replace(
            "*epoch*", "{epoch:04d}")

    def train(self, train_dataset, val_dataset, learning_rate, epochs, layers, custom_callbacks=None):
        """Train the model.
        train_dataset, val_dataset: Training and validation Dataset objects.
        learning_rate: The learning rate to train with
        epochs: Number of training epochs. Note that previous training epochs
                are considered to be done alreay, so this actually determines
                the epochs to train in total rather than in this particaular
                call.
        layers: Allows selecting wich layers to train. It can be:
            - A regular expression to match layer names to train
            - One of these predefined values:
              heads: The RPN, classifier and mask heads of the network
              all: All the layers
              3+: Train Resnet stage 3 and up
              4+: Train Resnet stage 4 and up
              5+: Train Resnet stage 5 and up
        custom_callbacks: Optional. Add custom callbacks to be called
            with the keras fit_generator method. Must be list of type keras.callbacks..
        """
        assert self.mode == "training", "Create model in training mode."

        # Pre-defined layer regular expressions
        layer_regex = {
            # all layers but the backbone
            "heads": r"(srcnn\_.*)|(srpn\_.*)|(fpn\_.*)",
            # From a specific Resnet stage and up
            "3+": r"(res3.*)|(bn3.*)|(res4.*)|(bn4.*)|(res5.*)|(bn5.*)|(srcnn\_.*)|(srpn\_.*)|(fpn\_.*)",
            "4+": r"(res4.*)|(bn4.*)|(res5.*)|(bn5.*)|(srcnn\_.*)|(srpn\_.*)|(fpn\_.*)",
            "5+": r"(res5.*)|(bn5.*)|(srcnn\_.*)|(srpn\_.*)|(fpn\_.*)",
            # All layers
            "all": ".*",
        }
        if layers in layer_regex.keys():
            layers = layer_regex[layers]

        # Data generators

        train_generator = data_generator(train_dataset, self.config, shuffle=True,
                                         batch_size=self.config.BATCH_SIZE,
                                         )
        val_generator = data_generator(val_dataset, self.config, shuffle=True,
                                         batch_size=self.config.BATCH_SIZE,
                                         )
        # Create log_dir if it does not exist
        if not os.path.exists(self.log_dir):
            os.makedirs(self.log_dir)

        # Callbacks
        callbacks = [
            keras.callbacks.TensorBoard(log_dir=self.log_dir,
                                        histogram_freq=0, write_graph=True, write_images=False),
            keras.callbacks.ModelCheckpoint(self.checkpoint_path,
                                            verbose=0, save_weights_only=True),
        ]

        # Add custom callbacks to the list
        if custom_callbacks:
            callbacks += custom_callbacks

        # Train
        log("\nStarting at epoch {}. LR={}\n".format(self.epoch, learning_rate))
        log("Checkpoint Path: {}".format(self.checkpoint_path))
        self.set_trainable(layers)
        self.compile(learning_rate, self.config.LEARNING_MOMENTUM)

        # Work-around for Windows: Keras fails on Windows when using
        # multiprocessing workers. See discussion here:
        # https://github.com/matterport/Mask_RCNN/issues/13#issuecomment-353124009
        KM.Model.fit_generator
        self.keras_model.fit_generator(
            train_generator,
            initial_epoch=self.epoch,
            epochs=epochs,
            steps_per_epoch=self.config.STEPS_PER_EPOCH,
            callbacks=callbacks,
            validation_data=val_generator,
            validation_steps=self.config.VALIDATION_STEPS,
        )
        self.epoch = max(self.epoch, epochs)

# 测试代码
# config = sconfig.Config()
# srcnn =SRCNN('training','',config)
# srcnn =srcnn.build()
# srcnn.summary()
# plot_model(srcnn,'srcnn_train.png',True)
