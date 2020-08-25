# -*- coding: utf-8 -*-
"""
@author: Terada
"""
from keras.models import Sequential, Model
from keras.layers import Dense, MaxPooling2D, Flatten, Dropout
from keras.layers import Conv2D, BatchNormalization, ZeroPadding2D, MaxPool2D
from keras.layers import Input, Convolution2D, AveragePooling2D, merge, Reshape, Activation, concatenate
from keras.regularizers import l2
#from keras.engine.topology import Container

from src.keras_networks import lenet, malti_net, alexnet, net7
from src.keras_vgg import vgg16, vgg16mtl, vgg16_ex, vgg19_ex
from src.keras_resnet import ResnetBuilder, ResnetBuilderMTL
from src.keras_senet import SEResNeXt

def select(_network, input_size, output_size = None):
    if _network == 'multi':
        return malti_net(input_size)
    if _network == 'net7':
        return net7(input_size)
    if _network == 'lenet':
        return lenet(input_size)
    if _network == 'vgg16':
        return vgg16(input_size)
    if _network == 'vgg16mtl':
        return vgg16mtl(input_size)
    if _network == 'vgg19':
        return vgg19_ex(input_size)
    if _network == 'res18':
        resbuild = ResnetBuilder()
        return resbuild.build_resnet_18((1, input_size[0], input_size[1]), output_size)
    if _network == 'res34':
        resbuild = ResnetBuilder()
        return resbuild.build_resnet_34((1, input_size[0], input_size[1]), output_size)
    if _network == 'res50':
        resbuild = ResnetBuilder()
        return resbuild.build_resnet_50((1, input_size[0], input_size[1]), output_size)
    if _network == 'res101':
        resbuild = ResnetBuilder()
        return resbuild.build_resnet_101((1, input_size[0], input_size[1]), output_size)
    if _network == 'res152':
        resbuild = ResnetBuilder()
        return resbuild.build_resnet_152((1, input_size[0], input_size[1]), output_size)
    if _network == 'res18mtl':
        resbuild = ResnetBuilderMTL()
        return resbuild.build_resnet_18((1, input_size[0], input_size[1]), output_size)
    if _network == 'res34mtl':
        resbuild = ResnetBuilderMTL()
        return resbuild.build_resnet_34((1, input_size[0], input_size[1]), output_size)
    if _network == 'senet':
        inputs = (input_size[0], input_size[1], 1)
        return SEResNeXt(inputs, output_size, num_split=2, num_block=1, mtl=False).model
    if _network == 'senet_mtl':
        inputs = (input_size[0], input_size[1], 1)
        return SEResNeXt(inputs, output_size, num_split=2, num_block=1, mtl=True).model
