import numpy as np
import os.path as osp
from collections import namedtuple
from src.datasets import IGNORE_LABEL as IGNORE


########################################################################
#                         Download information                         #
########################################################################

FORM_URL = 'https://docs.google.com/forms/d/e/1FAIpQLSefhHMMvN0Uwjnj_vWQgYSvtFOtaoGFWsTIcRuBTnP09NHR7A/viewform?fbzx=5530674395784263977'

# DALES in LAS format
LAS_TAR_NAME = 'dales_semantic_segmentation_las.tar.gz'
LAS_UNTAR_NAME = "dales_las"

# DALES in PLY format
PLY_TAR_NAME = 'dales_semantic_segmentation_ply.tar.gz'
PLY_UNTAR_NAME = "dales_ply"

# DALES in PLY, only version with intensity and instance labels
OBJECTS_TAR_NAME = 'DALESObjects.tar.gz'
OBJECTS_UNTAR_NAME = "DALESObjects"


########################################################################
#                              Data splits                             #
########################################################################

# The validation set was arbitrarily chosen as the x last train tiles:
TILES = {
    'train': [
        '5080_54435_new',
        '5190_54400_new',
        '5105_54460_new',
        '5130_54355_new',
        '5165_54395_new',
        '5185_54390_new',
        '5180_54435_new',
        '5085_54320_new',
        '5100_54495_new',
        '5110_54320_new',
        '5140_54445_new',
        '5105_54405_new',
        '5185_54485_new',
        '5165_54390_new',
        '5145_54460_new',
        '5110_54460_new',
        '5180_54485_new',
        '5150_54340_new',
        '5145_54405_new',
        '5145_54470_new',
        '5160_54330_new',
        '5135_54495_new',
        '5145_54480_new',
        '5115_54480_new',
        '5110_54495_new',
        '5095_54440_new'],

    'val': [
        '5145_54340_new',
        '5095_54455_new',
        '5110_54475_new'],

    'test': [
        '5080_54470_new',
        '5100_54440_new',
        '5140_54390_new',
        '5080_54400_new',
        '5155_54335_new',
        '5150_54325_new',
        '5120_54445_new',
        '5135_54435_new',
        '5175_54395_new',
        '5100_54490_new',
        '5135_54430_new']}


########################################################################
#                                Labels                                #
########################################################################

DALES_NUM_CLASSES = 8

ID2TRAINID = np.asarray([8, 0, 1, 2, 3, 4, 5, 6, 7])

CLASS_NAMES = [
    'Ground',
    'Vegetation',
    'Cars',
    'Trucks',
    'Power lines',
    'Fences',
    'Poles',
    'Buildings',
    'Unknown']

CLASS_COLORS = np.asarray([
    [243, 214, 171], # sunset
    [ 70, 115,  66], # fern green
    [233,  50, 239],
    [243, 238,   0],
    [190, 153, 153],
    [  0, 233,  11],
    [239, 114,   0],
    [214,   66,  54], # vermillon
    [  0,   8, 116]])
