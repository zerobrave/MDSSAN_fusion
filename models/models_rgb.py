# -*- coding：UTF-8 -*-
'''
@File ：models_rgb.py
@Author : Zerbo
@Date : 2023/11/14 15:42
'''
from models.MSDCNN import *
from models.SSRNET import *
from models.TFNET import *
from models.MDSSAN import *
from models.HSR import *

MODELS_RGB = {  "MSD": MSDCNN,
            "SSR": SSRNET,
            "TF": TFNet,
            "HSR": HSRNet,
            "MDSSAN"：MDSSAN
    }