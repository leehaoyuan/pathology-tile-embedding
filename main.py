# -*- coding: utf-8 -*-
"""
Created on Tue Nov  2 15:08:11 2021

@author: lhy
"""
import os
os.environ['KMP_DUPLICATE_LIB_OK']='True'
import nest_asyncio
nest_asyncio.apply()
from pathml.core import HESlide
import argparse
from preprocess import preprocess
from napari_main import napari_main
parser = argparse.ArgumentParser()
parser.add_argument('--image_path', type=str, help='path of the input whole slide image')
parser.add_argument('--model_path',type=str,help='path of the model to generate embedding')
parser.add_argument('--prefix',type=str,help='path to store media result')
parser.add_argument('--tile_size',type=int,help='tile size')
parser.add_argument('--level',type=int,help='perform embedding at which level of image. the higher the level, the quicker the preprocess')
parser.add_argument('--xml_path',type=str,help='the path of label xml file')
parser.add_argument('--tissue_ratio',type=float,help='filter out tiles whose tissue ratio is below this value')
args = parser.parse_args()

###preprocess
wsi = HESlide(args.image_path, name = "example")
if not os.path.exists(args.prefix):
    os.makedirs(args.prefix)
tile_ma,coords,level_app,tile_orig,labels=preprocess(args,wsi)
napari_main(tile_ma,coords,level_app,tile_orig,labels,wsi,args)
