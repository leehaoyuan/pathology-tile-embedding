# -*- coding: utf-8 -*-
"""
Created on Tue Dec 21 18:51:14 2021

@author: lhy
"""
import os
os.environ['KMP_DUPLICATE_LIB_OK']='True'
import numpy as np
import pickle
import nest_asyncio
nest_asyncio.apply()
import torch
import torchvision
from pathml.core import HESlide
from pathml.preprocessing import TissueDetectionHE
import argparse
import time
from tqdm import tqdm
import xml.etree.ElementTree as ET
from shapely.strtree import STRtree
from shapely.geometry import MultiPolygon
from shapely.geometry import Polygon
def get_coords(tree):
    """
    input: 
        tree: xml ElementTree object
    output: 
        polys: dictionary. key = annotation 0, 1, etc. value = polygon coordinates 
        groups: dictionary. key = annotation 0, 1, etc. value = partofgroup: _0 or _2
    """
    
    
    root = tree.getroot() #root element
    
    polys = {} # key = annotation 0, 1, etc. value = polygon coordinates 
    groups = {} # key = annotation 0, 1, etc. value = partofgroup: _0 or _2

    for item in root.findall('Annotations/Annotation'):

            # initialize for each Annotation group
            x_co = []
            y_co = []
            name = item.get('Name') #retuns: Annotation 0,1,2,...
            group = item.get('PartOfGroup') #returns: _0,_2

            for child in item.findall('Coordinates/Coordinate'):

                x = child.get('X') # returns: 0,1,2,
                x_co.append(x)
                y = child.get('Y') # returns: 0,1,2,
                y_co.append(y)

            # make into list of coords
            x_co = np.array(x_co,dtype=np.float32)
            y_co = np.array(y_co,dtype=np.float32)
            coo = list(zip(y_co,x_co))

            # save into dict
            polys[name] = coo
            groups[name] = group
    
    return polys, groups
def load_model_weights(model, weights):
    model_dict = model.state_dict()
    weights = {k: v for k, v in weights.items() if k in model_dict}
    if weights == {}:
        print('No weight could be loaded..')
    model_dict.update(weights)
    model.load_state_dict(model_dict)
    return model
def find_exist_file(args):
    similar=False
    load_name=None
    for i in os.listdir(args.prefix):
        if i.endswith('.pickle'):
            load_name=os.path.join(args.prefix,i)
            with open(load_name,'rb') as file:
                loaded_file=pickle.load(file)
            similar=True
            if loaded_file['args'].tile_size!=args.tile_size:
                similar=False
                continue
            if loaded_file['args'].tissue_ratio!=args.tissue_ratio:
                similar=False
                continue
            media0=loaded_file['args'].image_path.split('/')[-1]
            media1=args.image_path.split('/')[-1]
            if media0!=media1:
                similar=False
                continue
            media0=loaded_file['args'].model_path.split('/')[-1]
            media1=args.model_path.split('/')[-1]
            if media0!=media1:
                similar=False
                continue
            media0=loaded_file['args'].level
            media1=args.level
            if media0!=media1:
                similar=False
                continue
            if similar:
                break
    return similar,load_name
def build_model(args):
    model1 = torchvision.models.__dict__['resnet18'](pretrained=False)
    state = torch.load(args.model_path, map_location=torch.device('cpu'))
    state_dict = state['state_dict']
    for key in list(state_dict.keys()):
        state_dict[key.replace('model.', '').replace('resnet.', '')] = state_dict.pop(key)

    model1 = load_model_weights(model1, state_dict)
    model1.fc = torch.nn.Sequential()
    model1.eval()
    model_orig = torchvision.models.__dict__['resnet18'](pretrained=True)
    model_orig.fc = torch.nn.Sequential()
    model_orig.eval()
    return model1,model_orig
def get_features_eval(t,model):
    """
    t: single tile in SlideData object, eg wsi.tiles[0] 
    model: model loaded
    
    """
    # transpose to 3,32,32, then tensor, 
    test = torch.tensor(t.transpose([2,1,0]))

    # reshape image to 4D 
    # PyTorch expects a 4-dimensional input, the first dimension being the number of samples
    test = test.unsqueeze(0)

    # transform to float, otherwise there's mistake on double/float
    # shape = 1,3,32,32
    test = test.float() 

    # pooled features
    o_pooled = model(test)

    # flatten the pooled vector: 1 X D
    o_pooled_flat = o_pooled.reshape(1,-1).detach().numpy()

    return o_pooled_flat
def tile_intersect_polys(query_geom,polygons):
    """
    This function checks if a tile overlaps with a group of polygons.
    query_geom: a polygon object(tile). e.g. Polygon([(30000, 30000), (40000, 30000), (40000, 40000), (30000, 40000)])
    polygons: a list of polygons(tumor). e.g. [Polygon(polys['Annotation 0']),...]
    """
    # create a shapely Tree object, but with query_geom
    s = STRtree([query_geom])
    # Returns a list of all geometries in the s(strtree) whose extents intersect the extent of multiple polygons
    multi = MultiPolygon(polygons)
    result = s.query(multi)
    # does my query_geom intersect with the polygon? True/False
    return query_geom in result
def get_features_wsi_eval(wsi,model,level,tissue_mask,ratio,tile_size,polygons1=None,tissue_ratio=0.5):
    i=0
    j=0
    coords=[]
    embedding=[]
    labels=[]
    ratio1=ratio*int(2**level)
    tile_size1=tile_size*int(2**level)
    for i in tqdm(range(0,wsi.slide.get_image_shape(0)[0],tile_size1)):
        while j<wsi.slide.get_image_shape(0)[1]:
            if np.sum(tissue_mask[int(i/ratio1):int((i+tile_size1)/ratio1),int(j/ratio1):int((j+tile_size1)/ratio1)]==127)>=int(0.5*tile_size1*tile_size1/ratio1/ratio1):
                media = wsi.slide.extract_region(location = (i, j), level = level, size =(tile_size,tile_size))
                o_pooled_flat=get_features_eval(media,model)
                embedding.append(o_pooled_flat[0])
                coords.append([i,j])
                if polygons1 is not None:
                    media_coord=[(i,j),(i+tile_size1,j),(i+tile_size1,j+tile_size1),(i,j+tile_size1)]
                    query_p = Polygon(media_coord)
                    labels.append(tile_intersect_polys(query_p,polygons1))
            j+=tile_size1
        j=0
        i+=tile_size1
    return coords,embedding,labels
def preprocess(args,wsi):
    similar,load_name=find_exist_file(args)
    if not similar:
        print("no available preprocessed data, run preprocessing from begining")
        polygons=None
        if args.xml_path is not None:
            tree = ET.parse(args.xml_path)
            polys, groups = get_coords(tree) 
            polygons = [Polygon(polys[k]) for k in polys.keys()]
        level_app=args.level
        if level_app is not None:
            level_app=min(level_app,wsi.slide.slide.level_count-1)
        if level_app is None:
            level_app=0
        level_i=min(wsi.slide.slide.level_count-1,level_app+1)
        high_level = wsi.slide.extract_region(location = (0, 0), level = level_i, size = wsi.slide.get_image_shape(level = level_i))
        tissue_detector = TissueDetectionHE()
        tissue_mask_high_level = tissue_detector.F(high_level)
        ratio=int(2**(level_i-level_app))
        ###load Model
        model1,model_orig=build_model(args)
        ###Generate Embedding
        with torch.no_grad():
            coords,tile_ma,labels = get_features_wsi_eval(wsi,model1,level_app,tissue_mask_high_level,ratio,args.tile_size,polygons,args.tissue_ratio)
            _,tile_orig,_ = get_features_wsi_eval(wsi,model_orig,level_app,tissue_mask_high_level,ratio,args.tile_size)
        if len(labels)==0:
            labels=None
        diction={'emb':tile_ma,'coords':coords,'args':args,'level':level_app,'emb_orig':tile_orig,'labels':labels}
        now=str(int(time.time()))
        file_name=args.image_path.split('/')[-1].split('.')[0]+now+'.pickle'
        with open(os.path.join(args.prefix,file_name),'wb') as file:
            pickle.dump(diction,file)
    else:
        print('load preprocessed data from '+load_name)
        with open(load_name,'rb') as file:
            media=pickle.load(file)
            tile_ma=media["emb"]
            coords=media["coords"]
            level_app=media['level']
            tile_orig=media['emb_orig']
            labels=media['labels']
    return tile_ma,coords,level_app,tile_orig,labels