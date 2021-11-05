# -*- coding: utf-8 -*-
"""
Created on Tue Nov  2 15:08:11 2021

@author: lhy
"""
import os
os.environ['KMP_DUPLICATE_LIB_OK']='True'
import numpy as np
import napari
import copy
import pickle
import nest_asyncio
nest_asyncio.apply()
import torch
import torchvision
import timm
from pathml.core import HESlide
from sklearn.decomposition import PCA
from pathml.preprocessing import Pipeline, BoxBlur, TissueDetectionHE
import csv
from sklearn.cluster import KMeans
from magicgui import magicgui
import argparse
import time
parser = argparse.ArgumentParser()
parser.add_argument('--image_path', type=str, help='path of the input whole slide image')
parser.add_argument('--model_path',type=str,help='path of the model to generate embedding')
parser.add_argument('--prefix',type=str,help='path to store media result')
parser.add_argument('--tile_size',type=int,help='tile size')
args = parser.parse_args()
def generate_pca_point(row,matrix,point_size):
    #print(matrix[0:10])
    matrix_min=np.min(matrix,axis=0)
    matrix=matrix-np.min(matrix,axis=0)
    matrix_max=np.max(matrix,axis=0)
    col=int(row*matrix_max[1]/matrix_max[0])
    plot_matrix=np.ones((row,col,3),dtype='uint8')*255
    min_col=0.05*col
    min_row=0.05*row
    max_row=0.95*row
    max_col=0.95*col
    matrix[:,0]=min_row+matrix[:,0]*(max_row-min_row)/matrix_max[0]
    matrix[:,1]=min_col+matrix[:,1]*(max_col-min_col)/matrix_max[1]
    matrix=matrix.astype(int)
    width=max(int(min(row,col)*0.002),1)
    row_interval=int(row/4)
    col_interval=int(col/4)
    color=220
    for i in range(1,4):
        plot_matrix[i*row_interval:i*row_interval+width,:]=color
    for i in range(1,4):
        plot_matrix[:,i*col_interval:i*col_interval+width,:]=color
    return plot_matrix,matrix,min_col,min_row,max_row,max_col,matrix_min,matrix_max
def transform_origpos(x,y,row,col,orig_row,orig_col,plot_col):
    x1=x
    y1=y
    x1=int(x1*row/orig_row)
    y1=int(y1*col/orig_col)
    y1=y1+plot_col
    return x1,y1
def readincoord(file_path):
    coord=[]
    with open(file_path) as file:
        reader=csv.reader(file)
        for i in reader:
            coord.append(i)
    coord=np.array(coord,dtype=int)
    coord=[list(coord[i]) for i in range(coord.shape[0])]
    return coord
def gen_cluster_mask(coords,row,col,orig_row,orig_col,plot_col,tile_size):
    polys=[]
    for i in range(len(coords)):
        media_x,media_y=transform_origpos(coords[i][0],coords[i][1],row,col,orig_row,orig_col,plot_col)
        media1_x,media1_y=transform_origpos(coords[i][0]+tile_size,coords[i][1]+tile_size,row,col,orig_row,orig_col,plot_col)
        media=np.array([[media_x,media_y],[media_x,media1_y],[media1_x,media1_y],[media1_x,media_y]],dtype=int)
        polys.append(media)
    return polys
def gen_mask(coords,row,col,orig_row,orig_col,plot_col,tile_size,line_width):
    lines=[]
    ld=0.5*float(line_width)
    for i in range(len(coords)):
        media_x,media_y=transform_origpos(coords[i][0],coords[i][1],row,col,orig_row,orig_col,plot_col)
        media1_x,media1_y=transform_origpos(coords[i][0]+tile_size,coords[i][1]+tile_size,row,col,orig_row,orig_col,plot_col)
        media=np.array([[media_x,media_y-ld],[media_x,media1_y],[media1_x,media1_y],[media1_x,media_y],[media_x-ld,media_y]])
        lines.append(media)
    return lines
def load_model_weights(model, weights):
    model_dict = model.state_dict()
    weights = {k: v for k, v in weights.items() if k in model_dict}
    if weights == {}:
        print('No weight could be loaded..')
    model_dict.update(weights)
    model.load_state_dict(model_dict)
    return model
###preprocess
wsi = HESlide(args.image_path, name = "example")
if not os.path.exists(args.prefix):
    os.makedirs(args.prefix)
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
        if similar:
            break
if not similar:
    print("no available preprocessed data, run preprocessing from begining")
    pipeline = Pipeline([
        TissueDetectionHE(mask_name = "tissue", min_region_size=500,
                          threshold=30, outer_contours_only=True)
    ])
    wsi.run(pipeline, distributed=False,tile_size = args.tile_size)
    tissue_masks=[]
    for i in wsi.tiles:
        tissue_masks.append(int(np.sum(i.masks['tissue']!=0)))
    num_tiles = len(wsi.tiles)
    coords = []
    for i in range(num_tiles):
        coords.append(wsi.tiles[i].coords)
    ###load Model
    model1 = torchvision.models.__dict__['resnet18'](pretrained=False)
    state = torch.load(args.model_path, map_location=torch.device('cpu'))

    state_dict = state['state_dict']
    for key in list(state_dict.keys()):
        state_dict[key.replace('model.', '').replace('resnet.', '')] = state_dict.pop(key)

    model1 = load_model_weights(model1, state_dict)
    model1.fc = torch.nn.Sequential()
    ###Generate Embedding
    def get_features_eval(t,model):
        """
        t: single tile in SlideData object, eg wsi.tiles[0] 
        model: model loaded
        """
        # transpose to 3,32,32, then tensor, 
        test = torch.tensor(t.image.transpose([2,1,0]))

        # reshape image to 4D 
        # PyTorch expects a 4-dimensional input, the first dimension being the number of samples
        test = test.unsqueeze(0)

        # transform to float, otherwise there's mistake on double/float
        # shape = 1,3,32,32
        test = test.float() 

        # eval mode
        model.eval()

        # pooled features
        o_pooled = model(test)

        # flatten the pooled vector: 1 X D
        o_pooled_flat = o_pooled.reshape(1,-1).detach().numpy()

        return o_pooled_flat

    # make into a function
    def get_features_wsi_eval(wsi,model):
        '''
        wsi: SlideObject
        get_features: function above: Tile -> Tensor
        model: model loaded above
        '''
        features = []
        for ti in wsi.tiles:
            features.append(list(get_features_eval(ti,model)[0]))
        return features
    with torch.no_grad():
        tile_ma = get_features_wsi_eval(wsi,model1)
    assert len(coords)==len(tissue_masks)
    assert len(tile_ma)==len(tissue_masks)
    tile_ma=[tile_ma[i] for i in range(len(tile_ma)) if tissue_masks[i]>0]
    coords=[coords[i] for i in range((len(coords))) if tissue_masks[i]>0]
    diction={'emb':tile_ma,'coords':coords,'args':args}
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
n_clusters=5
tile_ma=np.array(tile_ma,dtype=float)
pca = PCA(n_components=2)
pca_matrix=pca.fit_transform(tile_ma)
kmean=KMeans(n_clusters=n_clusters, random_state=0).fit(tile_ma)
thumbnail = wsi.slide.get_thumbnail(size = (5000, 5000))
color_list=['green','blue','orange','black','purple','yellow','pink']
point_size=int(0.005*thumbnail.shape[0])
layer2_data,point_matrix,min_col,min_row,max_row,max_col,matrix_min,matrix_max=generate_pca_point(thumbnail.shape[0],pca_matrix,point_size)
line_width=8
polys=gen_cluster_mask(coords,thumbnail.shape[0],thumbnail.data.shape[1],wsi.shape[0],wsi.shape[1],layer2_data.shape[1],args.tile_size)
viewer = napari.Viewer()
p_color_list=[color_list[i] for i in kmean.labels_]
layer1=viewer.add_image(np.concatenate((layer2_data,thumbnail),axis=1),name='wsi')
layer2=viewer.add_points(point_matrix,size=point_size,properties={'coord_x':np.array(coords)[:,0],'coord_y':np.array(coords)[:,1]},edge_color=p_color_list,name='pca plot')
layer3=viewer.add_points(point_matrix[0:1],size=point_size*3,symbol='cross',opacity=0,edge_color='red',face_color='red',name='cross')
layer4=viewer.add_shapes(polys,properties={'coord_x':pca_matrix[:,0],'coord_y':pca_matrix[:,1]},edge_width=0,face_color=p_color_list,opacity=0.5,name='tile')
layer5=viewer.add_shapes(np.array([[1114,2997.5],[1114,3064],[1178,3064],[1178,3000],[1111.5,3000]]),edge_color='red',edge_width=line_width,shape_type='path',opacity=0.0,name='grid')
layer1.editable=False
layer3.editable=False
layer5.editable=False
def update_grid_position(event):
    if len(event.source.selected_data)>0:
        media=list(event.source.selected_data)
        coords=[layer4.data[i] for i in media]
        for i in range(len(coords)):
            coords[i]=np.concatenate((coords[i],coords[i][0:1]),axis=0)
            coords[i][0,1]=coords[i][0,1]-0.5*float(line_width)
            coords[i][4,0]=coords[i][4,0]-0.5*float(line_width)
        layer5.data=coords
        layer5.edge_width=[line_width]*len(coords)
        layer5.opacity=1.0
    else:
        layer5.opacity=0.0
layer2.events.mode.connect(update_grid_position)
def update_cross_position(event):
    if len(event.source.selected_data)>0:
        media=list(event.source.selected_data)
        coords=[layer2.data[i] for i in media]
        layer3.data=np.array(coords)
        layer3.edge_color=['red']*len(media)
        layer3.face_color=['red']*len(media)
        layer3.selected_data.clear()
        layer3.opacity=1.0
    else:
        layer3.selected_data.clear()
        layer3.opacity=0.0
layer4.events.mode.connect(update_cross_position)
@magicgui(slider_int={"widget_type": "Slider", "min": 1,'max':7,'label': 'Clusters of Kmeans'})
def slider(slider_int=5):
    kmean=KMeans(n_clusters=slider_int, random_state=0).fit(tile_ma)
    p_color_list=[color_list[i] for i in kmean.labels_]
    layer2.edge_color=p_color_list
    layer4.face_color=p_color_list
viewer.window.add_dock_widget(slider)
napari.run()