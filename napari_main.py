# -*- coding: utf-8 -*-
"""
Created on Tue Dec 21 23:21:06 2021

@author: lhy
"""
import os
os.environ['KMP_DUPLICATE_LIB_OK']='True'
import numpy as np
import napari
import nest_asyncio
nest_asyncio.apply()
from sklearn.decomposition import PCA
import csv
from sklearn.cluster import KMeans
from magicgui import magicgui
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
    max_row=-1
    for i in range(len(coords)):
        if coords[i][0]>max_row:
            max_row=coords[i][0]
        media_x,media_y=transform_origpos(coords[i][0],coords[i][1],row,col,orig_row,orig_col,plot_col)
        media1_x,media1_y=transform_origpos(coords[i][0]+tile_size,coords[i][1]+tile_size,row,col,orig_row,orig_col,plot_col)
        media=np.array([[media_x,media_y],[media_x,media1_y],[media1_x,media1_y],[media1_x,media_y]],dtype=int)
        polys.append(media)
    #print(max_row)
    return polys
def gen_mask(coords,row,col,orig_row,orig_col,plot_col,tile_size,line_width):
    lines=[]
    ld=0.5*float(line_width)
    #print(row/orig_row)
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
def napari_main(tile_ma,coords,level_app,tile_orig,labels,wsi,args):
    global cur_orig
    global cur_n_clusters
    tile_ma=np.array(tile_ma,dtype=float)
    pca = PCA(n_components=2)
    pca_matrix=pca.fit_transform(tile_ma)
    kmean=KMeans(n_clusters=5, random_state=0).fit(tile_ma)
    tile_orig=np.array(tile_orig,dtype=float)
    pca_orig = PCA(n_components=2)
    pca_matrix_orig=pca_orig.fit_transform(tile_orig)
    level_display=wsi.slide.slide.level_count-1
    for i in range(wsi.slide.slide.level_count-1):
        if max(wsi.slide.get_image_shape(i))<15000:
            level_display=i
            break
    thumbnail=wsi.slide.extract_region(location=(0,0),level=level_display,size=wsi.slide.get_image_shape(level_display))
    color_list=['green','blue','orange','brown','purple','yellow','pink']
    point_size=int(0.005*thumbnail.shape[0])
    layer2_data,point_matrix,min_col,min_row,max_row,max_col,matrix_min,matrix_max=generate_pca_point(thumbnail.shape[0],pca_matrix,point_size)
    layer2_data_orig,point_matrix_orig,min_col_orig,min_row_orig,max_row_orig,max_col_orig,matrix_min_orig,matrix_max_orig=generate_pca_point(thumbnail.shape[0],pca_matrix_orig,point_size)
    line_width=8
    polys=gen_cluster_mask(coords,thumbnail.shape[0],thumbnail.shape[1],wsi.slide.get_image_shape(0)[0],wsi.slide.get_image_shape(0)[1],layer2_data.shape[1],args.tile_size*int(2**level_app))
    if labels is not None:
        coords_label=[coords[i] for i in range(len(coords)) if labels[i]]
        polys_label=gen_cluster_mask(coords_label,thumbnail.shape[0],thumbnail.shape[1],wsi.slide.get_image_shape(0)[0],wsi.slide.get_image_shape(0)[1],layer2_data.shape[1],args.tile_size*int(2**level_app))
        polys_label_orig=gen_cluster_mask(coords_label,thumbnail.shape[0],thumbnail.shape[1],wsi.slide.get_image_shape(0)[0],wsi.slide.get_image_shape(0)[1],layer2_data_orig.shape[1],args.tile_size*int(2**level_app))
    polys_orig=gen_cluster_mask(coords,thumbnail.shape[0],thumbnail.shape[1],wsi.slide.get_image_shape(0)[0],wsi.slide.get_image_shape(0)[1],layer2_data_orig.shape[1],args.tile_size*int(2**level_app))
    polys_orig=gen_cluster_mask(coords,thumbnail.shape[0],thumbnail.shape[1],wsi.slide.get_image_shape(0)[0],wsi.slide.get_image_shape(0)[1],layer2_data_orig.shape[1],args.tile_size*int(2**level_app))
    viewer = napari.Viewer()
    p_color_list=[color_list[i] for i in kmean.labels_]
    layer1=viewer.add_image(np.concatenate((layer2_data,thumbnail),axis=1),name='wsi')
    layer2=viewer.add_points(point_matrix,size=point_size,edge_color=p_color_list,name='pca plot')
    layer3=viewer.add_points(point_matrix[0:1],size=point_size*3,symbol='cross',opacity=0,edge_color='red',face_color='red',name='cross')
    layer4=viewer.add_shapes(polys,edge_width=0,face_color=p_color_list,opacity=0.5,name='tile')
    layer5=viewer.add_shapes(np.array([[1114,2997.5],[1114,3064],[1178,3064],[1178,3000],[1111.5,3000]]),edge_color='red',edge_width=line_width,shape_type='path',opacity=0.0,name='grid')
    if labels is not None:
        layer6=viewer.add_shapes(polys_label,edge_width=0,face_color='black',opacity=0.5,name='cancer label')
    layer1.editable=False
    layer3.editable=False
    layer5.editable=False
    layer6.editable=False
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
    @magicgui(slider_int={"widget_type": "Slider", "min": 1,'max':7,'label': 'Clusters of Kmeans'},
              radio_option={
                      "widget_type": "RadioButtons",
                      "orientation": "vertical",
                      "choices": [("orig_resnet", True), ("path_resent", False)]})
    def widgets(slider_int=5,radio_option=False):
        layer3.opacity=0.0
        layer5.opacity=0.0
        if radio_option:
            kmean=KMeans(n_clusters=slider_int, random_state=0).fit(tile_orig)
            p_color_list=[color_list[i] for i in kmean.labels_]
            layer1.data=np.concatenate((layer2_data_orig,thumbnail),axis=1)
            layer2.data=point_matrix_orig
            layer2.edge_color=p_color_list
            layer4.data=polys_orig
            layer4.face_color=p_color_list
            if labels is not None:
                layer6.data=polys_label_orig
        else:
            kmean=KMeans(n_clusters=slider_int, random_state=0).fit(tile_ma)
            p_color_list=[color_list[i] for i in kmean.labels_]
            layer1.data=np.concatenate((layer2_data,thumbnail),axis=1)
            layer2.data=point_matrix
            layer2.edge_color=p_color_list
            layer4.data=polys
            layer4.face_color=p_color_list
            if labels is not None:
                layer6.data=polys_label
    viewer.window.add_dock_widget(widgets)
    napari.run()