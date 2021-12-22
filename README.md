# An interative pathology image viewer
This repo contains code for the Capstone Project of the MS Heath Data Science at Harvard University. We developed an interactive viewer to help pathologists view images through embeddings generated from deep learning models. 

## Source of demo image
Camelyon 2016 dataset. This can be applied to any pathology images. 

## Pipeline flow explained
1. Divide images into tiles
2. Generate embeddings of tiles using a model pre-trained on pathology images
3. Conduct K-means to classify tiles based on their embeddings
4. Run the xml parser to parse the cancerous polygons annotated on the image
5. Integrate the above info into the Napari viewer

## How to run pipeline
Run pipeline by running
python main.py 
--image_path [path of the input wsi]

--model_path [path of the model to generate embedding] 

--prefix [path to store the media data] 

--tile_size [tile size]

--level [perform embedding at which level of image. the higher the level, the quicker the preprocess]

--xml_path [the path of label xml file]

--tissue_ratio [filter out tiles whose tissue ratio is below this value]

If there is no media data in the prefix directory, program will run preprocessing from very begining and could take a long time. Otherwise, preprocessed data will be loaded.

## xml_parser
Example xml file used: test_046.xml from the Camelyon 2016 dataset. Download image from this link: https://drive.google.com/drive/folders/0BzsdkU4jWx9Bb19WNndQTlUwb2M?resourcekey=0-FREBAxB4QK4bt9Zch_g5Mg

## Demo
https://youtu.be/MdAcIYptNvE
