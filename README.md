Run pipeline by running
python napari_ui.py --image_path [path of the input wsi] --model_path [path of the model to generate embedding] --prefix [path to store the media data]

If there is no media data in the prefix directory, program will run preprocessing from very begining and could take a long time. Otherwise, preprocessed data will be loaded.

## xml_parser
Example xml file used: test_046.xml from the Camelyon 2016 dataset. Download image from this link: https://drive.google.com/drive/folders/0BzsdkU4jWx9Bb19WNndQTlUwb2M?resourcekey=0-FREBAxB4QK4bt9Zch_g5Mg
