# Asahi mosaic line generator

Partition a dataset into mosaic .shp files based on geolocation/road location/house and building location

### Libraries used

Python3
GDAL==2.4.2
geopandas==0.8.1
imageio==2.9.0
llvmlite==0.34.0
matplotlib==3.2.1
networkx==2.4
numba==0.51.0
numpy==1.18.5
osmnx==0.9
pandas==1.1.0
Pillow==5.1.0
pyshp==2.1.2
rasterio==1.1.5
Rtree==0.9.4
scikit-image==0.17.2
scipy==1.5.1
tensorflow-gpu==1.14.0
utm==0.6.0
opencv==4.4.0

### Dataset

1_単写真オルソデータ dataset1
H29_Fujisawa        dataset2
H29_HigashiKurume   dataset3

https://s3.console.aws.amazon.com/s3/upload/results-jon/houses_roads.tar.xz?region=ap-northeast-2


### Run

python3 inference_houses.py --img-path16bit /path/to/dataset/ --img-path8bit /path/to/dataset8bit/ --dataset dataset1
python3 inference_roads.py --img-path16bit /path/to/dataset/ --img-path8bit /path/to/dataset8bit/ --dataset dataset1

Where img-path16bit is the path to a directory containing all the .tif and .tfw files for a dataset to be processed, the images will be automatically converted to 8bit and saved to the path contained in img-path8bit, the optional parameter dataset is there to separate different datasets by name.
This commands will process the images and generate masks for the detected houses and roads respectively and will save them to ./houses and ./roads

python3 mosaic.py --img-path16bit /path/to/dataset/ --img-path8bit /path/to/dataset8bit/ --dataset dataset1
This commands will process the images and generate individual mosaic lines and will save them to ./mosaic_tmp

python3  piecing_releasefinal.py --img-path16bit /path/to/dataset/ --img-path8bit /path/to/dataset8bit/ --dataset dataset1
This commands will process the images and generate mosaic lines that fit together as a dataset and will save them to ./output


### Time

Due to the large image file size the execution time for the road/house mask generation is around 1 minute per mask in a RTX2080 GPU, intermediate mosaic lines take around 1 minute per mask, but might take up to 10 minutes or more depending on amount of buildings and compleity of the graph, generating the final mosaic .shp should take around a day of processing for ~200 images 


