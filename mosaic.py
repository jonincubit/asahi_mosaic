from __future__ import print_function
from skeleton import build_graph
from scipy.spatial import Voronoi, voronoi_plot_2d
import shutil
import math
import networkx as nx
from shapely.geometry import Point, LineString
import glob
import json
from skimage.morphology import convex_hull_image
import numpy as np
import datetime
from PIL import ImageDraw
from PIL import Image

import argparse
import gc
from datetime import datetime
from matplotlib.pylab import plt
import os
import sys
import time
from time import sleep
import cv2
import PIL
PIL.Image.MAX_IMAGE_PIXELS = 933120000
sys.path.append("./")
SAVE_DIR = './mosaic_tmp/'


def convert_to_8Bit(inputRaster, outputRaster, outputPixType='Byte', outputFormat='GTiff', rescale_type='rescale', percentiles=[2, 98]):
    srcRaster = gdal.Open(inputRaster)
    cmd = ['gdal_translate', '-ot', outputPixType, '-of',
           outputFormat]
    for bandId in range(srcRaster.RasterCount):
        bandId = bandId+1
        band = srcRaster.GetRasterBand(bandId)
        if rescale_type == 'rescale':
            bmin = band.GetMinimum()
            bmax = band.GetMaximum()
            if bmin is None or bmax is None:
                (bmin, bmax) = band.ComputeRasterMinMax(1)
            band_arr_tmp = band.ReadAsArray()
            bmin = np.percentile(band_arr_tmp.flatten(),  percentiles[0])
            bmax = np.percentile(band_arr_tmp.flatten(), percentiles[1])
        elif isinstance(rescale_type, dict):
            bmin, bmax = rescale_type[bandId]
        else:
            bmin, bmax = 0, 65535

        cmd.append('-scale_{}'.format(bandId))
        cmd.append('{}'.format(bmin))
        cmd.append('{}'.format(bmax))
        cmd.append('{}'.format(0))
        cmd.append('{}'.format(255))

    cmd.append(inputRaster)
    cmd.append(outputRaster)
    print(("Conversion command:", cmd))
    subprocess.call(cmd)

    return


def expand_voronoi(voronoi_data, cropped_y, cropped_x, frame_falsev2, dimxl):
    temp = np.zeros_like(frame_falsev2)
    cont = 1
    voronoi_data_dup = []
    for coord in range(len(voronoi_data)):
        if([int(voronoi_data[coord][0]/4)-int((cropped_y*(float(frame_falsev2.shape[0])/float(dimxl[0])))), int(voronoi_data[coord][1]/4)-int((cropped_x*(float(frame_falsev2.shape[0])/float(dimxl[0]))))] not in voronoi_data_dup):
            voronoi_data_dup.append([int(voronoi_data[coord][0]/4)-int((cropped_y*(float(frame_falsev2.shape[0])/float(
                dimxl[0])))), int(voronoi_data[coord][1]/4)-int((cropped_x*(float(frame_falsev2.shape[0])/float(dimxl[0]))))])
            temp[(int(voronoi_data[coord][0]/4)-int((cropped_y*(float(frame_falsev2.shape[0])/float(dimxl[0])))),
                  int(voronoi_data[coord][1]/4)-int((cropped_x*(float(frame_falsev2.shape[0])/float(dimxl[0])))))] = cont
            cont = cont+1
    temp2 = cv2.resize(temp, dsize=None, fx=1.2, fy=1.2,
                       interpolation=cv2.INTER_NEAREST)
    len_temp2 = len(np.where(temp2 > 0)[0])
    temp2 = temp2[int((temp2.shape[0]-temp.shape[0])/2):temp2.shape[0]-int((temp2.shape[0]-temp.shape[0])/2),
                  int((temp2.shape[1]-temp.shape[1])/2):temp2.shape[1]-int((temp2.shape[1]-temp.shape[1])/2)]
    if(len(np.where(temp2 > 0)[0]) != len_temp2):
        return voronoi_data
    new_voronoi_data = []
    cont = 1
    for coord in range(len(voronoi_data_dup)):
        dims = np.where(temp2 == cont)
        new_voronoi_data.append(
            [cropped_y+(dims[0][np.argmin(dims[0])]*4), cropped_x+(dims[1][np.argmin(dims[0])]*4)])
        if(dims[0][np.argmin(dims[0])] > temp.shape[0] or dims[1][np.argmin(dims[0])] > temp.shape[1]):
            exit()
        cont = cont+1
    del temp
    del temp2
    del voronoi_data_dup
    return new_voronoi_data


def get_voronoi(filenames):
    filenames.sort()
    voronoi_points = []
    voronoi_meta = {}
    cont = 0
    cont2 = 0
    for im_fn_orig in filenames:

        im = Image.open(im_fn_orig)
        width, height = im.size
        if os.path.isfile(im_fn_orig[:-4]+".tfw"):
            f = open(im_fn_orig[:-4]+".tfw", "r")
        else:
            f = open(im_fn_orig[:-4].replace("8bit", "")+".tfw", "r")
        contents = f.read()
        left_upper_x = float(contents.split("\n")[4])
        left_upper_y = float(contents.split("\n")[5])
        pixel_size = abs(float(contents.split("\n")[0]))
        voronoi_meta[im_fn_orig] = [(left_upper_y+(pixel_size/2)), (left_upper_x-(pixel_size/2)), (left_upper_y+(pixel_size/2))-(
            (height*pixel_size)/2), (left_upper_x-(pixel_size/2))+((width*pixel_size)/2), height, width, pixel_size]
        voronoi_points.append([(left_upper_x-(pixel_size/2))+((width*pixel_size)/2),
                               (left_upper_y+(pixel_size/2))-((height*pixel_size)/2)])
        cont = cont+1
    vor = Voronoi(voronoi_points)
    voronoi_dict = {}
    cont2 = 0
    for im_fn_orig in filenames:
        if(-1 in vor.regions[vor.point_region[cont2]]):
            voronoi_dict[os.path.basename(im_fn_orig)] = [[int(voronoi_meta[im_fn_orig][5]*0.3), int(voronoi_meta[im_fn_orig][4]*0.3)], [int(voronoi_meta[im_fn_orig][5]*0.3), int(
                voronoi_meta[im_fn_orig][4]*0.7)], [int(voronoi_meta[im_fn_orig][5]*0.7), int(voronoi_meta[im_fn_orig][4]*0.7)], [int(voronoi_meta[im_fn_orig][5]*0.7), int(voronoi_meta[im_fn_orig][4]*0.3)]]
            cont2 = cont2+1
            continue
        pointsy = [-1*int(round((vor.vertices[x][1] - voronoi_meta[im_fn_orig][0]) /
                                voronoi_meta[im_fn_orig][6])) for x in vor.regions[vor.point_region[cont2]]]
        pointsx = [int(round((1*(vor.vertices[x][0] - voronoi_meta[im_fn_orig][1])) /
                             voronoi_meta[im_fn_orig][6])) for x in vor.regions[vor.point_region[cont2]]]
        plt.text(voronoi_meta[im_fn_orig][3], voronoi_meta[im_fn_orig]
                 [2], os.path.basename(im_fn_orig)[:-4], fontsize=10)
        if(sum(1 for number in pointsx if number < 0.1*voronoi_meta[im_fn_orig][5]) > 0 or sum(1 for number in pointsy if number < 0.1*voronoi_meta[im_fn_orig][4]) > 0 or sum(1 for number in pointsx if number > 0.9*voronoi_meta[im_fn_orig][5]) > 0 or sum(1 for number in pointsy if number > 0.9*voronoi_meta[im_fn_orig][4]) > 0):
            voronoi_dict[os.path.basename(im_fn_orig)] = [[int(voronoi_meta[im_fn_orig][5]*0.3), int(voronoi_meta[im_fn_orig][4]*0.3)], [int(voronoi_meta[im_fn_orig][5]*0.3), int(
                voronoi_meta[im_fn_orig][4]*0.7)], [int(voronoi_meta[im_fn_orig][5]*0.7), int(voronoi_meta[im_fn_orig][4]*0.7)], [int(voronoi_meta[im_fn_orig][5]*0.7), int(voronoi_meta[im_fn_orig][4]*0.3)]]
            cont2 = cont2+1
            continue
        voronoi_dict[os.path.basename(im_fn_orig)] = [
            [pointsx[x], pointsy[x]] for x in range(len(pointsx))]
        cont2 = cont2+1
    return voronoi_dict


def flatten(l):
    return [item for sublist in l for item in sublist]


def distance(bb1, bb2):
    p1 = [bb1[1], bb1[0]]
    p2 = [bb2[1], bb2[0]]
    return math.sqrt(((p1[0]-p2[0])**2)+((p1[1]-p2[1])**2))


def get_arguments():
    parser = argparse.ArgumentParser(description="Mosaic")
    parser.add_argument("--img-path16bit", type=str, default="",
                        help="Path to the asahi image files(16 bit).")
    parser.add_argument("--img-path8bit", type=str, default="",
                        help="Path to the asahi image files(8 bit).")
    parser.add_argument("--dataset", type=str, default="",
                        help="Optional dataset prefix")
    parser.add_argument("--houses-dir", type=str, default="./houses",
                        help="Path to house masks")
    parser.add_argument("--roads-dir", type=str, default="./roads/",
                        help="Path to road masks")
    parser.add_argument("--save-dir", type=str, default=SAVE_DIR,
                        help="Where to save predicted mosaic lines")
    return parser.parse_args()


def main():
    args = get_arguments()
    cont = 0
    cont2 = 0
    tnp = 0.0
    tpp = 0.0
    fpp = 0.0
    fnp = 0.0
    iou_r = 0.0
    iou_l = 0.0
    iou_o = 0.0
    contr = 0
    contl = 0
    conto = 0
    max_line_thickness = 300
    if(args.img_path16bit is not ""):
        filenames16bit = [img for img in glob.glob(
            args.img_path16bit+"/*.tif")]
        if not os.path.exists(args.img_path8bit):
            os.makedirs(args.img_path8bit)
        for im_fn_orig in filenames16bit:
            if(not os.path.isfile(args.img_path8bit+"/"+os.path.basename(im_fn_orig))):
                convert_to_8Bit(im_fn_orig, args.img_path8bit +
                                "/"+os.path.basename(im_fn_orig))
    filenames = [img for img in glob.glob(args.img_path8bit+"/*.tif") if not os.path.isfile(
        args.save_dir+'mosaic_mask_'+args.dataset+"_"+os.path.basename(img)+'.txt')]
    if not os.path.exists(args.save_dir):
        os.makedirs(args.save_dir)
    voronoi_dict = get_voronoi(
        [img for img in glob.glob(args.img_path8bit+"/*.tif")])
    for im_fn_orig in filenames:
        print(im_fn_orig)
        line_thickness = 25  # 25
        subtract_houses = 0
        repeat = 1
        max_line_thickness = 300
        real_img = cv2.imread(im_fn_orig)
        if(real_img is None):
            print("img error")
            continue
        house_cropped = cv2.imread(
            args.houses_dir+"/houses_"+args.dataset+"_"+os.path.basename(im_fn_orig)[:-4]+".png", 0)
        backtorgb2 = cv2.imread(
            args.roads_dir+"/roads_"+args.dataset+"_"+os.path.basename(im_fn_orig)[:-4]+".png", 0)
        cropped_x = 0
        cropped_y = 0
        if(real_img.shape[0] > real_img.shape[1]):
            cropped = real_img.shape[0]-real_img.shape[1]
            cropped_x = int(cropped/2)
            real_img = real_img[int(
                cropped/2):int(cropped/2)+real_img.shape[1], :]
            backtorgb2 = backtorgb2[int((backtorgb2.shape[0]-backtorgb2.shape[1])/2):int(
                (backtorgb2.shape[0]-backtorgb2.shape[1])/2)+backtorgb2.shape[1], :]
            house_cropped = house_cropped[int(
                cropped/2):int(cropped/2)+real_img.shape[1], :]
        else:
            cropped = real_img.shape[1]-real_img.shape[0]
            cropped_y = int(cropped/2)
            real_img = real_img[:, int(
                cropped/2):int(cropped/2)+real_img.shape[0]]
            backtorgb2 = backtorgb2[:, int((backtorgb2.shape[1]-backtorgb2.shape[0])/2):int(
                ((backtorgb2.shape[1]-backtorgb2.shape[0])/2)+backtorgb2.shape[0])]
            house_cropped = house_cropped[:, int(
                cropped/2):int(cropped/2)+real_img.shape[0]]
        dimxl = real_img.shape
        xl_dims = dimxl
        houses_orig = cv2.resize(
            house_cropped, (int(real_img.shape[0]/4), int(real_img.shape[1]/4)))
        real_img = cv2.resize(
            real_img, (int(real_img.shape[0]/4), int(real_img.shape[1]/4)))
        cont = cont+1
        while(repeat == 1):
            repeat = 0
            framev = backtorgb2.copy()
            framev2 = backtorgb2.copy()
            frame_falsev = np.zeros(
                (backtorgb2.shape[0], backtorgb2.shape[1]), np.uint8)
            frame_falsev2 = np.zeros(
                (backtorgb2.shape[0], backtorgb2.shape[1]), np.uint8)
            thresh1 = 0.1
            thresh2 = 0.7
            thresh3 = 0.9
            thresh4 = 0.3
            thresh5 = 0.6
            thresh6 = 0.4
            voronoi_data = voronoi_dict[os.path.basename(im_fn_orig)]
            if(voronoi_data == None):
                repeat = 0
                print("Voronoi map not available for image "+im_fn_orig)
                break
            voronoi_data = expand_voronoi(
                voronoi_data, cropped_y, cropped_x, frame_falsev2, dimxl)
            frame_falsev = Image.fromarray(frame_falsev)
            draw = ImageDraw.Draw(frame_falsev)
            for coord in range(len(voronoi_data)):
                if(coord == len(voronoi_data)-1):
                    draw.line([(int(voronoi_data[coord][0]/4)-int((cropped_y*(float(frame_falsev2.shape[0])/float(dimxl[0])))), int(voronoi_data[coord][1]/4)-int((cropped_x*(float(frame_falsev2.shape[0])/float(dimxl[0]))))), (int(
                        voronoi_data[0][0]/4)-int((cropped_y*(float(frame_falsev2.shape[0])/float(dimxl[0])))), int(voronoi_data[0][1]/4)-int((cropped_x*(float(frame_falsev2.shape[0])/float(dimxl[0])))))], fill=255, width=max_line_thickness)
                    break
                draw.line([(int(voronoi_data[coord][0]/4)-int((cropped_y*(float(frame_falsev2.shape[0])/float(dimxl[0])))), int(voronoi_data[coord][1]/4)-int((cropped_x*(float(frame_falsev2.shape[0])/float(dimxl[0]))))), (int(
                    voronoi_data[coord+1][0]/4)-int((cropped_y*(float(frame_falsev2.shape[0])/float(dimxl[0])))), int(voronoi_data[coord+1][1]/4)-int((cropped_x*(float(frame_falsev2.shape[0])/float(dimxl[0])))))], fill=255, width=max_line_thickness)
            frame_falsev = np.array(frame_falsev)
            ret, labels = cv2.connectedComponents(
                cv2.bitwise_not(frame_falsev))
            frame_falsev_m = convex_hull_image(frame_falsev.copy())
            frame_falsev[frame_falsev_m > 0] = 255
            frame_falsev[labels == labels[int(
                labels.shape[0]/2)][int(labels.shape[1]/2)]] = 0
            del labels
            frame_falsev2 = frame_falsev.copy()
            framev[frame_falsev == 0] = 0
            framev[int(framev.shape[0]*thresh5):, :] = 0
            framev2[frame_falsev2 == 0] = 0
            framev2[:int(framev.shape[0]*thresh6), :] = 0
            kernel3 = np.ones((3, 3), np.uint8)
            frame_falsev_eroded = cv2.erode(
                frame_falsev, kernel3, iterations=int(line_thickness/2))
            frame_falsev_guide = frame_falsev.copy()
            frame_falsev_guide[frame_falsev_eroded == 255] = 0
            if(line_thickness % 50 == 0):
                dimsmo = np.where(frame_falsev_guide[int(
                    frame_falsev_guide.shape[0]*0.5), :] > 0)[0]
                frame_falsev_guideshape = frame_falsev_guide.shape
                frame_falsev_guide = Image.fromarray(frame_falsev_guide)
                draw = ImageDraw.Draw(frame_falsev_guide)
                draw.line([(min(dimsmo), int(frame_falsev_guideshape[0]*0.5)), (min(dimsmo)+int(
                    line_thickness/2)+25, int(frame_falsev_guideshape[0]*0.5))], fill=255, width=2)
                draw.line([(max(dimsmo)-int(line_thickness/2)-25, int(frame_falsev_guideshape[0]*0.5)),
                           (max(dimsmo), int(frame_falsev_guideshape[0]*0.5))], fill=255, width=2)
                draw.line([(min(dimsmo), int(frame_falsev_guideshape[0]*0.5)+25), (min(dimsmo)+int(
                    line_thickness/2)+25, int(frame_falsev_guideshape[0]*0.5)+25)], fill=255, width=2)
                draw.line([(max(dimsmo)-int(line_thickness/2)-25, int(frame_falsev_guideshape[0]*0.5)+25),
                           (max(dimsmo), int(frame_falsev_guideshape[0]*0.5)+25)], fill=255, width=2)
                draw.line([(min(dimsmo)+int(line_thickness/2)+25, int(frame_falsev_guideshape[0]*0.5)), (min(
                    dimsmo)+int(line_thickness/2)+25, int(frame_falsev_guideshape[0]*0.5)+25)], fill=255, width=2)
                draw.line([(max(dimsmo)-int(line_thickness/2)-25, int(frame_falsev_guideshape[0]*0.5)), (max(
                    dimsmo)-int(line_thickness/2)-25, int(frame_falsev_guideshape[0]*0.5)+25)], fill=255, width=2)
                frame_falsev_guide = np.array(frame_falsev_guide, np.uint8)
                framev[frame_falsev_guide == 255] = 255
                framev2[frame_falsev_guide == 255] = 255
            else:
                framev_m = Image.fromarray(np.zeros_like(framev))
                draw = ImageDraw.Draw(framev_m)
                for coord in range(len(voronoi_data)):
                    if(coord == len(voronoi_data)-1):
                        draw.line([(int(voronoi_data[coord][0]/4)-int((cropped_y*(float(frame_falsev2.shape[0])/float(dimxl[0])))), int(voronoi_data[coord][1]/4)-int((cropped_x*(float(frame_falsev2.shape[0])/float(dimxl[0]))))), (int(
                            voronoi_data[0][0]/4)-int((cropped_y*(float(frame_falsev2.shape[0])/float(dimxl[0])))), int(voronoi_data[0][1]/4)-int((cropped_x*(float(frame_falsev2.shape[0])/float(dimxl[0])))))], fill=255, width=line_thickness)
                        break
                    draw.line([(int(voronoi_data[coord][0]/4)-int((cropped_y*(float(frame_falsev2.shape[0])/float(dimxl[0])))), int(voronoi_data[coord][1]/4)-int((cropped_x*(float(frame_falsev2.shape[0])/float(dimxl[0]))))), (int(
                        voronoi_data[coord+1][0]/4)-int((cropped_y*(float(frame_falsev2.shape[0])/float(dimxl[0])))), int(voronoi_data[coord+1][1]/4)-int((cropped_x*(float(frame_falsev2.shape[0])/float(dimxl[0])))))], fill=255, width=line_thickness)
                framev_m = np.array(framev_m, np.uint8)
                dimsmo = np.where(
                    framev_m[int(framev_m.shape[0]*0.5), :] > 0)[0]
                framev_mshape = framev_m.shape
                framev_m = Image.fromarray(framev_m)
                draw = ImageDraw.Draw(framev_m)
                draw.line([(min(dimsmo)-20, int(framev_mshape[0]*0.5)), (min(dimsmo) +
                                                                         line_thickness+10, int(framev_mshape[0]*0.5))], fill=255, width=2)
                draw.line([(max(dimsmo)-line_thickness-10, int(framev_mshape[0]*0.5)),
                           (max(dimsmo)+20, int(framev_mshape[0]*0.5))], fill=255, width=2)
                draw.line([(min(dimsmo)-20, int(framev_mshape[0]*0.5)+25), (min(dimsmo) +
                                                                            line_thickness+10, int(framev_mshape[0]*0.5)+25)], fill=255, width=2)
                draw.line([(max(dimsmo)-line_thickness-10, int(framev_mshape[0]*0.5)+25),
                           (max(dimsmo)+20, int(framev_mshape[0]*0.5)+25)], fill=255, width=2)
                draw.line([(min(dimsmo)+line_thickness+10, int(framev_mshape[0]*0.5)), (min(
                    dimsmo)+line_thickness+10, int(framev_mshape[0]*0.5)+25)], fill=255, width=2)
                draw.line([(max(dimsmo)-line_thickness-10, int(framev_mshape[0]*0.5)), (max(
                    dimsmo)-line_thickness-10, int(framev_mshape[0]*0.5)+25)], fill=255, width=2)
                framev_m = np.array(framev_m, np.uint8)
                ret, labels = cv2.connectedComponents(
                    cv2.bitwise_not(framev_m))
                framev_m2 = convex_hull_image(framev_m.copy())
                framev_m[framev_m2 > 0] = 255
                framev_m[labels == labels[int(
                    labels.shape[0]/2)][int(labels.shape[1]/2)]] = 0
                framev[framev_m > 0] = 255
                framev2[framev_m > 0] = 255
                del labels
            guide_maskv = framev.copy()
            if(line_thickness % 50 == 0):
                line_thickness_add = 75
                guide_maskv[frame_falsev_guide == 255] = 255
            else:
                line_thickness_add = 25
                guide_maskv_m = Image.fromarray(np.zeros_like(guide_maskv))
                draw = ImageDraw.Draw(guide_maskv_m)
                for coord in range(len(voronoi_data)):
                    if(coord == len(voronoi_data)-1):
                        draw.line([(int(voronoi_data[coord][0]/4)-int((cropped_y*(float(frame_falsev2.shape[0])/float(dimxl[0])))), int(voronoi_data[coord][1]/4)-int((cropped_x*(float(frame_falsev2.shape[0])/float(dimxl[0]))))), (int(
                            voronoi_data[0][0]/4)-int((cropped_y*(float(frame_falsev2.shape[0])/float(dimxl[0])))), int(voronoi_data[0][1]/4)-int((cropped_x*(float(frame_falsev2.shape[0])/float(dimxl[0])))))], fill=255, width=line_thickness)
                        break
                    draw.line([(int(voronoi_data[coord][0]/4)-int((cropped_y*(float(frame_falsev2.shape[0])/float(dimxl[0])))), int(voronoi_data[coord][1]/4)-int((cropped_x*(float(frame_falsev2.shape[0])/float(dimxl[0]))))), (int(
                        voronoi_data[coord+1][0]/4)-int((cropped_y*(float(frame_falsev2.shape[0])/float(dimxl[0])))), int(voronoi_data[coord+1][1]/4)-int((cropped_x*(float(frame_falsev2.shape[0])/float(dimxl[0])))))], fill=255, width=line_thickness)
                guide_maskv_m = np.array(guide_maskv_m, np.uint8)
                dimsmo = np.where(
                    guide_maskv_m[int(guide_maskv_m.shape[0]*0.5), :] > 0)[0]
                framev_mshape = guide_maskv_m.shape
                guide_maskv_m = Image.fromarray(guide_maskv_m)
                draw = ImageDraw.Draw(guide_maskv_m)
                draw.line([(min(dimsmo)-20, int(framev_mshape[0]*0.5)), (min(dimsmo) +
                                                                         line_thickness+10, int(framev_mshape[0]*0.5))], fill=255, width=2)
                draw.line([(max(dimsmo)-line_thickness-10, int(framev_mshape[0]*0.5)),
                           (max(dimsmo)+20, int(framev_mshape[0]*0.5))], fill=255, width=2)
                draw.line([(min(dimsmo)-20, int(framev_mshape[0]*0.5)+25), (min(dimsmo) +
                                                                            line_thickness+10, int(framev_mshape[0]*0.5)+25)], fill=255, width=2)
                draw.line([(max(dimsmo)-line_thickness-10, int(framev_mshape[0]*0.5)+25),
                           (max(dimsmo)+20, int(framev_mshape[0]*0.5)+25)], fill=255, width=2)
                draw.line([(min(dimsmo)+line_thickness+10, int(framev_mshape[0]*0.5)), (min(
                    dimsmo)+line_thickness+10, int(framev_mshape[0]*0.5)+25)], fill=255, width=2)
                draw.line([(max(dimsmo)-line_thickness-10, int(framev_mshape[0]*0.5)), (max(
                    dimsmo)-line_thickness-10, int(framev_mshape[0]*0.5)+25)], fill=255, width=2)
                guide_maskv_m = np.array(guide_maskv_m, np.uint8)
                ret, labels = cv2.connectedComponents(
                    cv2.bitwise_not(guide_maskv_m))
                guide_maskv_m2 = convex_hull_image(guide_maskv_m.copy())
                guide_maskv_m[guide_maskv_m2 > 0] = 255
                guide_maskv_m[labels == labels[int(
                    labels.shape[0]/2)][int(labels.shape[1]/2)]] = 0
                guide_maskv[guide_maskv_m > 0] = 255
                del guide_maskv_m
                del guide_maskv_m2
                del labels
                del framev_m
            framev[framev > 0] = 255
            framev2[framev2 > 0] = 255
            graph_size = real_img.shape[1]  # 3500#3500
            houses = cv2.dilate(houses_orig, kernel3, iterations=1)
            if(subtract_houses == 0):
                framev2[houses > 0] = 0
                framev[houses > 0] = 0
                housesdil = cv2.dilate(houses, kernel3, iterations=15)
                housesdil[houses > 0] = 0
                housesdil[guide_maskv == 0] = 0
                housesdil2 = housesdil.copy()
                housesdil[int(framev.shape[0]*thresh2):, :] = 0
                housesdil2[:int(framev.shape[0]*thresh4), :] = 0
                framev2[housesdil2 > 0] = 255
                framev[housesdil > 0] = 255
            if(subtract_houses == 1):
                framev2[houses > 0] = 0
                framev[houses > 0] = 0
                housesdil = cv2.dilate(houses, kernel3, iterations=15)
                housesdil[houses == 254] = 0
                housesdil[guide_maskv == 0] = 0
                housesdil2 = housesdil.copy()
                housesdil[int(framev.shape[0]*thresh2):, :] = 0
                housesdil2[:int(framev.shape[0]*thresh4), :] = 0
                framev2[housesdil2 > 0] = 255
                framev[housesdil > 0] = 255
            if(len(np.where(framev[:int(framev.shape[0]*thresh5), :][3*int(framev[:int(framev.shape[0]*thresh5), :].shape[0]/4):, int(framev[:int(framev.shape[0]*thresh5), :].shape[1]/2)] > 0)[0]) > 0):
                framev[int(framev.shape[0]*(thresh5-0.05)):, :] = 0
            if(len(np.where(framev2[int(framev2.shape[0]*thresh6):, :][:int(framev2[int(framev2.shape[0]*thresh6):, :].shape[0]/4), int(framev2[int(framev2.shape[0]*thresh6):, :].shape[1]/2)] > 0)[0]) > 0):
                framev2[:int(framev.shape[0]*(thresh6+0.05)), :] = 0
            dims = [[int(framev.shape[0]*(0.2)), int(framev.shape[0]*thresh5)],
                    [int(framev.shape[1]*thresh1), int(framev.shape[1]*thresh3)]]
            framev = cv2.resize(framev[min(dims[0]):max(dims[0]), min(
                dims[1]):max(dims[1])], (framev.shape[1], framev.shape[0]))
            dims2 = [[int(framev2.shape[0]*thresh6), int(framev2.shape[0]*(0.8))],
                     [int(framev2.shape[1]*thresh1), int(framev2.shape[1]*thresh3)]]
            framev2 = cv2.resize(framev2[min(dims2[0]):max(dims2[0]), min(
                dims2[1]):max(dims2[1])], (framev2.shape[1], framev2.shape[0]))
            resize_scale0 = float(
                max(dims[0])-min(dims[0]))/float(framev.shape[0])
            resize_scale1 = float(
                max(dims[1])-min(dims[1]))/float(framev.shape[1])
            resize_scale20 = float(
                max(dims2[0])-min(dims2[0])) / float(framev2.shape[0])
            resize_scale21 = float(
                max(dims2[1])-min(dims2[1]))/float(framev2.shape[1])
            wkt, g, vertices, graphimg, ske = build_graph(framev)
            wkt2, g2, vertices2, graphimg2, ske2 = build_graph(framev2)
            ps = np.array(vertices)
            ps2 = np.array(vertices2)
            seen_nodes = []
            for ids in list(g.nodes(data=True)):
                if framev[int(ids[1]['o'][0]), int(ids[1]['o'][1])] == 0:
                    g.remove_node(ids[0])
            for ids in list(g2.nodes(data=True)):
                if framev2[int(ids[1]['o'][0]), int(ids[1]['o'][1])] == 0:
                    g2.remove_node(ids[0])
            cont_edge = 0
            rem_edge = 1
            cont_edge = cont_edge+1
            rem_edge = 0
            for (s, e) in list(g.edges()):
                vals = flatten([[v] for v in g[s][e].values()])
                for val in vals:
                    ps = val.get('pts', [])
                    for po in ps:
                        if framev[po[0], po[1]] == 0:
                            g.remove_edge(s, e)
                            break
            cont_edge = 0
            rem_edge = 1
            cont_edge = cont_edge+1
            rem_edge = 0
            for (s, e) in list(g2.edges()):
                vals = flatten([[v] for v in g2[s][e].values()])
                for val in vals:
                    ps = val.get('pts', [])
                    for po in ps:
                        if framev2[po[0], po[1]] == 0:
                            g2.remove_edge(s, e)
                            break
            cont_edge = 1
            while(cont_edge > 0):
                cont_edge = 0
                for ids in list(g.nodes(data=True)):
                    if len(g.edges(ids[0])) < 1:
                        g.remove_node(ids[0])
                        cont_edge = cont_edge+1
                        break
                    if len(g.edges(ids[0])) == 1 and len(g[ids[0]][list(g.edges(ids[0]))[0][1]][list(g[ids[0]][list(g.edges(ids[0]))[0][1]].keys())[0]]['pts']) < 150:
                        g.remove_node(ids[0])
                        cont_edge = cont_edge+1
                        break
                for ids in list(g2.nodes(data=True)):
                    if len(g2.edges(ids[0])) < 1:
                        g2.remove_node(ids[0])
                        cont_edge = cont_edge+1
                        break
                    if len(g2.edges(ids[0])) == 1 and len(g2[ids[0]][list(g2.edges(ids[0]))[0][1]][list(g2[ids[0]][list(g2.edges(ids[0]))[0][1]].keys())[0]]['pts']) < 150:
                        g2.remove_node(ids[0])
                        cont_edge = cont_edge+1
                        break
            cont_edge = 1
            exa = []
            while(cont_edge > 0):
                cont_edge = 0
                for ids in list(g.nodes(data=True)):
                    if len(g.edges(ids[0])) == 2:
                        wei = g[ids[0]][list(g.edges(ids[0]))[
                            0][1]][0]['weight']+g[ids[0]][list(g.edges(ids[0]))[1][1]][0]['weight']

                        pts1 = g[ids[0]][list(g.edges(ids[0]))[0][1]][list(
                            g[ids[0]][list(g.edges(ids[0]))[0][1]].keys())[0]]['pts']
                        pts2 = g[ids[0]][list(g.edges(ids[0]))[1][1]][list(
                            g[ids[0]][list(g.edges(ids[0]))[1][1]].keys())[0]]['pts']
                        # if(flip):
                        if(distance([ids[1]['o'][0], ids[1]['o'][1]], [pts1[0][0], pts1[0][1]]) < distance([ids[1]['o'][0], ids[1]['o'][1]], [pts1[-1][0], pts1[-1][1]])):
                            pts1 = pts1[::-1]
                        # if(flip):
                        if(distance([ids[1]['o'][0], ids[1]['o'][1]], [pts2[0][0], pts2[0][1]]) > distance([ids[1]['o'][0], ids[1]['o'][1]], [pts2[-1][0], pts2[-1][1]])):
                            pts2 = pts2[::-1]

                        pps = np.append(
                            pts1, [[ids[1]['o'][0], ids[1]['o'][1]]], axis=0)
                        pps = np.append(pps, pts2, axis=0)
                        if(distance([(g.nodes(data=True))[list(g.edges(ids[0]))[0][1]]['o'][0], (g.nodes(data=True))[list(g.edges(ids[0]))[0][1]]['o'][1]], [pps[0][0], pps[0][1]]) > distance([(g.nodes(data=True))[list(g.edges(ids[0]))[0][1]]['o'][0], (g.nodes(data=True))[list(g.edges(ids[0]))[0][1]]['o'][1]], [pps[-1][0], pps[-1][1]])):  # if(flip):
                            pps = pps[::-1]
                        if(list(g.edges(ids[0]))[0][1] != list(g.edges(ids[0]))[1][1]):
                            g.add_edge(list(g.edges(ids[0]))[0][1], list(
                                g.edges(ids[0]))[1][1], weight=wei, pts=pps)
                        g.remove_node(ids[0])
                        cont_edge = cont_edge+1
                        break
            cont_edge = 1
            exa = []
            while(cont_edge > 0):
                cont_edge = 0
                for ids in list(g2.nodes(data=True)):
                    if len(g2.edges(ids[0])) == 2:
                        wei = g2[ids[0]][list(g2.edges(ids[0]))[
                            0][1]][0]['weight']+g2[ids[0]][list(g2.edges(ids[0]))[1][1]][0]['weight']
                        pts1 = g2[ids[0]][list(g2.edges(ids[0]))[0][1]][list(
                            g2[ids[0]][list(g2.edges(ids[0]))[0][1]].keys())[0]]['pts']
                        pts2 = g2[ids[0]][list(g2.edges(ids[0]))[1][1]][list(
                            g2[ids[0]][list(g2.edges(ids[0]))[1][1]].keys())[0]]['pts']
                        # if(flip):
                        if(distance([ids[1]['o'][0], ids[1]['o'][1]], [pts1[0][0], pts1[0][1]]) < distance([ids[1]['o'][0], ids[1]['o'][1]], [pts1[-1][0], pts1[-1][1]])):
                            pts1 = pts1[::-1]
                        # if(flip):
                        if(distance([ids[1]['o'][0], ids[1]['o'][1]], [pts2[0][0], pts2[0][1]]) > distance([ids[1]['o'][0], ids[1]['o'][1]], [pts2[-1][0], pts2[-1][1]])):
                            pts2 = pts2[::-1]

                        pps = np.append(
                            pts1, [[ids[1]['o'][0], ids[1]['o'][1]]], axis=0)
                        pps = np.append(pps, pts2, axis=0)
                        if(distance([(g2.nodes(data=True))[list(g2.edges(ids[0]))[0][1]]['o'][0], (g2.nodes(data=True))[list(g2.edges(ids[0]))[0][1]]['o'][1]], [pps[0][0], pps[0][1]]) > distance([(g2.nodes(data=True))[list(g2.edges(ids[0]))[0][1]]['o'][0], (g2.nodes(data=True))[list(g2.edges(ids[0]))[0][1]]['o'][1]], [pps[-1][0], pps[-1][1]])):  # if(flip):
                            pps = pps[::-1]
                        if(list(g2.edges(ids[0]))[0][1] != list(g2.edges(ids[0]))[1][1]):
                            g2.add_edge(list(g2.edges(ids[0]))[0][1], list(
                                g2.edges(ids[0]))[1][1], weight=wei, pts=pps)

                        g2.remove_node(ids[0])
                        cont_edge = cont_edge+1
                        break
            cont_edge = 1
            while(cont_edge > 0):
                cont_edge = 0
                for ids in list(g.nodes(data=True)):
                    if len(g.edges(ids[0])) < 1:
                        g.remove_node(ids[0])
                        cont_edge = cont_edge+1
                        break
                    if len(g.edges(ids[0])) == 1 and len(g[ids[0]][list(g.edges(ids[0]))[0][1]][list(g[ids[0]][list(g.edges(ids[0]))[0][1]].keys())[0]]['pts']) < 150:
                        g.remove_node(ids[0])
                        cont_edge = cont_edge+1
                        break
                for ids in list(g2.nodes(data=True)):
                    if len(g2.edges(ids[0])) < 1:
                        g2.remove_node(ids[0])
                        cont_edge = cont_edge+1
                        break
                    if len(g2.edges(ids[0])) == 1 and len(g2[ids[0]][list(g2.edges(ids[0]))[0][1]][list(g2[ids[0]][list(g2.edges(ids[0]))[0][1]].keys())[0]]['pts']) < 150:
                        g2.remove_node(ids[0])
                        cont_edge = cont_edge+1
                        break
            while(True):
                verts = []
                contvert = 0
                fromid = -1
                toid = -1
                fromid2 = -1
                toid2 = -1
                start = -1
                end = -1
                start2 = -1
                end2 = -1
                verts_worse = []
                worse_nodes = []
                worse_nodes2 = []
                checked = -1
                out_of_nodes = 0
                for ids in list(g.nodes(data=True)):
                    if(toid == -1 and ids[1]['o'][1] > 0 and ids[1]['o'][1] < int(framev.shape[1]*0.33) and ids[1]['o'][0] > int(framev.shape[0]*0.75) and ids[1]['o'][0] < int(framev.shape[0]*1)):  # 0.45
                        toid = ids[0]
                        end = [ids[1]['o'][0], ids[1]['o'][1]]
                        for ids2 in list(g.nodes(data=True)):
                            if(fromid == -1 and [ids2[0], toid] not in seen_nodes and ids2[1]['o'][1] > int(framev.shape[1]*0.66) and ids2[1]['o'][1] < int(framev.shape[1]*1) and ids2[1]['o'][0] > int(framev.shape[0]*0.75) and ids2[1]['o'][0] < int(framev.shape[0]*1)):
                                out_of_nodes = 1
                                fromid = ids2[0]
                                start = [ids2[1]['o'][0], ids2[1]['o'][1]]
                                seen_nodes.append([fromid, toid])
                            if(fromid != -1 and toid != -1 and checked == -1):
                                tt = 0
                                if(nx.has_path(g, fromid, toid)):
                                    tt = 1
                                    checked = 1
                                if((tt) == 0):
                                    fromid = -1
                                    checked = -1
                        if(checked == -1):
                            toid = -1
                    if(frame_falsev[ids[1]['o'][0], ids[1]['o'][1]] == 255):
                        worse_nodes.append(ids[0])
                        verts_worse.append([ids[1]['o'][0], ids[1]['o'][1]])
                if(out_of_nodes == 0):
                    line_thickness = line_thickness+line_thickness_add
                    if(line_thickness > 300 and max_line_thickness == 300 and subtract_houses == 0):
                        line_thickness = 25
                        max_line_thickness = 500
                        repeat = 1
                        break
                    if(line_thickness > 500 and subtract_houses == 0):
                        subtract_houses = 1
                        line_thickness = 25
                        max_line_thickness = 300
                        repeat = 1
                        break
                    if(line_thickness > 300 and subtract_houses > 0):
                        if(subtract_houses == 2):
                            repeat = 0
                            break
                        repeat = 1
                        subtract_houses = 2
                        line_thickness = 25
                        break
                    repeat = 1
                    break
                if(fromid == -1 or toid == -1):
                    continue
                dist10 = 0
                dist11 = 0
                dist20 = 0
                dist21 = 0
                checked = -1
                for ids in list(g2.nodes(data=True)):
                    dist10t = ids[1]['o'][0]-(end[0]-int(framev.shape[0]*0.5))
                    dist11t = ids[1]['o'][1]-(end[1])
                    dist20t = ids[1]['o'][0] - \
                        (start[0]-int(framev.shape[0]*0.5))
                    dist21t = ids[1]['o'][1]-(start[1])
                    if(toid2 == -1 and abs(dist10t) <= 2 and abs(dist11t) <= 2):
                        toid2 = ids[0]
                        end2 = [ids[1]['o'][0], ids[1]['o'][1]]
                        dist10 = dist10t
                        dist11 = dist11t
                    if(fromid2 == -1 and abs(dist20t) <= 2 and abs(dist21t) <= 2):
                        fromid2 = ids[0]
                        start2 = [ids[1]['o'][0], ids[1]['o'][1]]
                        dist20 = dist20t
                        dist21 = dist21t
                    if(fromid2 != -1 and toid2 != -1 and checked == -1):
                        tt = 0
                        if(nx.has_path(g2, fromid2, toid2)):
                            tt = 1
                        checked = 1
                        if((tt) == 0):
                            fromid2 = -1
                            toid2 = -1
                            checked = -1
                    if(frame_falsev2[ids[1]['o'][0], ids[1]['o'][1]] == 255):
                        worse_nodes2.append(ids[0])
                verts = []
                if(fromid2 == -1 or toid2 == -1):
                    continue
                break
            if(repeat == 1):
                continue
            blob = np.zeros_like(framev)
            nodes_output = []
            edges_output = []
            all_output = []
            best_path = None
            num_bad_nodes = 1000
            best_len = 10000
            edges = g.edges()
            cont_path = 0
            start_time_path = time.time()
            for path in nx.all_shortest_paths(g, source=fromid, target=toid):
                verts = []
                cont_path = cont_path+1
                if(cont_path > 2000 or (cont_path % 2 == 0 and (time.time() - start_time_path) > 600)):
                    break
                temp = 0
                for x in range(len(path)-1):
                    if (path[x] in worse_nodes and path[x+1] in worse_nodes):
                        temp = temp+1
                if(temp < num_bad_nodes or (temp == num_bad_nodes and len(path) < best_len)):
                    num_bad_nodes = temp
                    best_path = path
                    best_len = len(path)
            best_path2 = None
            num_bad_nodes2 = 1000
            best_len = 10000
            edges = g2.edges()
            cont_path = 0
            start_time_path = time.time()
            for path2 in nx.all_shortest_paths(g2, source=fromid2, target=toid2):
                cont_path = cont_path+1
                if(cont_path > 2000 or (cont_path % 2 == 0 and (time.time() - start_time_path) > 600)):
                    break
                temp = 0
                for x in range(len(path2)-1):
                    if (path2[x] in worse_nodes2 and path2[x+1] in worse_nodes2):
                        temp = temp+1
                if(temp < num_bad_nodes2 or (temp == num_bad_nodes2 and len(path2) < best_len)):
                    num_bad_nodes2 = temp
                    best_path2 = path2
                    best_len = len(path2)
            path_points = []
            nod = list(g.nodes(data=True))
            for pa in best_path:
                ids = [tt for tt in nod if tt[0] == pa][0]
                path_points.append([int((min(dims[0])+int(ids[1]['o'][0]*resize_scale0))),
                                    int((min(dims[1])+int(ids[1]['o'][1]*resize_scale1)))])
            path_points2 = []
            nod = list(g2.nodes(data=True))
            for pa in best_path2:
                ids = [tt for tt in nod if tt[0] == pa][0]
                path_points2.append([int((min(dims2[0])+int((ids[1]['o'][0]+dist10)*resize_scale20))), int(
                    (min(dims2[1])+int((ids[1]['o'][1]+dist11)*resize_scale21)))])
            pco = 0
            index1 = 0
            index2 = 0
            for x in path_points:
                pco = pco+1
                dists = [distance(x, z) for z in path_points2]
                if(min(dists) < 2 and pco < len(path_points)/2):
                    index1 = path_points.index(x)
                    index2 = dists.index(min(dists))
            pco = 0
            index3 = len(path_points)
            index4 = len(path_points2)
            for x in path_points:
                pco = pco+1
                dists = [distance(x, z) for z in path_points2]
                if(min(dists) < 2 and pco > len(path_points)/2):
                    index3 = path_points.index(x)+1
                    index4 = dists.index(min(dists))+1
                    break
            best_path = best_path[index1:index3]
            best_path2 = best_path2[index2:index4]
            verts = []
            horizontal_shift = 0
            vertical_shift = 0
            if(True):
                path = best_path
                nod = list(g.nodes(data=True))
                for pa in path:
                    ids = [tt for tt in nod if tt[0] == pa][0]
                    verts.append([ids[1]['o'][0], ids[1]['o'][1]])
                    blob[min(dims[0])+int(ids[1]['o'][0]*resize_scale0)
                         ][min(dims[1])+int(ids[1]['o'][1]*resize_scale1)] = 255
                    nodes_output.append([ids[1]['o'][0], ids[1]['o'][1]])
                edges = g.edges()
                for x in range(len(path)-1):
                    ids = [tt for tt in nod if tt[0] == path[x]][0]
                    if(len(all_output) == 0 or all_output[-1] != [int((min(dims[1])+int(ids[1]['o'][1]*resize_scale1))-horizontal_shift), int((min(dims[0])+int(ids[1]['o'][0]*resize_scale0))-vertical_shift)]):
                        all_output.append([int((min(dims[1])+int(ids[1]['o'][1]*resize_scale1))-horizontal_shift), int((min(dims[0])+int(
                            ids[1]['o'][0]*resize_scale0))-vertical_shift)])  # all_output.append([ids[1]['o'][0],ids[1]['o'][1]])
                    flip = False
                    if (path[x], path[x+1]) in edges:
                        vals = flatten(
                            [[v] for v in g[path[x]][path[x+1]].values()])
                    else:
                        vals = flatten(
                            [[v] for v in g[path[x+1]][path[x]].values()])
                        flip = True
                    for val in vals:
                        ps = val.get('pts', [])
                        if(distance([ids[1]['o'][0], ids[1]['o'][1]], [ps[0][0], ps[0][1]]) > distance([ids[1]['o'][0], ids[1]['o'][1]], [ps[-1][0], ps[-1][1]])):
                            ps = ps[::-1]
                        for po in ps:
                            blob[min(dims[0])+int(po[0]*resize_scale0)
                                 ][min(dims[1])+int(po[1]*resize_scale1)] = 255
                            edges_output.append([po[0], po[1]])
                            if(len(all_output) == 0 or all_output[-1] != [int((min(dims[1])+int(po[1]*resize_scale1))-horizontal_shift), int((min(dims[0])+int(po[0]*resize_scale0))-vertical_shift)]):
                                all_output.append([int((min(dims[1])+int(po[1]*resize_scale1))-horizontal_shift), int(
                                    (min(dims[0])+int(po[0]*resize_scale0))-vertical_shift)])
                        break
                ids = [tt for tt in nod if tt[0] == path[-1]][0]
                if(len(all_output) == 0 or all_output[-1] != [int((min(dims[1])+int(ids[1]['o'][1]*resize_scale1))-horizontal_shift), int((min(dims[0])+int(ids[1]['o'][0]*resize_scale0))-vertical_shift)]):
                    all_output.append([int((min(dims[1])+int(ids[1]['o'][1]*resize_scale1))-horizontal_shift), int((min(dims[0])+int(
                        ids[1]['o'][0]*resize_scale0))-vertical_shift)])  # all_output.append([ids[1]['o'][0],ids[1]['o'][1]])
            if(len(verts) == 0):
                print("NO PATH")
                continue
            nodes_output = nodes_output[::-1]
            edges_output = edges_output[::-1]
            all_output = all_output[::-1]
            verts2 = []
            if True:
                path2 = best_path2
                nod = list(g2.nodes(data=True))
                for pa in path2:
                    ids = [tt for tt in nod if tt[0] == pa][0]
                    verts2.append([ids[1]['o'][0], ids[1]['o'][1]])
                    blob[min(dims2[0])+int(ids[1]['o'][0]*resize_scale20)
                         ][min(dims2[1])+int(ids[1]['o'][1]*resize_scale21)] = 255
                    nodes_output.append([ids[1]['o'][0], ids[1]['o'][1]])
                edges = g2.edges()
                for x in range(len(path2)-1):
                    ids = [tt for tt in nod if tt[0] == path2[x]][0]
                    if(len(all_output) == 0 or all_output[-1] != [int((min(dims2[1])+int((ids[1]['o'][1]+dist11)*resize_scale21))-horizontal_shift), int((min(dims2[0])+int((ids[1]['o'][0]+dist10)*resize_scale20))-vertical_shift)]):
                        all_output.append([int((min(dims2[1])+int((ids[1]['o'][1]+dist11)*resize_scale21))-horizontal_shift), int((min(dims2[0])+int(
                            (ids[1]['o'][0]+dist10)*resize_scale20))-vertical_shift)])  # all_output.append([ids[1]['o'][0],ids[1]['o'][1]])
                    flip = False
                    if (path2[x], path2[x+1]) in edges:
                        vals = flatten(
                            [[v] for v in g2[path2[x]][path2[x+1]].values()])
                    else:
                        vals = flatten(
                            [[v] for v in g2[path2[x+1]][path2[x]].values()])
                        flip = True
                    for val in vals:
                        ps = val.get('pts', [])
                        if(distance([ids[1]['o'][0], ids[1]['o'][1]], [ps[0][0], ps[0][1]]) > distance([ids[1]['o'][0], ids[1]['o'][1]], [ps[-1][0], ps[-1][1]])):
                            ps = ps[::-1]
                        for po in ps:
                            blob[min(dims2[0])+int(po[0]*resize_scale20)
                                 ][min(dims2[1])+int(po[1]*resize_scale21)] = 255
                            edges_output.append([po[0], po[1]])
                            if(len(all_output) == 0 or all_output[-1] != [int((min(dims2[1])+int((po[1]+dist11)*resize_scale21))-horizontal_shift), int((min(dims2[0])+int((po[0]+dist10)*resize_scale20))-vertical_shift)]):
                                all_output.append([int((min(dims2[1])+int((po[1]+dist11)*resize_scale21))-horizontal_shift), int(
                                    (min(dims2[0])+int((po[0]+dist10)*resize_scale20))-vertical_shift)])
                        break
                ids = [tt for tt in nod if tt[0] == path2[-1]][0]
                if(len(all_output) == 0 or all_output[-1] != [int((min(dims2[1])+int((ids[1]['o'][1]+dist11)*resize_scale21))-horizontal_shift), int((min(dims2[0])+int((ids[1]['o'][0]+dist10)*resize_scale20))-vertical_shift)]):
                    all_output.append([int((min(dims2[1])+int((ids[1]['o'][1]+dist11)*resize_scale21))-horizontal_shift), int(
                        (min(dims2[0])+int((ids[1]['o'][0]+dist10)*resize_scale20))-vertical_shift)])
            if(len(verts2) == 0):
                print("NO PATH")
                continue
            rem_dup = 1
            while rem_dup == 1:
                rem_dup = 0
                for x in range(len(all_output)):
                    if all_output[x] in all_output[x+1:]:
                        if(len(all_output[:x]+all_output[len(all_output)-all_output[::-1].index(all_output[x])-1:]) < 1000):
                            continue
                        all_output = all_output[:x]+all_output[len(
                            all_output)-all_output[::-1].index(all_output[x])-1:]
                        rem_dup = 1
                        break
            temp_mask = np.zeros(
                (int(framev.shape[0]), int(framev.shape[0])), np.uint8)
            for x in all_output[::-1]:
                temp_mask[int(x[1]), int(x[0])] = 255
            all_output_save = [[x[0]+int((cropped_y*(float(blob.shape[0])/float(dimxl[0])))), x[1]+int(
                (cropped_x*(float(blob.shape[0])/float(dimxl[0]))))] for x in all_output]
            f = open(args.save_dir+'mosaic_mask_'+args.dataset +
                     "_"+os.path.basename(im_fn_orig)+'.txt', "w+")
            f.write("Size:\r\n")
            f.write(str(blob.shape[0]+int((2*cropped_x*(float(blob.shape[0])/float(dimxl[0])))))+","+str(
                blob.shape[1]+int((2*cropped_y*(float(blob.shape[0])/float(dimxl[0]))))))
            f.write("\r\n")
            f.write("Number of nodes:\r\n")
            f.write(str(len(all_output)))
            f.write("\r\n")
            f.write("Nodes:\r\n")
            f.write(str(all_output_save))
            f.write("\r\n")
            f.close()
            del all_output_save
            del nodes_output
            del edges_output
            del temp_mask
            del blob
        del guide_maskv
        del frame_falsev
        del frame_falsev2
        del frame_falsev_m
        del frame_falsev_eroded
        del frame_falsev_guide
        del framev
        del framev2
        del backtorgb2
        del house_cropped
        del real_img
        del houses_orig
        gc.collect()


if __name__ == '__main__':
    main()
