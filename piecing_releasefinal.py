from __future__ import print_function
from scipy.spatial import Voronoi, voronoi_plot_2d
import shapefile
from scipy import ndimage
import shutil
import math
import networkx as nx
from shapely.geometry import Point, LineString
import subprocess
import glob
import json
import numpy as np
import datetime
from PIL import Image

import argparse
from datetime import datetime
from matplotlib.pylab import plt
import os
import gc
import sys
import time
from time import sleep
import cv2
import PIL
PIL.Image.MAX_IMAGE_PIXELS = 933120000


def check_bounds(mosaic_bw, dims, proc_imgs, current_upper_x, current_upper_y, seen, tfws, xwind, ywind):
        width = tfws[proc_imgs][1]
        height = tfws[proc_imgs][0]
        if(proc_imgs in seen):
           return False
        pixel_size = tfws[proc_imgs][4]
        left_upper_x = tfws[proc_imgs][3]
        left_upper_y = tfws[proc_imgs][2]
        x_disp = round(((current_upper_x-left_upper_x)/pixel_size))
        y_disp = round(((left_upper_y-current_upper_y)/pixel_size))
        dimxmin = ywind+min(dims[0])
        dimxmax = ywind+max(dims[0])
        dimymin = xwind+min(dims[1])
        dimymax = xwind+max(dims[1])
        bounds = True
        if(dimxmin < x_disp+(height*0.1) or dimxmax > int(x_disp)+(height*0.9) or dimymin < y_disp+(width*0.1) or dimymax > int(y_disp)+(width*0.9)):
              bounds = False
        seen.append(proc_imgs)
        return bounds


def remove_singlets(temp_mask, mosaic_bw, filenames, current_upper_x, current_upper_y, tfws, fill_dict, x_disp, y_disp, completed_images, drawerr):
        repeat = 0
        temp_mask_ret = np.zeros(
            (temp_mask.shape[0], temp_mask.shape[1]), np.uint8)
        threes = np.where(temp_mask > 0)
        dimxmin = min(threes[0])
        dimxmax = max(threes[0])
        dimymin = min(threes[1])
        dimymax = max(threes[1])
        blobcolor = temp_mask[max(dimxmin-1, 0):min(dimxmax+1, temp_mask_ret.shape[0]),
                                  max(dimymin-1, 0):min(dimymax+1, temp_mask_ret.shape[1])]
        h, w = blobcolor.shape
        seed = (int(w/2), int(h/2))
        mask = np.zeros((h+2, w+2), np.uint8)
        floodflags = 4
        floodflags |= cv2.FLOODFILL_MASK_ONLY
        floodflags |= (255 << 8)
        num, im, mask, rect = cv2.floodFill(
            blobcolor, mask, seed, 255, 0, 0, floodflags)
        mask = mask[1:-1, 1:-1]
        temp_mask_ret[max(dimxmin-1, 0):min(dimxmax+1, temp_mask_ret.shape[0]),
                          max(dimymin-1, 0):min(dimymax+1, temp_mask_ret.shape[1])][mask > 0] = 255
        for x in range(len(threes[0])):
           if(neighbors8_ex(temp_mask_ret, threes[0][x], threes[1][x]) > 0):
             temp_mask[threes[0][x]][threes[1][x]] = 254
        temp_mask_copy = np.zeros((mask.shape[0], mask.shape[1]), np.uint8)
        temp_mask_copy[temp_mask[max(dimxmin-1, 0):min(dimxmax+1, temp_mask_ret.shape[0]), max(
            dimymin-1, 0):min(dimymax+1, temp_mask_ret.shape[1])] == 255] = 255
        ret, labels = cv2.connectedComponents(temp_mask_copy)
        cont_lab = 0
        iterelem = np.array(np.unique(labels, return_counts=True)).T
        # >1
        if (len(iterelem) > 1 and (max(iterelem[1:, 1]) > 2 or (drawerr < 2 and max(iterelem[1:, 1]) > 1))):
         for lab, counts in iterelem:
           if(lab != 0):
              cont_lab = 1
              seen = []
              dims = np.where(labels == lab)
              filled = 0
              for filcont in range(4):
                if filled == 1:
                   break
                if(filcont == 0):
                  dimy = x_disp+max(dimxmin-1, 0)+dims[0][np.argmin(dims[0])]
                  dimx = y_disp+max(dimymin-1, 0)+dims[1][np.argmin(dims[0])]
                if(filcont == 1):
                  dimy = x_disp+max(dimxmin-1, 0)+dims[0][np.argmax(dims[0])]
                  dimx = y_disp+max(dimymin-1, 0)+dims[1][np.argmax(dims[0])]
                if(filcont == 2):
                  dimy = x_disp+max(dimxmin-1, 0)+dims[0][np.argmin(dims[1])]
                  dimx = y_disp+max(dimymin-1, 0)+dims[1][np.argmin(dims[1])]
                if(filcont == 3):
                  dimy = x_disp+max(dimxmin-1, 0)+dims[0][np.argmax(dims[1])]
                  dimx = y_disp+max(dimymin-1, 0)+dims[1][np.argmax(dims[1])]
                for dircont in range(8):
                 if(dircont == 0):
                  dimy2 = dimy-1
                  dimx2 = dimx-1
                 if(dircont == 1):
                  dimy2 = dimy
                  dimx2 = dimx-1
                 if(dircont == 2):
                  dimy2 = dimy-1
                  dimx2 = dimx
                 if(dircont == 3):
                  dimy2 = dimy+1
                  dimx2 = dimx+1
                 if(dircont == 4):
                  dimy2 = dimy
                  dimx2 = dimx+1
                 if(dircont == 5):
                  dimy2 = dimy+1
                  dimx2 = dimx
                 if(dircont == 6):
                  dimy2 = dimy-1
                  dimx2 = dimx+1
                 if(dircont == 7):
                  dimy2 = dimy+1
                  dimx2 = dimx-1
                 if(mosaic_bw[dimy2][dimx2] != 0 and (counts < 2 or (fill_dict[dimy][dimx] is None or fill_dict[dimy][dimx][mosaic_bw[dimy2][dimx2]-1] is None or counts not in fill_dict[dimy][dimx][mosaic_bw[dimy2][dimx2]-1])) and (counts < 1000 or check_bounds(mosaic_bw, dims, mosaic_bw[dimy2][dimx2], current_upper_x, current_upper_y, seen, tfws, y_disp+max(dimymin-1, 0), x_disp+max(dimxmin-1, 0)))):
                  mosaic_bw[int(x_disp):int(x_disp)+temp_mask_ret.shape[0], int(y_disp):int(y_disp)+temp_mask_ret.shape[1]][max(dimxmin-1, 0):min(
                      dimxmax+1, temp_mask_ret.shape[0]), max(dimymin-1, 0):min(dimymax+1, temp_mask_ret.shape[1])][dims] = mosaic_bw[dimy2][dimx2]
                  if filenames[mosaic_bw[dimy2][dimx2]-1] in completed_images:
                     completed_images.remove(
                         filenames[mosaic_bw[dimy2][dimx2]-1])
                     repeat = 1
                  if(fill_dict[dimy][dimx] is None):
                     fill_dict[dimy][dimx] = [
                         None for xn in range(len(filenames))]
                     fill_dict[dimy][dimx][mosaic_bw[dimy2]
                         [dimx2]-1] = [counts]
                  else:
                     if(fill_dict[dimy][dimx][mosaic_bw[dimy2][dimx2]-1] is None):
                        fill_dict[dimy][dimx][mosaic_bw[dimy2]
                            [dimx2]-1] = [counts]
                     else:
                        if counts not in fill_dict[dimy][dimx][mosaic_bw[dimy2][dimx2]-1]:
                          fill_dict[dimy][dimx][mosaic_bw[dimy2]
                              [dimx2]-1].append(counts)
                        else:
                          seen.append(mosaic_bw[dimy2][dimx2])
                          continue
                  filled = 1
                  break
        if(cont_lab == 0):
           temp_mask[temp_mask > 0] = 255
           return temp_mask, False, None, repeat
        else:
           temp_mask_ret[temp_mask == 254] = 255
           return temp_mask, True, temp_mask_ret, repeat


def remove_threes(temp_mask, temp_maskall, temp_maskedges, proc_imgs, mosaic_bw, filenames, completed_images, x_disp, y_disp, drawerr, fill_dict, first_col, prev_im, nex_im, x_disp1, y_disp1):
    repeat = 0
    threes = np.array(np.where(temp_mask > 0))
    block = np.zeros(temp_mask.shape, np.bool)
    extras = []
    for x in range(len(threes[0])):
      if(neighbors8(temp_mask, threes[0][x], threes[1][x]) == 1):
            temp_mask[threes[0][x]][threes[1][x]] = 0
            dimx = y_disp+threes[1][x]
            dimy = x_disp+threes[0][x]
            if(drawerr != 2 and mosaic_bw[dimy][dimx] == proc_imgs):
             proc_imgs2 = max(mosaic_bw[dimy][dimx+1] if mosaic_bw[dimy][dimx+1] != proc_imgs else 0, mosaic_bw[dimy][dimx-1] if mosaic_bw[dimy][dimx-1] != proc_imgs else 0,
                              mosaic_bw[dimy-1][dimx] if mosaic_bw[dimy-1][dimx] != proc_imgs else 0, mosaic_bw[dimy+1][dimx] if mosaic_bw[dimy+1][dimx] != proc_imgs else 0)
             if(proc_imgs2 != 0):
              if(fill_dict[dimy][dimx] is None):
                     fill_dict[dimy][dimx] = [
                         None for xn in range(len(filenames))]
                     fill_dict[dimy][dimx][proc_imgs2-1] = [1]
              else:
                     if(fill_dict[dimy][dimx][proc_imgs2-1] is None):
                        fill_dict[dimy][dimx][proc_imgs2-1] = [1]
                     else:
                        if 1 not in fill_dict[dimy][dimx][proc_imgs2-1]:
                           fill_dict[dimy][dimx][proc_imgs2-1].append(1)
                        else:
                           proc_imgs2 = max(mosaic_bw[dimy][dimx+1] if mosaic_bw[dimy][dimx+1] != 0 and mosaic_bw[dimy][dimx+1] != proc_imgs2 and (fill_dict[dimy][dimx][mosaic_bw[dimy][dimx+1]-1] is None or 1 not in fill_dict[dimy][dimx][mosaic_bw[dimy][dimx+1]-1]) else 0, mosaic_bw[dimy][dimx-1] if mosaic_bw[dimy][dimx-1] != 0 and mosaic_bw[dimy][dimx-1] != proc_imgs2 and (fill_dict[dimy][dimx][mosaic_bw[dimy][dimx-1]-1] is None or 1 not in fill_dict[dimy][dimx][mosaic_bw[dimy][dimx-1]-1])
                                            else 0, mosaic_bw[dimy-1][dimx] if mosaic_bw[dimy-1][dimx] != 0 and mosaic_bw[dimy-1][dimx] != proc_imgs2 and (fill_dict[dimy][dimx][mosaic_bw[dimy-1][dimx]-1] is None or 1 not in fill_dict[dimy][dimx][mosaic_bw[dimy-1][dimx]-1]) else 0, mosaic_bw[dimy+1][dimx] if mosaic_bw[dimy+1][dimx] != 0 and mosaic_bw[dimy+1][dimx] != proc_imgs2 and (fill_dict[dimy][dimx][mosaic_bw[dimy+1][dimx]-1] is None or 1 not in fill_dict[dimy][dimx][mosaic_bw[dimy+1][dimx]-1]) else 0)
                           if(proc_imgs2 != 0 and proc_imgs2 != proc_imgs):
                             if(fill_dict[dimy][dimx][proc_imgs2-1] is None):
                               fill_dict[dimy][dimx][proc_imgs2-1] = [1]
                             else:
                               fill_dict[dimy][dimx][proc_imgs2-1].append(1)
             mosaic_bw[dimy][dimx] = proc_imgs2
             if proc_imgs2 != 0 and filenames[proc_imgs2-1] in completed_images:
               completed_images = [
                   ci for ci in completed_images if ci is not filenames[proc_imgs2-1]]
               repeat = 1
            threes[:, x] = -1
            continue
    for x in range(len(threes[0])):
      for y in range(4):
        # XXX
        #  0
        if y == 0:
           dimy1 = threes[0][x]-1
           dimx1 = threes[1][x]-1
           dimy2 = threes[0][x]-1
           dimx2 = threes[1][x]+1
           dimy3 = threes[0][x]-1
           dimx3 = threes[1][x]
           dimy4 = threes[0][x]
           dimx4 = threes[1][x]-1
           dimy5 = threes[0][x]
           dimx5 = threes[1][x]+1
        #   X
        #   X0
        #   X
        if y == 1:
           dimy1 = threes[0][x]-1
           dimx1 = threes[1][x]-1
           dimy2 = threes[0][x]+1
           dimx2 = threes[1][x]-1
           dimy3 = threes[0][x]
           dimx3 = threes[1][x]-1
           dimy4 = threes[0][x]-1
           dimx4 = threes[1][x]
           dimy5 = threes[0][x]+1
           dimx5 = threes[1][x]
        #   X
        #  0X
        #   X
        if y == 2:
           dimy1 = threes[0][x]+1
           dimx1 = threes[1][x]+1
           dimy2 = threes[0][x]-1
           dimx2 = threes[1][x]+1
           dimy3 = threes[0][x]
           dimx3 = threes[1][x]+1
           dimy4 = threes[0][x]-1
           dimx4 = threes[1][x]
           dimy5 = threes[0][x]+1
           dimx5 = threes[1][x]
        #  0
        # XXX
        if y == 3:
           dimy1 = threes[0][x]+1
           dimx1 = threes[1][x]+1
           dimy2 = threes[0][x]+1
           dimx2 = threes[1][x]-1
           dimy3 = threes[0][x]+1
           dimx3 = threes[1][x]
           dimy4 = threes[0][x]
           dimx4 = threes[1][x]-1
           dimy5 = threes[0][x]
           dimx5 = threes[1][x]+1
        if(temp_mask[dimy1][dimx1] > 0 and temp_mask[dimy2][dimx2] > 0 and (neighbors8(temp_mask, threes[0][x], threes[1][x]) == 2 or (neighbors8(temp_mask, threes[0][x], threes[1][x]) == 3 and temp_mask[dimy3][dimx3] > 0))):

              if((temp_maskedges is not None and temp_maskedges[dimy4][dimx4] >= proc_imgs and temp_maskedges[dimy5][dimx5] >= proc_imgs)):
                 if(temp_mask[dimy4][dimx4] == 0):
                    temp_mask[dimy4][dimx4] = 255
                    threes = np.append(threes, [[dimy4], [dimx4]], axis=1)
                 if(temp_mask[dimy5][dimx5] == 0):
                    temp_mask[dimy5][dimx5] = 255
                    threes = np.append(threes, [[dimy5], [dimx5]], axis=1)
                 block[threes[0][x]][threes[1][x]] = True
                 continue
              if((mosaic_bw[int(x_disp):int(x_disp)+temp_mask.shape[0], int(y_disp):int(y_disp)+temp_mask.shape[1]][dimy5][dimx5] == 0 and mosaic_bw[int(x_disp):int(x_disp)+temp_mask.shape[0], int(y_disp):int(y_disp)+temp_mask.shape[1]][dimy4][dimx4] == 0 and temp_mask[dimy3][dimx3] == 0) or (mosaic_bw[int(x_disp):int(x_disp)+temp_mask.shape[0], int(y_disp):int(y_disp)+temp_mask.shape[1]][dimy5][dimx5] == proc_imgs and mosaic_bw[int(x_disp):int(x_disp)+temp_mask.shape[0], int(y_disp):int(y_disp)+temp_mask.shape[1]][dimy4][dimx4] == proc_imgs and temp_mask[dimy3][dimx3] == 0)):
                 if(temp_mask[dimy4][dimx4] == 0):
                    temp_mask[dimy4][dimx4] = 255
                    threes = np.append(threes, [[dimy4], [dimx4]], axis=1)
                 if(temp_mask[dimy5][dimx5] == 0):
                   temp_mask[dimy5][dimx5] = 255
                   threes= np.append(threes, [[dimy5], [dimx5]], axis=1)
                 block[threes[0][x]][threes[1][x]] = True
                 continue
              if(temp_mask[dimy3][dimx3] == 0):
                 temp_mask[dimy3][dimx3] = 255
                 threes= np.append(threes, [[dimy3], [dimx3]], axis=1)
              temp_mask[threes[0][x]][threes[1][x]] = 0
              dimx = y_disp+threes[1][x]
              dimy = x_disp+threes[0][x]
              if(mosaic_bw[dimy][dimx] != proc_imgs):
                 continue
              if(drawerr != 2 and mosaic_bw[dimy][dimx] == proc_imgs):
               proc_imgs2= max(mosaic_bw[dimy][dimx+1] if mosaic_bw[dimy][dimx+1] != proc_imgs else 0, mosaic_bw[dimy][dimx-1] if mosaic_bw[dimy][dimx-1] != proc_imgs else 0, mosaic_bw[dimy-1][dimx] if mosaic_bw[dimy-1][dimx] != proc_imgs else 0, mosaic_bw[dimy+1][dimx] if mosaic_bw[dimy+1][dimx] != proc_imgs else 0)
               if(proc_imgs2 != 0):
                if(fill_dict[dimy][dimx] is None):
                     fill_dict[dimy][dimx] = [None for xn in range(len(filenames))]
                     fill_dict[dimy][dimx][proc_imgs2-1] = [1]
                else:
                     if(fill_dict[dimy][dimx][proc_imgs2-1] is None):
                        fill_dict[dimy][dimx][proc_imgs2-1] = [1]
                     else:
                        if 1 not in fill_dict[dimy][dimx][proc_imgs2-1]:
                           fill_dict[dimy][dimx][proc_imgs2-1].append(1)
                        else:
                           proc_imgs2 = max(mosaic_bw[dimy][dimx+1] if mosaic_bw[dimy][dimx+1] != 0 and mosaic_bw[dimy][dimx+1] != proc_imgs2 and (fill_dict[dimy][dimx][mosaic_bw[dimy][dimx+1]-1] is None or 1 not in fill_dict[dimy][dimx][mosaic_bw[dimy][dimx+1]-1]) else 0, mosaic_bw[dimy][dimx-1] if mosaic_bw[dimy][dimx-1] != 0 and mosaic_bw[dimy][dimx-1] != proc_imgs2 and (fill_dict[dimy][dimx][mosaic_bw[dimy][dimx-1]-1] is None or 1 not in fill_dict[dimy][dimx][mosaic_bw[dimy][dimx-1]-1]) else 0, mosaic_bw[dimy-1][dimx] if mosaic_bw[dimy-1][dimx] != 0 and mosaic_bw[dimy-1][dimx] != proc_imgs2 and (fill_dict[dimy][dimx][mosaic_bw[dimy-1][dimx]-1] is None or 1 not in fill_dict[dimy][dimx][mosaic_bw[dimy-1][dimx]-1]) else 0, mosaic_bw[dimy+1][dimx] if mosaic_bw[dimy+1][dimx] != 0 and mosaic_bw[dimy+1][dimx] != proc_imgs2 and (fill_dict[dimy][dimx][mosaic_bw[dimy+1][dimx]-1] is None or 1 not in fill_dict[dimy][dimx][mosaic_bw[dimy+1][dimx]-1]) else 0)
                           if(proc_imgs2 != 0 and proc_imgs2 != proc_imgs):
                             if(fill_dict[dimy][dimx][proc_imgs2-1] is None):
                               fill_dict[dimy][dimx][proc_imgs2-1] = [1]
                             else:
                               fill_dict[dimy][dimx][proc_imgs2-1].append(1)
               mosaic_bw[dimy][dimx] = proc_imgs2
               if proc_imgs2 != 0 and filenames[proc_imgs2-1] in completed_images:
                 completed_images= [ci for ci in completed_images if ci is not filenames[proc_imgs2-1]]
                 repeat = 1
              threes[:, x]= -1
              continue
    for x in range(len(threes[0])):
      if(threes[0][x] == -1):
           continue
      for y in range(2):
        if y == 0:
           dimy1 = threes[0][x]-1
           dimx1 = threes[1][x]+1
           dimy2 = threes[0][x]-1
           dimx2 = threes[1][x]
           dimy3 = threes[0][x]
           dimx3 = threes[1][x]+1
           dimy4 = threes[0][x]+1
           dimx4 = threes[1][x]
           dimy5 = threes[0][x]-1
           dimx5 = threes[1][x]+2
        if y == 1:
           dimy1 = threes[0][x]-1
           dimx1 = threes[1][x]-1
           dimy2 = threes[0][x]-1
           dimx2 = threes[1][x]
           dimy3 = threes[0][x]
           dimx3 = threes[1][x]-1
           dimy4 = threes[0][x]+1
           dimx4 = threes[1][x]
           dimy5 = threes[0][x]-1
           dimx5 = threes[1][x]-2

        if(temp_mask[dimy1][dimx1] > 0 and temp_mask[dimy2][dimx2] == 0 and temp_mask[dimy3][dimx3] == 0):
               if(temp_maskedges is not None and temp_maskedges[dimy2][dimx2] > temp_maskedges[dimy3][dimx3] and temp_maskedges[dimy2][dimx2] >= proc_imgs):
                  extras.append([dimy2, dimx2, threes[0][x],
                                threes[1][x], dimy1, dimx1])
                  continue
               else:
                  if(temp_maskedges is not None and temp_maskedges[dimy3][dimx3] >= proc_imgs):
                     extras.append([dimy3, dimx3, threes[0][x],
                                   threes[1][x], dimy1, dimx1])
                     continue
                  else:
                    nei2 = neighbors8_ex(temp_mask, dimy3, dimx3)
                    if(nei2 >= 2 and nei2 <= 4 and first_col and (mosaic_bw[int(x_disp):int(x_disp)+temp_mask.shape[0], int(y_disp):int(y_disp)+temp_mask.shape[1]][dimy2][dimx2] == prev_im or mosaic_bw[int(x_disp):int(x_disp)+temp_mask.shape[0], int(y_disp):int(y_disp)+temp_mask.shape[1]][dimy2][dimx2] == nex_im or mosaic_bw[int(x_disp):int(x_disp)+temp_mask.shape[0], int(y_disp):int(y_disp)+temp_mask.shape[1]][dimy2][dimx2] == 0)):
                      extras.append([dimy3, dimx3, threes[0][x],
                                    threes[1][x], dimy1, dimx1])
                      block[dimy1][dimx1] = True
                      block[threes[0][x]][threes[1][x]] = True
                      continue
                    nei1 = neighbors8_ex(temp_mask, dimy2, dimx2)
                    if(nei1 >= 2 and nei1 <= 4 and first_col and (mosaic_bw[int(x_disp):int(x_disp)+temp_mask.shape[0], int(y_disp):int(y_disp)+temp_mask.shape[1]][dimy3][dimx3] == prev_im or mosaic_bw[int(x_disp):int(x_disp)+temp_mask.shape[0], int(y_disp):int(y_disp)+temp_mask.shape[1]][dimy3][dimx3] == nex_im or mosaic_bw[int(x_disp):int(x_disp)+temp_mask.shape[0], int(y_disp):int(y_disp)+temp_mask.shape[1]][dimy3][dimx3] == 0)):
                        extras.append(
                            [dimy2, dimx2, threes[0][x], threes[1][x], dimy1, dimx1])
                        block[dimy1][dimx1] = True
                        block[threes[0][x]][threes[1][x]] = True
                        continue
                    if(nei1 == 2):
                      temp_mask[dimy2][dimx2] = 255
                      threes= np.append(threes, [[dimy2], [dimx2]], axis=1)
                      continue
                    if(nei2 == 2):
                      temp_mask[dimy3][dimx3] = 255
                      threes= np.append(threes, [[dimy3], [dimx3]], axis=1)
                      continue
                    if(nei1 == nei2 and nei1 == 3):
                      if(temp_maskall[dimy2][dimx2] == 255):
                          temp_mask[dimy2][dimx2] = 255
                          threes= np.append(threes, [[dimy2], [dimx2]], axis=1)
                          continue
                      if(temp_maskall[dimy3][dimx3] == 255):
                          temp_mask[dimy3][dimx3] = 255
                          threes= np.append(threes, [[dimy3], [dimx3]], axis=1)
                          continue
                    if(nei1 == 3):
                      extras.append([dimy2, dimx2, threes[0][x],
                                    threes[1][x], dimy1, dimx1])
                      block[dimy1][dimx1] = True
                      block[threes[0][x]][threes[1][x]] = True
                      continue
                    if(nei2 == 3):
                      extras.append([dimy3, dimx3, threes[0][x],
                                    threes[1][x], dimy1, dimx1])
                      block[dimy1][dimx1] = True
                      block[threes[0][x]][threes[1][x]] = True
                      continue
                    if(nei1 == 4 and temp_mask[dimy4][dimx4] > 0 and temp_mask[dimy5][dimx5] > 0):
                      extras.append([dimy2, dimx2, threes[0][x],
                                    threes[1][x], dimy1, dimx1])
                      block[dimy1][dimx1] = True
                      block[threes[0][x]][threes[1][x]] = True
                      continue
                    if(nei2 == 4 and temp_mask[dimy4][dimx4] > 0 and temp_mask[dimy5][dimx5] > 0):
                      extras.append([dimy3, dimx3, threes[0][x],
                                    threes[1][x], dimy1, dimx1])
                      block[dimy1][dimx1] = True
                      block[threes[0][x]][threes[1][x]] = True
                      continue

    for x in range(len(threes[0])):
        if(threes[0][x] == -1):
           continue
        if(temp_mask[threes[0][x]+1][threes[1][x]] > 0 and temp_mask[threes[0][x]][threes[1][x]+1] > 0 and neighbors8(temp_mask, threes[0][x], threes[1][x]) == 2):
            temp_mask[threes[0][x]][threes[1][x]] = 0
            extras.append([threes[0][x], threes[1][x], threes[0]
                          [x]+1, threes[1][x], threes[0][x], threes[1][x]+1])
            block[threes[0][x]+1][threes[1][x]] = True
            block[threes[0][x]][threes[1][x]+1] = True
            threes[:, x]= -1
            continue
        if(temp_mask[threes[0][x]-1][threes[1][x]] > 0 and temp_mask[threes[0][x]][threes[1][x]-1] > 0 and neighbors8(temp_mask, threes[0][x], threes[1][x]) == 2):
            temp_mask[threes[0][x]][threes[1][x]] = 0
            extras.append([threes[0][x], threes[1][x], threes[0]
                          [x]-1, threes[1][x], threes[0][x], threes[1][x]-1])
            block[threes[0][x]-1][threes[1][x]] = True
            block[threes[0][x]][threes[1][x]-1] = True
            threes[:, x]= -1
            continue
        if(temp_mask[threes[0][x]+1][threes[1][x]] > 0 and temp_mask[threes[0][x]][threes[1][x]-1] > 0 and neighbors8(temp_mask, threes[0][x], threes[1][x]) == 2):
            temp_mask[threes[0][x]][threes[1][x]] = 0
            extras.append([threes[0][x], threes[1][x], threes[0]
                          [x]+1, threes[1][x], threes[0][x], threes[1][x]-1])
            block[threes[0][x]+1][threes[1][x]] = True
            block[threes[0][x]][threes[1][x]-1] = True
            threes[:, x]= -1
            continue
        if(temp_mask[threes[0][x]-1][threes[1][x]] > 0 and temp_mask[threes[0][x]][threes[1][x]+1] > 0 and neighbors8(temp_mask, threes[0][x], threes[1][x]) == 2):
            temp_mask[threes[0][x]][threes[1][x]] = 0
            extras.append([threes[0][x], threes[1][x], threes[0]
                          [x]-1, threes[1][x], threes[0][x], threes[1][x]+1])
            block[threes[0][x]-1][threes[1][x]] = True
            block[threes[0][x]][threes[1][x]+1] = True
            threes[:, x]= -1
            continue


    for x in range(len(threes[0])):
        if(threes[0][x] == -1):
           continue
        for y in range(4):
         if y == 0:
           dimy1 = threes[0][x]
           dimx1= threes[1][x]+1  # 4neiX
           dimy2 = threes[0][x]+1
           dimx2= threes[1][x]  # 3neiX
           dimy3 = threes[0][x]-1
           dimx3= threes[1][x]  # Rem
           dimy4 = threes[0][x]
           dimx4= threes[1][x]-1  # opt
         # R
         # OX
         # X
         if y == 1:
           dimy1 = threes[0][x]+1
           dimx1= threes[1][x]  # 4neiX
           dimy2 = threes[0][x]
           dimx2= threes[1][x]-1  # 3neiX
           dimy3 = threes[0][x]
           dimx3= threes[1][x]+1  # Rem
           dimy4 = threes[0][x]-1
           dimx4= threes[1][x]  # opt
         # XOR
         # X
         if y == 2:
           dimy1 = threes[0][x]
           dimx1= threes[1][x]+1  # 4neiX
           dimy2 = threes[0][x]-1
           dimx2= threes[1][x]  # 3neiX
           dimy3 = threes[0][x]+1
           dimx3= threes[1][x]  # Rem
           dimy4 = threes[0][x]
           dimx4= threes[1][x]-1  # opt
         # X
         # OX
         # R
         if y == 3:
           dimy1 = threes[0][x]
           dimx1= threes[1][x]-1  # 4neiX
           dimy2 = threes[0][x]-1
           dimx2= threes[1][x]  # 3neiX
           dimy3 = threes[0][x]+1
           dimx3= threes[1][x]  # Rem
           dimy4 = threes[0][x]
           dimx4= threes[1][x]+1  # opt
         # X
         # XO
         # R
         if(temp_mask[dimy1][dimx1] > 0 and temp_mask[dimy2][dimx2] > 0 and ((temp_mask[dimy3][dimx3] > 0 and neighbors8(temp_mask, dimy2, dimx2) == 3 and neighbors8(temp_mask, dimy1, dimx1) == 4 and neighbors8(temp_mask, dimy3, dimx3) == 2) or (temp_mask[dimy4][dimx4] > 0 and neighbors8(temp_mask, dimy2, dimx2) == 4 and neighbors8(temp_mask, dimy1, dimx1) == 3 and neighbors8(temp_mask, dimy4, dimx4) == 2)) and neighbors8(temp_mask, threes[0][x], threes[1][x]) == 3):
            temp_mask[threes[0][x]][threes[1][x]]= 0  # up
            if temp_mask[dimy4][dimx4] > 0:
               dimy3 = dimy4
               dimx3 = dimx4
            temp_mask[dimy3][dimx3] = 0
            extras.append([threes[0][x], threes[1][x],
                          dimy1, dimx1, dimy2, dimx2])
            block[dimy2][dimx2] = True
            block[dimy1][dimx1] = True
            dimx = y_disp+dimx3
            dimy = x_disp+dimy3
            if(drawerr != 2 and mosaic_bw[dimy][dimx] == proc_imgs):
             proc_imgs2= max(mosaic_bw[dimy][dimx+1] if mosaic_bw[dimy][dimx+1] != proc_imgs else 0, mosaic_bw[dimy][dimx-1] if mosaic_bw[dimy][dimx-1] != proc_imgs else 0, mosaic_bw[dimy-1][dimx] if mosaic_bw[dimy-1][dimx] != proc_imgs else 0, mosaic_bw[dimy+1][dimx] if mosaic_bw[dimy+1][dimx] != proc_imgs else 0)
             if(proc_imgs2 != 0):
              if(fill_dict[dimy][dimx] is None):
                     fill_dict[dimy][dimx] = [None for xn in range(len(filenames))]
                     fill_dict[dimy][dimx][proc_imgs2-1] = [1]
              else:
                     if(fill_dict[dimy][dimx][proc_imgs2-1] is None):
                        fill_dict[dimy][dimx][proc_imgs2-1] = [1]
                     else:
                        if 1 not in fill_dict[dimy][dimx][proc_imgs2-1]:
                           fill_dict[dimy][dimx][proc_imgs2-1].append(1)
                        else:
                           proc_imgs2 = max(mosaic_bw[dimy][dimx+1] if mosaic_bw[dimy][dimx+1] != 0 and mosaic_bw[dimy][dimx+1] != proc_imgs2 and (fill_dict[dimy][dimx][mosaic_bw[dimy][dimx+1]-1] is None or 1 not in fill_dict[dimy][dimx][mosaic_bw[dimy][dimx+1]-1]) else 0, mosaic_bw[dimy][dimx-1] if mosaic_bw[dimy][dimx-1] != 0 and mosaic_bw[dimy][dimx-1] != proc_imgs2 and (fill_dict[dimy][dimx][mosaic_bw[dimy][dimx-1]-1] is None or 1 not in fill_dict[dimy][dimx][mosaic_bw[dimy][dimx-1]-1]) else 0, mosaic_bw[dimy-1][dimx] if mosaic_bw[dimy-1][dimx] != 0 and mosaic_bw[dimy-1][dimx] != proc_imgs2 and (fill_dict[dimy][dimx][mosaic_bw[dimy-1][dimx]-1] is None or 1 not in fill_dict[dimy][dimx][mosaic_bw[dimy-1][dimx]-1]) else 0, mosaic_bw[dimy+1][dimx] if mosaic_bw[dimy+1][dimx] != 0 and mosaic_bw[dimy+1][dimx] != proc_imgs2 and (fill_dict[dimy][dimx][mosaic_bw[dimy+1][dimx]-1] is None or 1 not in fill_dict[dimy][dimx][mosaic_bw[dimy+1][dimx]-1]) else 0)
                           if(proc_imgs2 != 0 and proc_imgs2 != proc_imgs):
                             if(fill_dict[dimy][dimx][proc_imgs2-1] is None):
                               fill_dict[dimy][dimx][proc_imgs2-1] = [1]
                             else:
                               fill_dict[dimy][dimx][proc_imgs2-1].append(1)
             mosaic_bw[dimy][dimx] = proc_imgs2
             if proc_imgs2 != 0 and filenames[proc_imgs2-1] in completed_images:
               completed_images = [ci for ci in completed_images if ci is not filenames[proc_imgs2-1]]
               repeat = 1
            threes[:, x]= -1
            continue


         if y == 0:
           dimy1 = threes[0][x]-1
           dimx1= threes[1][x]+1  # dX
           dimy2 = threes[0][x]+1
           dimx2= threes[1][x]  # sX
           dimy3 = threes[0][x]
           dimx3= threes[1][x]-1  # Rem
           dimy4 = threes[0][x]-1
           dimx4= threes[1][x]  # opt
           dimy5 = threes[0][x]+1
           dimx5= threes[1][x]+1  # opt2
         #  X
         # RO
         # X

         # X
         # RO
         #  X
         if y == 1:
           dimy1 = threes[0][x]-1
           dimx1= threes[1][x]-1  # dX
           dimy2 = threes[0][x]+1
           dimx2= threes[1][x]  # sX
           dimy3 = threes[0][x]
           dimx3= threes[1][x]+1  # Rem
           dimy4 = threes[0][x]-1
           dimx4= threes[1][x]  # opt
           dimy5 = threes[0][x]+1
           dimx5= threes[1][x]-1  # opt2
         # X
         # OR
         # X

         # X
         # OR
         # X
         if y == 2:
           dimy1 = threes[0][x]+1
           dimx1= threes[1][x]-1  # dX
           dimy2 = threes[0][x]
           dimx2= threes[1][x]+1  # sX
           dimy3 = threes[0][x]-1
           dimx3= threes[1][x]  # Rem
           dimy4 = threes[0][x]
           dimx4= threes[1][x]-1  # opt
           dimy5 = threes[0][x]+1
           dimx5= threes[1][x]+1  # opt2
         # R
         # OX
         # X

         # R
         # XO
         #  X
         if y == 3:
           dimy1 = threes[0][x]-1
           dimx1= threes[1][x]-1  # dX
           dimy2 = threes[0][x]
           dimx2= threes[1][x]+1  # sX
           dimy3 = threes[0][x]+1
           dimx3= threes[1][x]  # Rem
           dimy4 = threes[0][x]
           dimx4= threes[1][x]-1  # opt
           dimy5 = threes[0][x]-1
           dimx5= threes[1][x]+1  # opt2
         # X
         # OX
         # R

         #  X
         # XO
         # R
         if(((temp_mask[dimy1][dimx1] > 0 and temp_mask[dimy2][dimx2] > 0 and neighbors8(temp_mask, dimy2, dimx2) == 3 and neighbors8(temp_mask, dimy1, dimx1) == 2) or (temp_mask[dimy4][dimx4] > 0 and temp_mask[dimy5][dimx5] > 0 and neighbors8(temp_mask, dimy4, dimx4) == 3 and neighbors8(temp_mask, dimy5, dimx5) == 2)) and temp_mask[dimy3][dimx3] > 0 and neighbors8(temp_mask, threes[0][x], threes[1][x]) == 3 and neighbors8(temp_mask, dimy3, dimx3) == 2):
            temp_mask[dimy3][dimx3]= 0  # left
            dimx = y_disp+dimx3
            dimy = x_disp+dimy3
            if(drawerr != 2 and mosaic_bw[dimy][dimx] == proc_imgs):
             proc_imgs2= max(mosaic_bw[dimy][dimx+1] if mosaic_bw[dimy][dimx+1] != proc_imgs else 0, mosaic_bw[dimy][dimx-1] if mosaic_bw[dimy][dimx-1] != proc_imgs else 0, mosaic_bw[dimy-1][dimx] if mosaic_bw[dimy-1][dimx] != proc_imgs else 0, mosaic_bw[dimy+1][dimx] if mosaic_bw[dimy+1][dimx] != proc_imgs else 0)
             if(proc_imgs2 != 0):
              if(fill_dict[dimy][dimx] is None):
                     fill_dict[dimy][dimx] = [None for xn in range(len(filenames))]
                     fill_dict[dimy][dimx][proc_imgs2-1] = [1]
              else:
                     if(fill_dict[dimy][dimx][proc_imgs2-1] is None):
                        fill_dict[dimy][dimx][proc_imgs2-1] = [1]
                     else:
                        if 1 not in fill_dict[dimy][dimx][proc_imgs2-1]:
                           fill_dict[dimy][dimx][proc_imgs2-1].append(1)
                        else:
                           proc_imgs2 = max(mosaic_bw[dimy][dimx+1] if mosaic_bw[dimy][dimx+1] != 0 and mosaic_bw[dimy][dimx+1] != proc_imgs2 and (fill_dict[dimy][dimx][mosaic_bw[dimy][dimx+1]-1] is None or 1 not in fill_dict[dimy][dimx][mosaic_bw[dimy][dimx+1]-1]) else 0, mosaic_bw[dimy][dimx-1] if mosaic_bw[dimy][dimx-1] != 0 and mosaic_bw[dimy][dimx-1] != proc_imgs2 and (fill_dict[dimy][dimx][mosaic_bw[dimy][dimx-1]-1] is None or 1 not in fill_dict[dimy][dimx][mosaic_bw[dimy][dimx-1]-1]) else 0, mosaic_bw[dimy-1][dimx] if mosaic_bw[dimy-1][dimx] != 0 and mosaic_bw[dimy-1][dimx] != proc_imgs2 and (fill_dict[dimy][dimx][mosaic_bw[dimy-1][dimx]-1] is None or 1 not in fill_dict[dimy][dimx][mosaic_bw[dimy-1][dimx]-1]) else 0, mosaic_bw[dimy+1][dimx] if mosaic_bw[dimy+1][dimx] != 0 and mosaic_bw[dimy+1][dimx] != proc_imgs2 and (fill_dict[dimy][dimx][mosaic_bw[dimy+1][dimx]-1] is None or 1 not in fill_dict[dimy][dimx][mosaic_bw[dimy+1][dimx]-1]) else 0)
                           if(proc_imgs2 != 0 and proc_imgs2 != proc_imgs):
                             if(fill_dict[dimy][dimx][proc_imgs2-1] is None):
                               fill_dict[dimy][dimx][proc_imgs2-1] = [1]
                             else:
                               fill_dict[dimy][dimx][proc_imgs2-1].append(1)
             mosaic_bw[dimy][dimx] = proc_imgs2
             if proc_imgs2 != 0 and filenames[proc_imgs2-1] in completed_images:
               completed_images = [ci for ci in completed_images if ci is not filenames[proc_imgs2-1]]
               repeat = 1
            threes[:, x]= -1
            continue
    for x in range(len(threes[0])):
      if(threes[0][x] == -1):
        continue
      for y in range(4):
        if y == 0:
           dimy1 = threes[0][x]-1
           dimx1 = threes[1][x]-1
           dimy2 = threes[0][x]-1
           dimx2 = threes[1][x]+1
           dimy3 = threes[0][x]-1
           dimx3 = threes[1][x]
           dimy4 = threes[0][x]
           dimx4 = threes[1][x]-1
           dimy5 = threes[0][x]
           dimx5 = threes[1][x]+1
        if y == 1:
           dimy1 = threes[0][x]-1
           dimx1 = threes[1][x]-1
           dimy2 = threes[0][x]+1
           dimx2 = threes[1][x]-1
           dimy3 = threes[0][x]
           dimx3 = threes[1][x]-1
           dimy4 = threes[0][x]-1
           dimx4 = threes[1][x]
           dimy5 = threes[0][x]+1
           dimx5 = threes[1][x]
        if y == 2:
           dimy1 = threes[0][x]+1
           dimx1 = threes[1][x]+1
           dimy2 = threes[0][x]-1
           dimx2 = threes[1][x]+1
           dimy3 = threes[0][x]
           dimx3 = threes[1][x]+1
           dimy4 = threes[0][x]-1
           dimx4 = threes[1][x]
           dimy5 = threes[0][x]+1
           dimx5 = threes[1][x]
        if y == 3:
           dimy1 = threes[0][x]+1
           dimx1 = threes[1][x]+1
           dimy2 = threes[0][x]+1
           dimx2 = threes[1][x]-1
           dimy3 = threes[0][x]+1
           dimx3 = threes[1][x]
           dimy4 = threes[0][x]
           dimx4 = threes[1][x]-1
           dimy5 = threes[0][x]
           dimx5 = threes[1][x]+1
        if(block[threes[0][x]][threes[1][x]] == False and temp_mask[dimy1][dimx1] > 0 and temp_mask[dimy2][dimx2] > 0 and (neighbors8(temp_mask, threes[0][x], threes[1][x]) == 2 or (neighbors8(temp_mask, threes[0][x], threes[1][x]) == 3 and temp_mask[dimy3][dimx3] > 0))):
              if(temp_maskedges is not None and temp_maskedges[dimy4][dimx4] >= proc_imgs and temp_maskedges[dimy5][dimx5] >= proc_imgs):
                 if(temp_mask[dimy4][dimx4] == 0):
                    temp_mask[dimy4][dimx4] = 255
                    threes= np.append(threes, [[dimy4], [dimx4]], axis=1)
                 if(temp_mask[dimy5][dimx5] == 0):
                    temp_mask[dimy5][dimx5] = 255
                    threes= np.append(threes, [[dimy5], [dimx5]], axis=1)
                 continue
              if(temp_mask[dimy3][dimx3] == 0):
                 temp_mask[dimy3][dimx3] = 255
                 threes= np.append(threes, [[dimy3], [dimx3]], axis=1)
              temp_mask[threes[0][x]][threes[1][x]] = 0
              dimx = y_disp+threes[1][x]
              dimy = x_disp+threes[0][x]
              if(mosaic_bw[dimy][dimx] != proc_imgs):
                 continue
              if(drawerr != 2 and mosaic_bw[dimy][dimx] == proc_imgs):
               proc_imgs2= max(mosaic_bw[dimy][dimx+1] if mosaic_bw[dimy][dimx+1] != proc_imgs else 0, mosaic_bw[dimy][dimx-1] if mosaic_bw[dimy][dimx-1] != proc_imgs else 0, mosaic_bw[dimy-1][dimx] if mosaic_bw[dimy-1][dimx] != proc_imgs else 0, mosaic_bw[dimy+1][dimx] if mosaic_bw[dimy+1][dimx] != proc_imgs else 0)
               if(proc_imgs2 != 0):
                if(fill_dict[dimy][dimx] is None):
                     fill_dict[dimy][dimx] = [None for xn in range(len(filenames))]
                     fill_dict[dimy][dimx][proc_imgs2-1] = [1]
                else:
                     if(fill_dict[dimy][dimx][proc_imgs2-1] is None):
                        fill_dict[dimy][dimx][proc_imgs2-1] = [1]
                     else:
                        if 1 not in fill_dict[dimy][dimx][proc_imgs2-1]:
                           fill_dict[dimy][dimx][proc_imgs2-1].append(1)
                        else:
                           proc_imgs2 = max(mosaic_bw[dimy][dimx+1] if mosaic_bw[dimy][dimx+1] != 0 and mosaic_bw[dimy][dimx+1] != proc_imgs2 and (fill_dict[dimy][dimx][mosaic_bw[dimy][dimx+1]-1] is None or 1 not in fill_dict[dimy][dimx][mosaic_bw[dimy][dimx+1]-1]) else 0, mosaic_bw[dimy][dimx-1] if mosaic_bw[dimy][dimx-1] != 0 and mosaic_bw[dimy][dimx-1] != proc_imgs2 and (fill_dict[dimy][dimx][mosaic_bw[dimy][dimx-1]-1] is None or 1 not in fill_dict[dimy][dimx][mosaic_bw[dimy][dimx-1]-1]) else 0, mosaic_bw[dimy-1][dimx] if mosaic_bw[dimy-1][dimx] != 0 and mosaic_bw[dimy-1][dimx] != proc_imgs2 and (fill_dict[dimy][dimx][mosaic_bw[dimy-1][dimx]-1] is None or 1 not in fill_dict[dimy][dimx][mosaic_bw[dimy-1][dimx]-1]) else 0, mosaic_bw[dimy+1][dimx] if mosaic_bw[dimy+1][dimx] != 0 and mosaic_bw[dimy+1][dimx] != proc_imgs2 and (fill_dict[dimy][dimx][mosaic_bw[dimy+1][dimx]-1] is None or 1 not in fill_dict[dimy][dimx][mosaic_bw[dimy+1][dimx]-1]) else 0)
                           if(proc_imgs2 != 0 and proc_imgs2 != proc_imgs):
                             if(fill_dict[dimy][dimx][proc_imgs2-1] is None):
                               fill_dict[dimy][dimx][proc_imgs2-1] = [1]
                             else:
                               fill_dict[dimy][dimx][proc_imgs2-1].append(1)
               mosaic_bw[dimy][dimx]= proc_imgs2
               if proc_imgs2 != 0 and filenames[proc_imgs2-1] in completed_images:
                 completed_images = [ci for ci in completed_images if ci is not filenames[proc_imgs2-1]]
                 repeat= 1
              threes[:, x]= -1
              continue
    return temp_mask, extras, completed_images, repeat

def convert_to_8Bit(inputRaster, outputRaster, outputPixType='Byte', outputFormat='GTiff', rescale_type='rescale', percentiles=[2, 98]):
    srcRaster= gdal.Open(inputRaster)
    cmd = ['gdal_translate', '-ot', outputPixType, '-of',
           outputFormat]
    for bandId in range(srcRaster.RasterCount):
        bandId= bandId+1
        band= srcRaster.GetRasterBand(bandId)
        if rescale_type == 'rescale':
            bmin = band.GetMinimum()
            bmax= band.GetMaximum()
            if bmin is None or bmax is None:
                (bmin, bmax)= band.ComputeRasterMinMax(1)
            band_arr_tmp= band.ReadAsArray()
            bmin = np.percentile(band_arr_tmp.flatten(),
                                 percentiles[0])
            bmax= np.percentile(band_arr_tmp.flatten(),
                                percentiles[1])
        elif isinstance(rescale_type, dict):
            bmin, bmax= rescale_type[bandId]
        else:
            bmin, bmax= 0, 65535

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

def expand(mosaic_bw, current_upper_x, current_upper_y, left_upper_x, left_upper_y, real_img_shape, pixel_size):
    append_x_up = 0
    append_x_down = 0
    append_y_up = 0
    append_y_down = 0
    if (left_upper_x+(-1*real_img_shape[0]*pixel_size)) < (current_upper_x+(-1*mosaic_bw.shape[0]*pixel_size)):
      append_x_up = int(round(((current_upper_x+(-1*mosaic_bw.shape[0]*pixel_size))-(left_upper_x+(-1*real_img_shape[0]*pixel_size)))/pixel_size))
    if left_upper_x > current_upper_x:
      append_x_down = int(round((left_upper_x-current_upper_x)/pixel_size))
      current_upper_x = left_upper_x
    if (left_upper_y+(real_img_shape[1]*pixel_size)) > (current_upper_y+(mosaic_bw.shape[1]*pixel_size)):
      append_y_up = int(round(((left_upper_y+(real_img_shape[1]*pixel_size))-(current_upper_y+(mosaic_bw.shape[1]*pixel_size)))/pixel_size))
    if (left_upper_y < current_upper_y):
      append_y_down = int(round((current_upper_y-left_upper_y)/pixel_size))
      current_upper_y = left_upper_y
    mosaic_bw = np.append(mosaic_bw, np.zeros((abs(append_x_up), mosaic_bw.shape[1]), dtype=np.uint8), axis=0)
    mosaic_bw = np.append(np.zeros((abs(append_x_down), mosaic_bw.shape[1]), dtype=np.uint8), mosaic_bw, axis=0)
    mosaic_bw = np.append(mosaic_bw, np.zeros((mosaic_bw.shape[0], abs(append_y_up)), dtype=np.uint8), axis=1)
    mosaic_bw = np.append(np.zeros((mosaic_bw.shape[0], abs(append_y_down)), dtype=np.uint8), mosaic_bw, axis=1)
    return mosaic_bw, current_upper_x, current_upper_y
def draw(mosaic_bw, all_output, current_upper_x, current_upper_y, left_upper_x, left_upper_y, proc_imgs, real_img_shape, pixel_size, name):
    x_disp = round((current_upper_x-left_upper_x)/pixel_size)
    y_disp = round((left_upper_y-current_upper_y)/pixel_size)
    extra = []
    temp = np.zeros((real_img_shape[0], real_img_shape[1]), dtype=np.uint8)
    for x in range(len(all_output[::-1])):
      if(x == len(all_output[::-1])-1):
        cv2.line(temp, (all_output[x][0], all_output[x][1]),
                 (all_output[0][0], all_output[0][1]), 255, 1)
        break
      cv2.line(temp, (all_output[x][0], all_output[x][1]),
               (all_output[x+1][0], all_output[x+1][1]), 255, 1)
    temp[temp > 0]= 255
    blobcolor= temp
    h, w = blobcolor.shape
    seed = (1, 1)
    mask= np.zeros((h+2, w+2), np.uint8)
    floodflags= 4
    floodflags |= cv2.FLOODFILL_MASK_ONLY
    floodflags |= (255 << 8)
    num, im, mask, rect = cv2.floodFill(blobcolor, mask, seed, 255, 0, 0, floodflags)
    mask= mask[1:-1, 1:-1]
    mask = cv2.bitwise_not(mask)
    mask[mosaic_bw[int(x_disp):int(x_disp)+real_img_shape[0], int(y_disp):int(y_disp)+real_img_shape[1]] > 0] = 0
    ret, labels= cv2.connectedComponents(mask)
    big_lab = -1
    big_count = 0
    for lab, counts in np.array(np.unique(labels, return_counts=True)).T:
       if(lab != 0 and counts > big_count):
          big_count = counts
          big_lab = lab
    mask[labels != big_lab]= 0
    mosaic_bw[int(x_disp):int(x_disp)+real_img_shape[0], int(y_disp):int(y_disp)+real_img_shape[1]][mask > 0] = proc_imgs

def get_arguments():
    parser= argparse.ArgumentParser(description="Piecing")
    parser.add_argument("--img-path16bit", type=str, default="",
                        help="Path to the asahi image files(16 bit).")
    parser.add_argument("--img-path8bit", type=str, default="",
                        help="Path to the asahi image files(8 bit).")
    parser.add_argument("--mosaiclines-path", type=str, default="./mosaic_tmp",
                        help="Path to the files from phase 1")
    parser.add_argument("--dataset", type=str, default="",
                        help="Optional dataset prefix")
    parser.add_argument("--houses-dir", type=str, default="./houses/",
                        help="Path to house masks")
    parser.add_argument("--roads-dir", type=str, default="./roads/",
                        help="Path to road masks")
    parser.add_argument("--save-dir", type=str, default="./output",
                        help="Where to save predicted mosaic lines")
    return parser.parse_args()

def neighbors(temp_mask, curx, cury):
    return int(temp_mask[curx+1][cury] > 0)+int(temp_mask[curx-1][cury] > 0)+int(temp_mask[curx][cury+1] > 0)+int(temp_mask[curx][cury-1] > 0)


def neighbors8_ex(temp_mask, curx, cury):
    a= int(temp_mask[curx+1][cury] > 0)+int(temp_mask[curx-1][cury] > 0)+int(temp_mask[curx][cury+1] > 0)+int(temp_mask[curx][cury-1] > 0)+int(temp_mask[curx+1][cury+1] > 0)+int(temp_mask[curx-1][cury-1] > 0)+int(temp_mask[curx-1][cury+1] > 0)+int(temp_mask[curx+1][cury-1] > 0)
    return a

def neighbors8(temp_mask, curx, cury):
    if(temp_mask[curx][cury] == 0):
       return 0
    a= int(temp_mask[curx+1][cury] > 0)+int(temp_mask[curx-1][cury] > 0)+int(temp_mask[curx][cury+1] > 0)+int(temp_mask[curx][cury-1] > 0)+int(temp_mask[curx+1][cury+1] > 0)+int(temp_mask[curx-1][cury-1] > 0)+int(temp_mask[curx-1][cury+1] > 0)+int(temp_mask[curx+1][cury-1] > 0)
    return a

def neighbors8_single(temp_mask, curx, cury):
    if(temp_mask[curx+1][cury] > 0):
       return curx+1, cury
    if(temp_mask[curx-1][cury] > 0):
       return curx-1, cury
    if(temp_mask[curx][cury+1] > 0):
       return curx, cury+1
    if(temp_mask[curx][cury-1] > 0):
       return curx, cury-1
    if(temp_mask[curx+1][cury+1] > 0):
       return curx+1, cury+1
    if(temp_mask[curx-1][cury-1] > 0):
       return curx-1, cury-1
    if(temp_mask[curx-1][cury+1] > 0):
       return curx-1, cury+1
    if(temp_mask[curx+1][cury-1] > 0):
       return curx+1, cury-1

def distance(bb1, bb2):
   p1= [bb1[1], bb1[0]]
   p2= [bb2[1], bb2[0]]
   return math.sqrt(((p1[0]-p2[0])**2)+((p1[1]-p2[1])**2))

def get_voronoi(filenames):
    filenames.sort()
    voronoi_points = []
    voronoi_meta = {}
    cont = 0
    cont2 = 0
    for im_fn_orig in filenames:
        im= Image.open(im_fn_orig)
        width, height= im.size
        if os.path.isfile(im_fn_orig[:-4]+".tfw"):
          f = open(im_fn_orig[:-4]+".tfw", "r")
        else:
          f= open(im_fn_orig[:-4].replace("8bit", "")+".tfw", "r")
        contents = f.read()
        left_upper_x = float(contents.split("\n")[4])
        left_upper_y = float(contents.split("\n")[5])
        pixel_size = abs(float(contents.split("\n")[0]))
        voronoi_meta[im_fn_orig] = [(left_upper_y+(pixel_size/2)), (left_upper_x-(pixel_size/2)), (left_upper_y+(pixel_size/2))-((height*pixel_size)/2), (left_upper_x-(pixel_size/2))+((width*pixel_size)/2), height, width, pixel_size]
        voronoi_points.append([(left_upper_x-(pixel_size/2))+((width*pixel_size)/2),
                              (left_upper_y+(pixel_size/2))-((height*pixel_size)/2)])
        cont = cont+1
    vor= Voronoi(voronoi_points)

def main():
  x_disp1 = 0
  y_disp1 = 7000
  args= get_arguments()
  if not os.path.exists(args.save_dir):
        os.makedirs(args.save_dir)
  if(args.img_path16bit is not ""):
     filenames16bit= [img for img in glob.glob(args.img_path16bit+"/*.tif")]
     for im_fn_orig in filenames16bit:
       if(not os.path.isfile(args.img_path8bit+"/"+os.path.basename(im_fn_orig))):
          convert_to_8Bit(im_fn_orig, args.img_path8bit + \
                          "/"+os.path.basename(im_fn_orig))
  to_process = np.unique([int(os.path.basename(x).split("_")[0]) for x in glob.glob(args.img_path8bit+"/*.tif")]).tolist()
  for vor_ind in range(min([int(os.path.basename(x).split("_")[0]) for x in glob.glob(args.img_path8bit+"/*.tif")]), max([int(os.path.basename(x).split("_")[0]) for x in glob.glob(args.img_path8bit+"/*.tif")])):
   filenames = [img for img in glob.glob(args.img_path8bit+"/*.tif") if int(os.path.basename(img).split("_")[0]) == vor_ind]
   vor_flag = 0
   for x in filenames:
     if not os.path.isfile(args.save_dir+"/mosaic_mask_"+args.dataset+"_"+os.path.basename(x)+".txt"):
        vor_flag = 1
        break
   filenames2 = [img for img in glob.glob(args.img_path8bit+"/*.tif") if int(os.path.basename(img).split("_")[0]) == vor_ind+1]
   for x in filenames2:
     if not os.path.isfile(args.save_dir+"/mosaic_mask_"+args.dataset+"_"+os.path.basename(x)+".txt"):
        vor_flag = 1
        break
   if(vor_flag == 0):
     to_process.remove(vor_ind)
   else:
     if(vor_ind == min([int(os.path.basename(x).split("_")[0]) for x in glob.glob(args.img_path8bit+"/*.tif")])):
      for x in filenames:
       if os.path.isfile(args.save_dir+"/mosaic_mask_"+args.dataset+"_"+os.path.basename(x)+".txt"):
        os.remove(args.save_dir+"/mosaic_mask_" + \
                  args.dataset+"_"+os.path.basename(x)+".txt")
     for x in filenames2:
       if os.path.isfile(args.save_dir+"/mosaic_mask_"+args.dataset+"_"+os.path.basename(x)+".txt"):
        os.remove(args.save_dir+"/mosaic_mask_" + \
                  args.dataset+"_"+os.path.basename(x)+".txt")
     break
  min_ind = min([int(os.path.basename(x).split("_")[0]) for x in glob.glob(args.img_path8bit+"/*.tif")])
  for vor_ind in range(min([int(os.path.basename(x).split("_")[0]) for x in glob.glob(args.img_path8bit+"/*.tif") if int(os.path.basename(x).split("_")[0]) in to_process]), max([int(os.path.basename(x).split("_")[0]) for x in glob.glob(args.img_path8bit+"/*.tif") if int(os.path.basename(x).split("_")[0]) in to_process])):
   filenamesn= [img for img in glob.glob(args.img_path8bit+"/*.tif") if int(os.path.basename(img).split("_")[0]) == vor_ind+1]
   filenamesn.sort()
   filenamesp = [img for img in glob.glob(args.img_path8bit+"/*.tif") if int(os.path.basename(img).split("_")[0]) == vor_ind]
   filenamesp.sort()
   if(min_ind != vor_ind):
      filenamesp = filenamesp[::-1]
   for x in filenamesn:
       if os.path.isfile(args.save_dir+"/mosaic_mask_"+args.dataset+"_"+os.path.basename(x)+".txt"):
        os.remove(args.save_dir+"/mosaic_mask_" + \
                  args.dataset+"_"+os.path.basename(x)+".txt")
   filenames = filenamesn+filenamesp
   print("Processing "+str(len(filenames))+" images")
   completed_images = []
   mosaic_bw = None
   mosaic_bw_edges = None
   real_imgs = {}
   houseimgs_bw = {}
   tfws = {}
   current_upper_x = -1
   current_upper_y = -1
   proc_imgs = 1
   first_col = 1
   print(filenames)
   for im_fn_orig in filenames:
        real_path = im_fn_orig
        if(real_path not in real_imgs):
           im= Image.open(im_fn_orig)
           width, height= im.size
           real_imgshape= [height, width]
           real_imgs[real_path]= [height, width]
        else:
           real_imgshape = real_imgs[real_path]
        print(im_fn_orig)
        if(os.path.isfile(args.save_dir+"/mosaic_mask_"+args.dataset+"_"+os.path.basename(im_fn_orig)+".txt")):
           f = open(args.save_dir+"/mosaic_mask_"+args.dataset+"_"+os.path.basename(im_fn_orig)+".txt", "r")
           first_col = 0
        else:
           f = open(args.mosaiclines_path+"/mosaic_mask_"+args.dataset+"_"+os.path.basename(im_fn_orig)+".txt", "r")
        contents = f.read()
        mosaic_x = int(contents.split("\n")[1].split(",")[0])
        mosaic_y = int(contents.split("\n")[1].split(",")[1])
        all_output = contents.split("\n")[5].replace("[", "").replace("]", "").split(",")
        all_output_temp = []
        cont = 0
        for x in range(int(len(all_output)/2)):
           all_output_temp.append(
               [int(all_output[cont]), int(all_output[cont+1])])
           cont = cont+2
        all_output = all_output_temp
        all_output= [[int((x[0]*(float(real_imgshape[0])/float(mosaic_x)))), int((x[1]*(float(real_imgshape[1])/float(mosaic_y))))] for x in all_output]
        if os.path.isfile(im_fn_orig[:-4]+".tfw"):
          f = open(im_fn_orig[:-4]+".tfw", "r")
        else:
          f= open(im_fn_orig[:-4].replace("8bit", "")+".tfw", "r")
        contents = f.read()
        left_upper_y = float(contents.split("\n")[4])
        left_upper_x = float(contents.split("\n")[5])
        pixel_size = float(contents.split("\n")[0])
        left_upper_x = left_upper_x+(pixel_size/2)
        left_upper_y = left_upper_y-(pixel_size/2)
        if(min_ind != vor_ind and int(os.path.basename(im_fn_orig).split("_")[0]) == vor_ind):
           tfws[len(filenamesn)+((len(filenamesp)+1)-(proc_imgs-(len(filenamesn))))] = [real_imgshape[0], real_imgshape[1], left_upper_y, left_upper_x, pixel_size]
        else:
           tfws[proc_imgs] = [real_imgshape[0], real_imgshape[1], left_upper_y, left_upper_x, pixel_size]
        if(mosaic_bw is None):
           mosaic_bw= np.zeros((real_imgshape[0], real_imgshape[1]), dtype=np.uint8)
           current_upper_x = left_upper_x
           current_upper_y = left_upper_y
        else:
           mosaic_bw, current_upper_x, current_upper_y = expand(mosaic_bw, current_upper_x, current_upper_y, left_upper_x, left_upper_y, real_imgshape, pixel_size)
        if(min_ind != vor_ind and int(os.path.basename(im_fn_orig).split("_")[0]) == vor_ind):
           draw(mosaic_bw, all_output, current_upper_x, current_upper_y, left_upper_x, left_upper_y, len(
               filenamesn)+((len(filenamesp)+1)-(proc_imgs-(len(filenamesn)))), real_imgshape, pixel_size, im_fn_orig)
        else:
           draw(mosaic_bw, all_output, current_upper_x, current_upper_y, left_upper_x,
                left_upper_y, proc_imgs, real_imgshape, pixel_size, im_fn_orig)
        proc_imgs = proc_imgs+1
   if(min_ind != vor_ind):
      filenamesp = filenamesp[::-1]
   filenames = filenamesn+filenamesp
   repeat = 1
   fill_dict = [[None] * (mosaic_bw.shape[1])] * (mosaic_bw.shape[0])
   initial = 0
   fixed = []
   fixed2 = []
   fixed3 = []
   drawerr = 0
   mosaic_bw_edges= np.zeros(mosaic_bw.shape, dtype=np.uint8)
   edges_dict = {}
   while(repeat == 1):
    repeat = 0
    ret, labels = cv2.connectedComponents((mosaic_bw == 0).astype(np.uint8)*255)
    print("filling blank spaces "+str(ret))
    if(drawerr == 2 and (ret > 2 and max([xit for xit in iterelem[1:, 1] if xit < 10000000]) > 3)):
      iterelem = np.array(np.unique(labels, return_counts=True)).T
      if(max([xit for xit in iterelem[1:, 1] if xit < 10000000]) > 2):
        exit()
    if(ret > 2):
     cont_lab = 0
     for lab, counts in np.array(np.unique(labels, return_counts=True)).T:
       if(lab != 0 and counts < 10000000):
         cont_lab = cont_lab+1
         seen = []
         dims= np.where(labels == lab)
         filled = 0
         for filcont in range(4):
           if filled == 1:
              break
           if(filcont == 0):
             dimy = dims[0][np.argmin(dims[0])]
             dimx = dims[1][np.argmin(dims[0])]
           if(filcont == 1):
             dimy = dims[0][np.argmax(dims[0])]
             dimx = dims[1][np.argmax(dims[0])]
           if(filcont == 2):
             dimy = dims[0][np.argmin(dims[1])]
             dimx = dims[1][np.argmin(dims[1])]
           if(filcont == 3):
             dimy = dims[0][np.argmax(dims[1])]
             dimx = dims[1][np.argmax(dims[1])]
           for dircont in range(8):
            if(dircont == 0):
             dimy2 = dimy-1
             dimx2 = dimx-1
            if(dircont == 1):
             dimy2 = dimy
             dimx2 = dimx-1
            if(dircont == 2):
             dimy2 = dimy-1
             dimx2 = dimx
            if(dircont == 3):
             dimy2 = dimy+1
             dimx2 = dimx+1
            if(dircont == 4):
             dimy2 = dimy
             dimx2 = dimx+1
            if(dircont == 5):
             dimy2 = dimy+1
             dimx2 = dimx
            if(dircont == 6):
             dimy2 = dimy-1
             dimx2 = dimx+1
            if(dircont == 7):
             dimy2 = dimy+1
             dimx2 = dimx-1
            if(mosaic_bw[dimy2][dimx2] != 0 and (counts < 2 or (fill_dict[dimy][dimx] is None or fill_dict[dimy][dimx][mosaic_bw[dimy2][dimx2]-1] is None or counts not in fill_dict[dimy][dimx][mosaic_bw[dimy2][dimx2]-1])) and (counts < 1000 or check_bounds(mosaic_bw, dims, mosaic_bw[dimy2][dimx2], current_upper_x, current_upper_y, seen, tfws, 0, 0))):
             mosaic_bw[dims] = mosaic_bw[dimy2][dimx2]
             if filenames[mosaic_bw[dimy2][dimx2]-1] in completed_images:
                completed_images.remove(filenames[mosaic_bw[dimy2][dimx2]-1])
             if(initial == 1):
                  if(fill_dict[dimy][dimx] is None):
                     fill_dict[dimy][dimx] = [None for xn in range(len(filenames))]
                     fill_dict[dimy][dimx][mosaic_bw[dimy2][dimx2]-1] = [counts]
                  else:
                     if(fill_dict[dimy][dimx][mosaic_bw[dimy2][dimx2]-1] is None):
                        fill_dict[dimy][dimx][mosaic_bw[dimy2][dimx2]-1] = [counts]
                     else:
                        if counts not in fill_dict[dimy][dimx][mosaic_bw[dimy2][dimx2]-1]:
                          fill_dict[dimy][dimx][mosaic_bw[dimy2]
                              [dimx2]-1].append(counts)
                        else:
                          seen.append(proc_imgs)
                          continue
             filled = 1
             break
    del labels

    gc.collect()
    proc_imgs = 1
    for im_fn_orig in filenames:
        if(im_fn_orig in completed_images):
          if((len(completed_images) == len(filenames)) and drawerr == 0 and repeat == 0):
           completed_images = []
           drawerr = 1
           repeat = 1
           break
          if((len(completed_images) == len(filenames)) and drawerr == 1 and repeat == 0):
           completed_images = []
           drawerr = 2
           repeat = 1
           break
          proc_imgs = proc_imgs+1
          continue
        real_path = im_fn_orig
        real_imgshape = real_imgs[real_path]
        print(im_fn_orig)
        if(proc_imgs not in houseimgs_bw):
              house_bw= cv2.imread(args.houses_dir+"/houses_"+args.dataset+"_"+os.path.basename(im_fn_orig)[:-4]+".png", 0)
              house_bw = ndimage.morphology.binary_dilation(house_bw, structure=ndimage.generate_binary_structure(2, 2)).astype(np.uint8)*255
              houseimgs_bw[proc_imgs] = house_bw
              tfw = tfws[proc_imgs]
              pixel_size = tfw[4]
              left_upper_x = tfw[3]
              left_upper_y = tfw[2]
              x_disp = round((current_upper_x-left_upper_x)/pixel_size)
              y_disp = round((left_upper_y-current_upper_y)/pixel_size)
              tfws[proc_imgs] = [real_imgshape[0], real_imgshape[1], left_upper_y, left_upper_x, pixel_size, x_disp, y_disp]
        else:
              house_bw = houseimgs_bw[proc_imgs].copy()
              tfw = tfws[proc_imgs]
              pixel_size = tfw[4]
              left_upper_x = tfw[3]
              left_upper_y = tfw[2]
              x_disp = tfw[5]
              y_disp = tfw[6]
        temp_mask= mosaic_bw[int(x_disp):int(x_disp)+real_imgshape[0], int(y_disp):int(y_disp)+real_imgshape[1]].copy()
        temp_mask[temp_mask != proc_imgs]= 0
        temp_mask[temp_mask > 0]= 255
        if(drawerr == 2):
          temp_edge = np.zeros((real_imgshape[0], real_imgshape[1]), dtype=np.uint8)
          temp_edge[edges_dict[proc_imgs]] = 255
          edgesnew = ndimage.morphology.binary_dilation(temp_edge, structure=ndimage.generate_binary_structure(2, 2)).astype(np.uint8)*255
          edgesnew[mosaic_bw_edges[int(x_disp):int(x_disp)+real_imgshape[0], int(y_disp):int(y_disp)+real_imgshape[1]] < proc_imgs] = 0
          h, w = temp_mask.shape
          seed = (1, 1)
          mask= np.zeros((h+2, w+2), np.uint8)
          floodflags= 8
          floodflags |= cv2.FLOODFILL_MASK_ONLY
          floodflags |= (255 << 8)
          num, im, mask, rect = cv2.floodFill(edgesnew, mask, seed, 255, 0, 0, floodflags)
          mask= mask[1:-1, 1:-1]
          mask = cv2.bitwise_not(mask)
          del temp_mask
          del edgesnew
          temp_mask = mask
        else:
          if(drawerr == 1):
            h, w = temp_mask.shape
            seed = (1, 1)
            mask= np.zeros((h+2, w+2), np.uint8)
            floodflags= 4
            floodflags |= cv2.FLOODFILL_MASK_ONLY
            floodflags |= (255 << 8)
            num, im, mask, rect = cv2.floodFill(temp_mask, mask, seed, 255, 0, 0, floodflags)
            mask= mask[1:-1, 1:-1]
            mask = cv2.bitwise_not(mask)
            temp_mask = mask
            mosaic_bw[int(x_disp):int(x_disp)+real_imgshape[0], int(y_disp):int(y_disp)+real_imgshape[1]][temp_mask > 0] = proc_imgs
        edgesnew= ndimage.morphology.binary_erosion(temp_mask)
        if(drawerr >= 1):
           temp_mask3 = temp_mask.copy()
        else:
           temp_mask3 = temp_mask
        temp_mask3[edgesnew > 0]= 0
        del edgesnew
        curlab = 0
        max_count = 0
        ret3, labels= cv2.connectedComponents(temp_mask3)
        for lab, counts in np.array(np.unique(labels, return_counts=True)).T:
           if(lab != 0 and max_count < counts):
              max_count = counts
              curlab = lab
        temp_mask3[labels != curlab] = 0
        del labels
        if(proc_imgs > 1 and int(os.path.basename(filenames[proc_imgs-2]).split("_")[0]) == vor_ind):
           prev_im = proc_imgs-1
        else:
           prev_im = 200
        if(proc_imgs+1 <= len(filenames) and int(os.path.basename(filenames[proc_imgs]).split("_")[0]) == vor_ind):
           nex_im = proc_imgs+1
        else:
           nex_im = 200
        if(im_fn_orig not in fixed3 and drawerr == 1):
          house_bwerr = house_bw.copy()
          house_bwerr[temp_mask3 == 0]= 0
          house_bwerr[house_bwerr > 0]= 255
          ret, labels= cv2.connectedComponents(house_bwerr)
          hflag = 0
          for lab in np.array(np.unique(labels, return_counts=False)).T:
           if(lab != 0):
             dims= np.where(labels == lab)
             dimy = dims[0][np.argmin(dims[0])]
             dimx = dims[1][np.argmin(dims[0])]
             reach = 1000
             house_bwcrop= house_bw[max(dimy-reach, 0):min(dimy+reach, real_imgshape[0]), max(dimx-reach, 0):min(dimx+reach, real_imgshape[1])].copy()
             house_bwcrop[temp_mask[max(dimy-reach, 0):min(dimy+reach, real_imgshape[0]), max(dimx-reach, 0):min(dimx+reach, real_imgshape[1])] == 0] = 0
             house_bw[max(dimy-reach, 0):min(dimy+reach, real_imgshape[0]), max(dimx-reach, 0):min(dimx+reach, real_imgshape[1])][mosaic_bw[int(x_disp):int(x_disp)+real_imgshape[0], int(y_disp):int(y_disp)+real_imgshape[1]][max(dimy-reach, 0):min(dimy+reach, real_imgshape[0]), max(dimx-reach, 0):min(dimx+reach, real_imgshape[1])] == 0] = 0
             ret, labelsfull = cv2.connectedComponents(house_bw[max(dimy-reach, 0):min(dimy+reach, real_imgshape[0]), max(dimx-reach, 0):min(dimx+reach, real_imgshape[1])])
             ret, labelscrop= cv2.connectedComponents(house_bwcrop)

             dimy2= min(dimy, reach)
             dimx2= min(dimx, reach)
             curlab = labelsfull[dimy2][dimx2]
             croplab = labelscrop[dimy2][dimx2]
             countall= np.count_nonzero(labelsfull == curlab)
             countcrop= np.count_nonzero(labelscrop == croplab)
             if(im_fn_orig not in fixed3 and countall-countcrop < 1500 and countall-countcrop > 0):
                dims2= np.where(labelsfull == curlab)
                dimxmin = min(dims2[0])
                dimxmax = max(dims2[0])
                dimymin = min(dims2[1])
                dimymax = max(dims2[1])
                temp_wind= mosaic_bw[int(x_disp):int(x_disp)+real_imgshape[0], int(y_disp):int(y_disp)+real_imgshape[1]][max(dimy-reach, 0):min(dimy+reach, real_imgshape[0]), max(dimx-reach, 0):min(dimx+reach, real_imgshape[1])][max(dimxmin-5, 0):min(dimxmax+5, real_imgshape[0]), max(dimymin-5, 0):min(dimymax+5, real_imgshape[1])].copy()
                temp_wind_eyes= mosaic_bw[int(x_disp):int(x_disp)+real_imgshape[0], int(y_disp):int(y_disp)+real_imgshape[1]][max(dimy-reach, 0):min(dimy+reach, real_imgshape[0]), max(dimx-reach, 0):min(dimx+reach, real_imgshape[1])][max(dimxmin-5, 0):min(dimxmax+5, real_imgshape[0]), max(dimymin-5, 0):min(dimymax+5, real_imgshape[1])].copy()
                temp_wind_eyes[labelsfull[max(dimxmin-5, 0):min(dimxmax+5, real_imgshape[0]), max(dimymin-5, 0):min(dimymax+5, real_imgshape[1])] != curlab] = 0
                temp_wind_eyes[temp_wind_eyes > 0]= 255
                h, w = temp_wind_eyes.shape
                seed = (1, 1)
                mask= np.zeros((h+2, w+2), np.uint8)
                floodflags= 4
                floodflags |= cv2.FLOODFILL_MASK_ONLY
                floodflags |= (255 << 8)
                num, im, mask, rect= cv2.floodFill(temp_wind_eyes, mask, seed, 255, 0, 0, floodflags)
                mask= mask[1:-1, 1:-1]
                mask = cv2.bitwise_not(mask)
                if(int(os.path.basename(im_fn_orig).split("_")[0]) == vor_ind and first_col == 0):
                  mask_vor = mask.copy()
                  mask_vor[mosaic_bw[int(x_disp):int(x_disp)+real_imgshape[0], int(y_disp):int(y_disp)+real_imgshape[1]][max(dimy-reach, 0):min(dimy+reach, real_imgshape[0]), max(dimx-reach, 0):min(dimx+reach, real_imgshape[1])][max(dimxmin-5, 0):min(dimxmax+5, real_imgshape[0]), max(dimymin-5, 0):min(dimymax+5, real_imgshape[1])] == proc_imgs] = 0
                  if(len(np.where((mask_vor > 0) & ((mosaic_bw[int(x_disp):int(x_disp)+real_imgshape[0], int(y_disp):int(y_disp)+real_imgshape[1]][max(dimy-reach, 0):min(dimy+reach, real_imgshape[0]), max(dimx-reach, 0):min(dimx+reach, real_imgshape[1])][max(dimxmin-5, 0):min(dimxmax+5, real_imgshape[0]), max(dimymin-5, 0):min(dimymax+5, real_imgshape[1])] == 0) | (mosaic_bw[int(x_disp):int(x_disp)+real_imgshape[0], int(y_disp):int(y_disp)+real_imgshape[1]][max(dimy-reach, 0):min(dimy+reach, real_imgshape[0]), max(dimx-reach, 0):min(dimx+reach, real_imgshape[1])][max(dimxmin-5, 0):min(dimxmax+5, real_imgshape[0]), max(dimymin-5, 0):min(dimymax+5, real_imgshape[1])] == nex_im) | (mosaic_bw[int(x_disp):int(x_disp)+real_imgshape[0], int(y_disp):int(y_disp)+real_imgshape[1]][max(dimy-reach, 0):min(dimy+reach, real_imgshape[0]), max(dimx-reach, 0):min(dimx+reach, real_imgshape[1])][max(dimxmin-5, 0):min(dimxmax+5, real_imgshape[0]), max(dimymin-5, 0):min(dimymax+5, real_imgshape[1])] == prev_im)))[0]) > 0):
                     continue
                mosaic_bw[int(x_disp):int(x_disp)+real_imgshape[0], int(y_disp):int(y_disp)+real_imgshape[1]][max(dimy-reach, 0):min(dimy+reach, real_imgshape[0]), max(dimx-reach, 0):min(dimx+reach, real_imgshape[1])][max(dimxmin-5, 0):min(dimxmax+5, real_imgshape[0]), max(dimymin-5, 0):min(dimymax+5, real_imgshape[1])][mask > 0] = proc_imgs
                hflag = 1
                temp_wind2= mosaic_bw[int(x_disp):int(x_disp)+real_imgshape[0], int(y_disp):int(y_disp)+real_imgshape[1]][max(dimy-reach, 0):min(dimy+reach, real_imgshape[0]), max(dimx-reach, 0):min(dimx+reach, real_imgshape[1])][max(dimxmin-5, 0):min(dimxmax+5, real_imgshape[0]), max(dimymin-5, 0):min(dimymax+5, real_imgshape[1])]
                for lab2, counts2 in np.array(np.unique(temp_wind, return_counts=True)).T:
                  if(lab2 != 0):
                     dims3= np.where(temp_wind == lab2)
                     dimy2 = dims3[0][np.argmin(dims3[0])]
                     dimx2 = dims3[1][np.argmin(dims3[0])]
                     if(counts2 != len(np.where(temp_wind2 == temp_wind[dimy2][dimx2])[0])):
                       if filenames[temp_wind[dimy2][dimx2]-1] != im_fn_orig and filenames[temp_wind[dimy2][dimx2]-1] in completed_images:
                          completed_images.remove(
                              filenames[temp_wind[dimy2][dimx2]-1])
                del temp_wind
                del temp_wind_eyes
             else:
              if(im_fn_orig not in fixed3 and countall-countcrop > 0 and countcrop > 0 and countcrop < 1500):
                dims= np.where(labelscrop == croplab)
                if(int(os.path.basename(im_fn_orig).split("_")[0]) == vor_ind and first_col == 0):
                  if(len(np.where(mosaic_bw[int(x_disp):int(x_disp)+real_imgshape[0], int(y_disp):int(y_disp)+real_imgshape[1]][max(dimy-reach, 0):min(dimy+reach, real_imgshape[0]), max(dimx-reach, 0):min(dimx+reach, real_imgshape[1])][dims] == 0)[0]) > 0):
                    continue
                mosaic_bw[int(x_disp):int(x_disp)+real_imgshape[0], int(y_disp):int(y_disp)+real_imgshape[1]][max(dimy-reach, 0):min(dimy+reach, real_imgshape[0]), max(dimx-reach, 0):min(dimx+reach, real_imgshape[1])][dims] = 0
                hflag = 1
                if im_fn_orig in completed_images:
                   completed_images.remove(im_fn_orig)
             del house_bwcrop
             del labelsfull
             del labelscrop
          del house_bw
          del house_bwerr
          del labels
          gc.collect()
          if(hflag == 1):
            if(im_fn_orig in fixed2):
               fixed3.append(im_fn_orig)
               repeat = 1
            if(im_fn_orig in fixed):
               fixed2.append(im_fn_orig)
               repeat = 1
            if(im_fn_orig not in fixed):
               fixed.append(im_fn_orig)
               repeat = 1
            proc_imgs = proc_imgs+1
            continue
        if(drawerr == 2):
          kernel= np.ones((3, 3), np.uint8)
          house_bw= cv2.erode(house_bw, kernel, iterations = 1)
          house_new = house_bw.copy()
          houseerr = house_bw.copy()
          house_bw[temp_mask == 0]= 0
          house_bw[house_bw > 0]= 255
          houseerr[temp_mask3 == 0]= 0
          house_new[mosaic_bw[int(x_disp):int(x_disp)+real_imgshape[0], int(y_disp):int(y_disp)+real_imgshape[1]] == 0] = 0
          ret, labelscrop= cv2.connectedComponents(house_bw)
          ret, labels= cv2.connectedComponents(houseerr)
          ret, labelsall= cv2.connectedComponents(house_new)
          for lab in np.array(np.unique(labels, return_counts=False)).T:
           if(lab != 0):
             dims= np.where(labels == lab)
             dimy = dims[0][np.argmin(dims[0])]
             dimx = dims[1][np.argmin(dims[0])]
             curlab = labelsall[dimy][dimx]
             croplab = labelscrop[dimy][dimx]
             if(np.count_nonzero(labelsall == curlab) != np.count_nonzero(labelscrop == croplab)):
                cv2.circle(mosaic_real, (int(y_disp)+dimx,
                           int(x_disp)+dimy), 100, (0, 0, 255), 10)
          del house_bw
          del house_new
          del houseerr
          del labels
          del labelsall
          del labelscrop
          gc.collect()
        temp_maskall = temp_mask
        temp_mask = temp_mask3

        temp_mask, removed_flag, temp_mask_ret, repeat_f= remove_singlets(temp_mask, mosaic_bw, filenames, current_upper_x, current_upper_y, tfws, fill_dict, x_disp, y_disp, completed_images, drawerr)
        if(repeat_f == 1):
           repeat = 1
        if(drawerr == 2 and removed_flag == True):
           exit()
        if(removed_flag == True):
           mosaic_bw[mosaic_bw == proc_imgs]= 0
           mosaic_bw[int(x_disp):int(x_disp)+real_imgshape[0], int(y_disp):int(y_disp)+real_imgshape[1]][temp_mask_ret > 0] = proc_imgs
           repeat = 1
           initial = 1
           proc_imgs = proc_imgs+1
           del temp_mask
           del temp_maskall
           del temp_mask3
           del temp_mask_ret
           gc.collect()
           continue
        if((len(completed_images)+1 == len(filenames)) and drawerr == 0 and repeat == 0):
          completed_images = []
          drawerr = 1
          repeat = 1
          del temp_mask
          del temp_maskall
          del temp_mask3
          del temp_mask_ret
          gc.collect()
          break
        if((len(completed_images)+1 == len(filenames)) and drawerr == 1 and repeat == 0):
          completed_images = []
          mosaic_bw_edges[mosaic_bw_edges == proc_imgs]= 0
          temp_mask, extras, completed_images, _= remove_threes(temp_mask, temp_maskall, None, proc_imgs, mosaic_bw, filenames, completed_images, x_disp, y_disp, drawerr, fill_dict, (int(os.path.basename(im_fn_orig).split("_")[0]) == vor_ind) & (int(os.path.basename(im_fn_orig).split("_")[0]) != min_ind), prev_im, nex_im, x_disp1, y_disp1)  # remove_threes(temp_mask,temp_maskall,None,proc_imgs)
          temp_mask[temp_mask > 0]= 255
          for x in extras:
            temp_mask[x[0]][x[1]] = 255
          edges_dict[proc_imgs]= np.where(temp_mask > 0)
          temp_mask[mosaic_bw_edges[int(x_disp):int(x_disp)+real_imgshape[0], int(y_disp):int(y_disp)+real_imgshape[1]] > proc_imgs] = 0
          mosaic_bw_edges[int(x_disp):int(x_disp)+real_imgshape[0], int(y_disp):int(y_disp)+real_imgshape[1]][temp_mask > 0] = proc_imgs
          drawerr = 2
          repeat = 1
          del temp_mask
          del temp_maskall
          del temp_mask3
          del temp_mask_ret
          gc.collect()
          break
        if(drawerr == 2):
           temp_mask, extras, completed_images, repeat_f = remove_threes(temp_mask, temp_maskall, mosaic_bw_edges[int(x_disp):int(x_disp)+real_imgshape[0], int(y_disp):int(y_disp)+real_imgshape[1]], proc_imgs, mosaic_bw, filenames, completed_images, x_disp, y_disp, drawerr, fill_dict, (int(os.path.basename(im_fn_orig).split("_")[0]) == vor_ind) & (int(os.path.basename(im_fn_orig).split("_")[0]) != min_ind), prev_im, nex_im, x_disp1, y_disp1)  # remove_threes(temp_mask,temp_maskall,mosaic_bw_edges[int(x_disp):int(x_disp)+real_imgshape[0],int(y_disp):int(y_disp)+real_imgshape[1]],proc_imgs)
        else:
           temp_mask, extras, completed_images, repeat_f= remove_threes(temp_mask, temp_maskall, None, proc_imgs, mosaic_bw, filenames, completed_images, x_disp, y_disp, drawerr, fill_dict, (int(os.path.basename(im_fn_orig).split("_")[0]) == vor_ind) & (int(os.path.basename(im_fn_orig).split("_")[0]) != min_ind), prev_im, nex_im, x_disp1, y_disp1)  # remove_threes(temp_mask,temp_maskall,None,proc_imgs)
        if(repeat_f == 1):
           repeat = 1
        all_output_save = np.zeros((0, 3), dtype=np.int16)
        curx = -1
        cury = -1
        ones= np.where(temp_mask > 0)
        index = np.argmin(ones[0])
        startx = ones[0][index]
        starty = ones[1][index]
        curx = startx
        cury = starty
        seen = {}
        last_index = 0
        all_output_save = np.append(all_output_save, [[cury, curx, 0]], axis=0)
        seen[str([curx, cury])] = 1
        now= datetime.datetime.now()
        elapsed = 0
        cont_exit = 0
        while((not ([starty, startx] == [cury, curx] and len(all_output_save) > 20000)) and elapsed < 60):
             cont_exit = cont_exit+1
             if(cont_exit % 1000 == 0):
                elapsed= (datetime.datetime.now() - now).total_seconds()
             if(last_index < 1 and temp_mask[curx+1][cury] > 0 and (str([curx+1, cury]) not in seen or seen[str([curx+1, cury])] == -1 or ([starty, startx] == [cury, curx+1] and len(all_output_save) > 20000))):
                curx = curx+1
                cury = cury
                seen[str([curx, cury])] = len(all_output_save)
                all_output_save = np.append(all_output_save, [[cury, curx, 1]], axis=0)
                last_index = 0
                continue
             if(last_index < 2 and temp_mask[curx][cury+1] > 0 and (str([curx, cury+1]) not in seen or seen[str([curx, cury+1])] == -1 or ([starty, startx] == [cury+1, curx] and len(all_output_save) > 20000))):
                curx = curx
                cury = cury+1
                seen[str([curx, cury])] = len(all_output_save)
                last_index = 0
                all_output_save = np.append(all_output_save, [[cury, curx, 2]], axis=0)
                continue
             if(last_index < 3 and temp_mask[curx-1][cury] > 0 and (str([curx-1, cury]) not in seen or seen[str([curx-1, cury])] == -1 or ([starty, startx] == [cury, curx-1] and len(all_output_save) > 20000))):
                curx = curx-1
                cury = cury
                seen[str([curx, cury])] = len(all_output_save)
                last_index = 0
                all_output_save = np.append(all_output_save, [[cury, curx, 3]], axis=0)
                continue
             if(last_index < 4 and temp_mask[curx][cury-1] > 0 and (str([curx, cury-1]) not in seen or seen[str([curx, cury-1])] == -1 or ([starty, startx] == [cury-1, curx] and len(all_output_save) > 20000))):
                curx = curx
                cury = cury-1
                seen[str([curx, cury])] = len(all_output_save)
                last_index = 0
                all_output_save = np.append(all_output_save, [[cury, curx, 4]], axis=0)
                continue
             if(last_index < 5 and temp_mask[curx+1][cury+1] > 0 and (str([curx+1, cury+1]) not in seen or seen[str([curx+1, cury+1])] == -1 or ([starty, startx] == [cury+1, curx+1] and len(all_output_save) > 20000))):
                curx = curx+1
                cury = cury+1
                seen[str([curx, cury])] = len(all_output_save)
                last_index = 0
                all_output_save = np.append(all_output_save, [[cury, curx, 5]], axis=0)
                continue
             if(last_index < 6 and temp_mask[curx-1][cury-1] > 0 and (str([curx-1, cury-1]) not in seen or seen[str([curx-1, cury-1])] == -1 or ([starty, startx] == [cury-1, curx-1] and len(all_output_save) > 20000))):
                curx = curx-1
                cury = cury-1
                seen[str([curx, cury])] = len(all_output_save)
                last_index = 0
                all_output_save = np.append(all_output_save, [[cury, curx, 6]], axis=0)
                continue
             if(last_index < 7 and temp_mask[curx+1][cury-1] > 0 and (str([curx+1, cury-1]) not in seen or seen[str([curx+1, cury-1])] == -1 or ([starty, startx] == [cury-1, curx+1] and len(all_output_save) > 20000))):
                curx = curx+1
                cury = cury-1
                seen[str([curx, cury])] = len(all_output_save)
                last_index = 0
                all_output_save = np.append(all_output_save, [[cury, curx, 7]], axis=0)
                continue
             if(last_index < 8 and temp_mask[curx-1][cury+1] > 0 and (str([curx-1, cury+1]) not in seen or seen[str([curx-1, cury+1])] == -1 or ([starty, startx] == [cury+1, curx-1] and len(all_output_save) > 20000))):
                curx = curx-1
                cury = cury+1
                seen[str([curx, cury])] = len(all_output_save)
                last_index = 0
                all_output_save = np.append(all_output_save, [[cury, curx, 8]], axis=0)
                continue
             if(seen[str([curx, cury])] > 0 and seen[str([curx, cury])] < 10 and len(all_output_save) > len(ones[0])-10):
                break
             temp, all_output_save= all_output_save[-1], all_output_save[:-1]
             seen[str([curx, cury])]= -1
             if(len(all_output_save > 0)):
                 curx = all_output_save[-1][1]
                 cury = all_output_save[-1][0]
                 last_index = temp[2]
             else:
                 curx = startx
                 cury = starty
                 last_index = 0
        if(elapsed >= 60):
           print("error path")
           exit()
        del seen
        temp_mask[temp_mask > 0]= 255
        all_output_save = all_output_save[:, [0, 1]]
        if(True):
         for x in extras:
           pos= np.where(np.all(all_output_save == [x[3], x[2]], axis=1))[0]
           if(x[4] is None):
               pos2 = -1
           else:
               pos2= np.where(np.all(all_output_save == [x[5], x[4]], axis=1))[0]
               if((pos[0] == 0 and pos2[0] == len(all_output_save)-1) or (pos2[0] == 0 and pos[0] == len(all_output_save)-1)):
                  pos = 0
                  pos2 = 0
               else:
                 if(abs(pos[0]-pos2[0]) < 2):
                     pos = pos[0]
                     pos2 = pos2[0]
                 else:
                     index_flag = 0
                     for index1 in pos:
                       for index2 in pos2:
                           if(abs(index1-index2) < 2):
                              pos = index1
                              pos2 = index2
                              index_flag = 1
                              break
                       if(index_flag == 1):
                          break

           all_output_save= np.insert(all_output_save, max(pos, pos2), [x[1], x[0]], axis=0)
           temp_mask[x[0]][x[1]] = 255
        totnodes = len(all_output_save)
        all_output_save_real= [[[(left_upper_y+(all_output_save[x][0]*pixel_size)), (left_upper_x-(all_output_save[x][1]*pixel_size))], [(left_upper_y+(all_output_save[(x+1) % totnodes][0]*pixel_size)), (left_upper_x-(all_output_save[(x+1) % totnodes][1]*pixel_size))]] for x in range(len(all_output_save))]
        if(repeat == 0 and drawerr == 2):
          f= open(args.save_dir+"/mosaic_mask_tmp_"+args.dataset+"_"+os.path.basename(im_fn_orig)+".txt", "w+")
          f.write("Size:\r\n")
          f.write(str(real_imgshape[0])+","+str(real_imgshape[1]))
          f.write("\r\n")
          f.write("Number of nodes:\r\n")
          f.write(str(len(all_output_save)))
          f.write("\r\n")
          f.write("Nodes:\r\n")
          f.write(str(all_output_save.tolist()))
          f.write("\r\n")
          f.close()
          w= shapefile.Writer(args.save_dir+os.path.basename(im_fn_orig), shapeType=3)
          w.field('DeletionFlag', 'C', 1, 0)
          w.field('NUM_POINTS', 'N', 10, 0)
          w.line(all_output_save_real)
          w.record(len(all_output_save_real))
          w.close()
          if(im_fn_orig == filenames[-1]):
             for x in filenames:
               if os.path.isfile(args.save_dir+"/mosaic_mask_tmp_"+args.dataset+"_"+os.path.basename(x)+".txt"):
                 os.rename(args.save_dir+"/mosaic_mask_tmp_"+args.dataset+"_"+os.path.basename(x) + \
                           ".txt", args.save_dir+"/mosaic_mask_"+args.dataset+"_"+os.path.basename(x)+".txt")
        if(drawerr == 1):
          mosaic_bw_edges[mosaic_bw_edges == proc_imgs]= 0
          edges_dict[proc_imgs]= np.where(temp_mask > 0)
          temp_mask[mosaic_bw_edges[int(x_disp):int(x_disp)+real_imgshape[0], int(y_disp):int(y_disp)+real_imgshape[1]] > proc_imgs] = 0
          mosaic_bw_edges[int(x_disp):int(x_disp)+real_imgshape[0], int(y_disp):int(y_disp)+real_imgshape[1]][temp_mask > 0]= proc_imgs
        del all_output_save
        del all_output_save_real
        del temp_mask
        del temp_mask3
        del temp_maskall
        gc.collect()
        if(im_fn_orig not in completed_images):
          completed_images.append(im_fn_orig)
        if((len(completed_images) == len(filenames)) and drawerr == 0):
          completed_images = []
          drawerr = 1
          repeat = 1
          break
        if((len(completed_images) == len(filenames)) and drawerr == 1):
          completed_images = []
          drawerr = 2
          repeat = 1
          break
        if((len(completed_images) != len(filenames)) and im_fn_orig == filenames[-1]):
          proc_imgs = proc_imgs+1
          repeat = 1
          continue
        proc_imgs = proc_imgs+1
   houseimgs_bw.clear()
   del houseimgs_bw


if __name__ == '__main__':
    main()
