#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Created on Sun Aug  8 19:13:44 2021

@author: aze_ace
"""

#########################################################
#This code is adapted from nii2png
#https://github.com/alexlaurence/NIfTI-Image-Converter
#########################################################


import os
import shutil

import nibabel as nib
import numpy as np
import pylab


def nii2png(input_path, output_path, idx_list, slice_ax='z'):
    """
    Converts 3D volumes to png images

    Parameters
    ----------
    input_path : str
        Input directory
    output_path : str
        Output directory
    idx_list : list of random indices
        list of random indices  to select volumes
    slice_ax : string
        axis to slide a volume
     slice 2D images and masks of nii volume
    -------

    """

    source_ct = list(sorted(os.listdir(os.path.join(input_path, 'CT'))))
    source_gt = list(sorted(os.listdir(os.path.join(input_path, 'GT'))))
    ct_dir = os.path.join(input_path, 'CT')
    gt_dir = os.path.join(input_path, 'GT')

    dst_ct = os.path.join(output_path, 'CT')
    dst_gt = os.path.join(output_path, 'GT')

    i = 0  # counter

    for idx in idx_list:

        ct = source_ct[idx]
        gt = source_gt[idx]
        ct_fname = os.path.join(ct_dir, ct)
        gt_fname = os.path.join(gt_dir, gt)

        img_array = nib.load(ct_fname).get_fdata()
        mask_array = nib.load(gt_fname).get_fdata()

        ct.replace('nii.gz', '')
        gt.replace('nii.gz', '')

        if slice_ax == 'z' and mask_array.shape[2] == img_array.shape[2]:

            total_slices = img_array.shape[2]

        elif slice_ax == 'y' and mask_array.shape[1] == img_array.shape[1]:

            total_slices = img_array.shape[1]

        elif slice_ax == 'x' and mask_array.shape[0] == img_array.shape[0]:

            total_slices = img_array.shape[0]

        else:
            break

        # iterate through slices
        for sl in range(0, total_slices):
            # alternate slices
            if (i % 1) == 0:
                # rotate or no rotate

                if slice_ax == 'z':
                    ct_data = img_array[:, :, sl]
                    gt_data = mask_array[:, :, sl]
                    if 'coronacases' in ct:
                        ct_data = np.rot90(ct_data, k=1)
                        gt_data = np.rot90(gt_data, k=1)

                    elif 'volume' in ct:
                        ct_data = np.rot90(ct_data[0:500, 70:500], k=-1)
                        gt_data = np.rot90(gt_data[0:500, 70:500], k=-1)

                    elif 'radiopaedia' in ct:
                        ct_data = np.rot90(ct_data, k=-1)
                        gt_data = np.rot90(gt_data, k=-1)

                elif slice_ax == 'x':
                    ct_data = img_array[sl, :, :]
                    gt_data = mask_array[sl, :, :]

                elif slice_ax == 'y':
                    ct_data = img_array[:, sl, :]
                    gt_data = mask_array[:, sl, :]

                else:
                    print('Axis not available')
                    break

                # selection of image slices based on the mask 

                print('Saving image...')

                img_name = 'ct_' + str(i) + slice_ax + '{:0>3}'.format(str(sl + 1)) + \
                           '.png'
                mask_name = 'gt_' + str(i) + slice_ax + "{:0>3}".format(str(sl + 1)) + \
                            '.png'

                if len(np.unique(gt_data)) != 1:  # mask with covid
                    # imageio.imwrite(image_name, data) 
                    pylab.imsave(img_name, ct_data, cmap='gray')
                    pylab.imsave(mask_name, gt_data, cmap='gray')

                    print('Saved...')

                    # move images to folder
                    print('Moving image...')
                    src_img = img_name
                    src_mask = mask_name
                    shutil.move(src_img, dst_ct)
                    shutil.move(src_mask, dst_gt)
                    i += 1
                    print('Moved...')
                    print('Finished converting images')
                else:
                    print('No mask')

            print('Num of img and mask slices in axes {},:{}'.format("z", i))

    



