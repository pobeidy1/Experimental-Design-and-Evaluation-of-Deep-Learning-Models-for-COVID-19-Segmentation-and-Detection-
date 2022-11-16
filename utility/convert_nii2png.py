#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
@author: aze_ace
"""

import os
import shutil

import nibabel as nib
import numpy as np
import cv2
import pylab


def nii2png(input_path, output_path, slice_ax='z', remove=False):
    """
    :param input_path: str
        Input directory to the nii volumes
    :param output_path: str
        Output directory to place the sliced volumes
    :param slice_ax: string
        axis to slide a volume
    :param remove:  boolean to remove the input directory
    :return: slice 2D images and masks of nii volume
    """
    assert slice_ax in ['x', 'y', 'z'], ' Only axes x, y and z available'

    size = (512, 512)

    source_ct = sorted(os.listdir(os.path.join(input_path, 'CT')))
    source_gt = sorted(os.listdir(os.path.join(input_path, 'GT')))
    ct_dir = os.path.join(input_path, 'CT')
    gt_dir = os.path.join(input_path, 'GT')

    dst_ct = os.path.join(output_path, 'CT')
    dst_gt = os.path.join(output_path, 'GT')

    assert len(source_ct) == len(source_gt), 'Unequal number of Images and Masks'

    i = 0  # counter

    no_mask = 0

    for idx in range(len(source_ct)):

        ct = source_ct[idx]
        gt = source_gt[idx]
        ct_fname = os.path.join(ct_dir, ct)
        gt_fname = os.path.join(gt_dir, gt)

        img_array = nib.load(ct_fname).get_fdata()
        mask_array = nib.load(gt_fname).get_fdata()

        print(mask_array.shape)

        ct.replace('nii.gz', '')
        gt.replace('nii.gz', '')

        if slice_ax == 'z' and mask_array.shape[2] == img_array.shape[2]:
            start = 0
            total = img_array.shape[2]

        elif slice_ax == 'y' and mask_array.shape[1] == img_array.shape[1]:
            start = 246
            total = img_array.shape[1] // 2 + 5

        elif slice_ax == 'x' and mask_array.shape[0] == img_array.shape[0]:
            start = 246
            total = img_array.shape[0] // 2 + 5

        else:
            continue

        # iterate through slices
        for sl in range(start, total):

            if slice_ax == 'z':
                ct_data = img_array[:, :, sl]
                gt_data = mask_array[:, :, sl]

            elif slice_ax == 'x':
                ct_data = img_array[sl, :, :]
                gt_data = mask_array[sl, :, :]

            else:
                ct_data = img_array[:, sl, :]
                gt_data = mask_array[:, sl, :]

                # selection of image slices based on the mask

            if ct_data.shape != size:
                ct_data = cv2.resize(ct_data, size, interpolation=cv2.INTER_NEAREST)
                gt_data = cv2.resize(gt_data, size, interpolation=cv2.INTER_NEAREST)

            if len(np.unique(gt_data)) > 1:  # mask with covid

                i += 1
                img_name = f'ct_gz{i}{slice_ax}' + '{:0>3}'.format(str(sl + 1)) + '.png'
                mask_name = f'gt_gz{i}{slice_ax}' + "{:0>3}".format(str(sl + 1)) + '.png'

                pylab.imsave(img_name, ct_data, cmap='gray')
                pylab.imsave(mask_name, gt_data, cmap='gray')

                # move images to folder
                src_img = img_name
                src_mask = mask_name
                shutil.move(src_img, dst_ct)
                shutil.move(src_mask, dst_gt)

            else:
                no_mask += 1

    if remove:
        shutil.rmtree(ct_dir)
        shutil.rmtree(gt_dir)
    print(f'Imanges without mask {no_mask}')

    print(f'Num of img and mask slices in axis {slice_ax}: {i}')