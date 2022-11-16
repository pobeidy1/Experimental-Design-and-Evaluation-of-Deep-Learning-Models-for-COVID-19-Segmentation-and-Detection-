import os
import numpy as np
import torch

from PIL import Image
from skimage.measure import label, regionprops


np.random.seed(1234)
torch.manual_seed(1234)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


class CovidMerged(torch.utils.data.Dataset):
    """
    A method that takes images and segmentation masks (gt) as input, threshold masks
    for the instance segmentation and extracts the bounding boxes from the masks coordinates.
    """

    def __init__(self, root_dir, min_area=2, trans=None):
        self.root_dir = root_dir
        self.min_area = min_area
        self.transforms = trans

        # Get the CT slices from the directory
        self.imgs = list(sorted(os.listdir(os.path.join(self.root_dir, 'CT'))))
        # Get the GT slices from
        self.masks = list(sorted(os.listdir(os.path.join(self.root_dir, 'GT'))))

    def __getitem__(self, idx):

        # Load images ad masks
        img_path = os.path.join(self.root_dir, 'CT', self.imgs[idx])
        mask_path = os.path.join(self.root_dir, 'GT', self.masks[idx])

        # Convert CT slice from RGBA to RGB
        img = Image.open(img_path).convert('RGB')

        # Covert masks from RGBA to RGB
        mask = Image.open(mask_path).convert('L')
        mask = np.array(mask)

        mask[mask != 0] = 255

        def mask_and_box_processing(s_mask, min_area):
            """
            Process the binary masks to instances mask and extract bounding boxes
            only for 1 class (0:bgr, 1: covid)
            inputs: mask (a NumPy array), threshold area
            output: binary masks and bounding boxes for each image instance
            
            adapted from:
            https://scikit-image.org/docs/dev/auto_examples/segmentation/plot_label.html#sphx-glr-auto-examples-segmentation-plot-label-py
            """

            # Label image regions
            label_mask = label(s_mask)

            boxes = []
            masks = []

            for region in regionprops(label_mask):
                # take regions with large enough areas
                if region.area >= min_area:
                    # get coordinates around segmented covid lesion
                    # x_min, y_min = region.bbox[1], region.bbox[0]
                    # x_max, y_max = region.bbox[3], region.bbox[2]
                    y_min, x_min, y_max, x_max = region.bbox
                    msk = np.zeros(label_mask.shape)
                    lab_idx = label_mask == region.label
                    msk[lab_idx] = 1
                    # Get the masks and boxes coordinates
                    boxes.append([x_min, y_min, x_max, y_max])
                    masks.append(msk)

            return boxes, masks

        # Extract masks and labels
        bboxes, masks_list = mask_and_box_processing(mask, self.min_area)

        # Convert to tensors
        masks_list = torch.as_tensor(np.array(masks_list), dtype=torch.uint8)
        bboxes = torch.as_tensor(np.array(bboxes), dtype=torch.float32)

        # number of regions with covid
        num_instances = len(bboxes)

        # There is only one class
        labels = torch.ones((num_instances,), dtype=torch.int64)
        image_id = torch.tensor([idx])
        # All instances are not crowd - coco metrics
        iscrowd = torch.zeros((len(bboxes)), dtype=torch.int64)
        # area used to separate the metric scores among small, medium and large boxes
        area = (bboxes[:, 3] - bboxes[:, 1]) * (bboxes[:, 2] - bboxes[:, 0])

        target = {'boxes': bboxes, 'labels': labels, 'masks': masks_list, 'image_id': image_id,
                  'area': area, 'iscrowd': iscrowd}

        if self.transforms is not None:
            img, target = self.transforms(img, target)

        return img, target

    def __len__(self):
        return len(self.imgs)

    def get_name(self, idx):
        # get image name
        return self.imgs[idx]
