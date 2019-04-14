
import os
import sys
import json
import datetime
import numpy as np
import skimage.draw
import os
import time
from pycocotools.cocoeval import COCOeval
from pycocotools import mask as maskUtils
from mrcnn import visualize
from mrcnn.model import log
import skimage
import matplotlib
import matplotlib.pyplot as plt
#os.environ["CUDA_VI0'SIBLE_DEVICES"] = '

# Root directory of the project
ROOT_DIR = os.path.abspath(".")

# Import Mask RCNN
sys.path.append(ROOT_DIR)  # To find local version of the library
from mrcnn.config import Config
from mrcnn import model as modellib, utils

# Path to trained weights file
COCO_WEIGHTS_PATH = os.path.join(ROOT_DIR, "mask_rcnn_coco.h5")

# Directory to save logs and model checkpoints, if not provided
# through the command line argument --logs
DEFAULT_LOGS_DIR = os.path.join(ROOT_DIR, "logs")

model_path = os.path.join(ROOT_DIR, "depend/logs")
head_path = os.path.join(ROOT_DIR, "depend/h5/mask_rcnn_object_0089.h5")
root_dir = os.path.abspath('./result')
video_path = os.path.abspath('./video')
image_path = os.path.abspath('./input_img')


class ShapesConfig(utils.Dataset):
    """Configuration for training on the toy shapes dataset.
    Derives from the base Config class and overrides values specific
    to the toy shapes dataset.
    """
    # Give the configuration a recognizable name
    NAME = "shapes"

    # Train on 1 GPU and 8 images per GPU. We can put multiple images on each
    # GPU because the images are small. Batch size is 8 (GPUs * images/GPU).
    GPU_COUNT = 1
    IMAGES_PER_GPU = 8

    # Number of classes (including background)
    NUM_CLASSES = 1 + 5  # background + 3 shapes

    # Use small images for faster training. Set the limits of the small side
    # the large side, and that determines the image shape.
    IMAGE_MIN_DIM = 128
    IMAGE_MAX_DIM = 128
    IMAGE_SHAPE = [400, 400, 3]

    # Use smaller anchors because our image and objects are small
    RPN_ANCHOR_SCALES = (8, 16, 32, 64, 128)  # anchor side in pixels

    # Reduce training ROIs per image because the images are small and have
    # few objects. Aim to allow ROI sampling to pick 33% positive ROIs.
    TRAIN_ROIS_PER_IMAGE = 32

    # Use a small epoch since the data is simple
    STEPS_PER_EPOCH = 100

    # use small validation steps since the epoch is small
    VALIDATION_STEPS = 5


class ShapesDataset(Config):
    """Generates the shapes synthetic dataset. The dataset consists of simple
    shapes (triangles, squares, circles) placed randomly on a blank surface.
    The images are generated on the fly. No file access required.
    """

    def load_shapes(self, dataset_dir, subset):
        """Load a subset of the Balloon dataset.
        dataset_dir: Root directory of the dataset.
        subset: Subset to load: train or val
        """
        # Add classes. We have only one class to add.
        self.add_class("object", 1, "tree")  # adjusted here
        self.add_class("object", 2, "water")  # adjusted here
        self.add_class("object", 3, "building")  # adjusted here
        self.add_class("object", 4, "road")  # adjusted here
        self.add_class("object", 5, "trail")  # adjusted here

        # Train or validation dataset?
        assert subset in ["train", "val"]
        dataset_dir = os.path.join(dataset_dir, subset)

        dataset_dir = os.path.abspath(".")
        # We mostly care about the x and y coordinates of each region
        annotations = json.load(open(os.path.join(dataset_dir, "5_classes.json")))
        annotations = list(annotations.values())  # don't need the dict keys

        # The VIA tool saves images in the JSON even if they don't have any
        # annotations. Skip unannotated images.
        annotations = [a for a in annotations if a['regions']]

        # Add images
        for a in annotations:
            # Get the x, y coordinaets of points of the polygons that make up
            # the outline of each object instance. There are stores in the
            # shape_attributes (see json format above)
            polygons = [r['shape_attributes'] for r in a['regions'].values()]

            class_names_str = [r['region_attributes']['object_name'] for r in a['regions'].values()]
            class_name_nums = []
            for i in class_names_str:
                if i == 'tree':
                    class_name_nums.append(1)
                if i == 'water':
                    class_name_nums.append(2)
                if i == 'building':
                    class_name_nums.append(3)
                if i == 'road':
                    class_name_nums.append(4)
                if i == 'trail':
                    class_name_nums.append(5)

            # print(a['filename'])
            # load_mask() needs the image size to convert polygons to masks.
            # Unfortunately, VIA doesn't include it in JSON, so we must read
            # the image. This is only managable since the dataset is tiny.
            dataset_dir = os.path.abspath("./train")
            image_path = os.path.join(dataset_dir, a['filename'])
            image = skimage.io.imread(image_path)
            height, width = image.shape[:2]

            self.add_image(
                "object",
                image_id=a['filename'],  # use file name as a unique image id
                path=image_path,
                width=width, height=height,
                polygons=polygons,
                class_list=np.array(
                    class_name_nums))  # UNSURE IF  I CAN JUST ADD THIS  HERE. OTHERWISE NEED  TO MODIFY DATASET UTIL

    def image_reference(self, image_id):
        """Return the path of the image."""
        info = self.image_info[image_id]
        if info["source"] == "object":  # adjusted here
            return info["path"]
        else:
            super(self.__class__, self).image_reference(image_id)

    def load_mask(self, image_id):
        """Generate instance masks for an image.
       Returns:
        masks: A bool array of shape [height, width, instance count] with
            one mask per instance.
        class_ids: a 1D array of class IDs of the instance masks.
        """
        # If not a balloon dataset image, delegate to parent class.
        image_info = self.image_info[image_id]
        if image_info["source"] != "object":  # adjusted here
            return super(self.__class__, self).load_mask(image_id)

        # Convert polygons to a bitmap mask of shape
        # [height, width, instance_count]
        info = self.image_info[image_id]
        mask = np.zeros([info["height"], info["width"], len(info["polygons"])],
                        dtype=np.uint8)
        for i, p in enumerate(info["polygons"]):
            # Get indexes of pixels inside the polygon and set them to 1
            rr, cc = skimage.draw.polygon(p['all_points_y'], p['all_points_x'])
            mask[rr, cc, i] = 1

        class_array = info['class_list']
        # Return mask, and array of class IDs of each instance. Since we have
        # one class ID only, we return an array of 1s

        # this function returns the masks as normal,  the class array of 3 classes
        return mask.astype(np.bool), class_array


dataset_val = ShapesDataset()
dataset = os.path.join(root_dir)
dataset_val.load_shapes(dataset, "val")
dataset_val.prepare()

class InferenceConfig(ShapesDataset):
    GPU_COUNT = 1
    IMAGES_PER_GPU = 1
    NUM_CLASSES = 6
config = InferenceConfig()
model = modellib.MaskRCNN(mode="inference", model_dir=model_path, config=config)
model.load_weights(head_path, by_name=True)
class_names = ['BG', 'tree', 'water', 'building', 'road', 'trial']
file_names = 'card.jpg'
image = skimage.io.imread(os.path.join(root_dir, file_names))
results = model.detect([image], verbose=1)
r = results[0]
#visualize.save_detect_info1(image, r['rois'], r['masks'], track)


# Compute VOC-Style mAP @ IoU=0.5
# Running on 10 images. Increase for better accuracy.
image_ids = np.random.choice(dataset_val.image_ids, 10)
APs = []
for image_id in image_ids:
    # Load image and ground truth data
    image, image_meta, gt_class_id, gt_bbox, gt_mask = \
        modellib.load_image_gt(dataset_val, InferenceConfig,
                               image_id, use_mini_mask=False)
    molded_images = np.expand_dims(modellib.mold_image(image, InferenceConfig), 0)
    # Run object detection
    results = model.detect([image], verbose=0)
    r = results[0]
    # Compute AP
    AP, precisions, recalls, overlaps = \
        utils.compute_ap(gt_bbox, gt_class_id, gt_mask,
                         r["rois"], r["class_ids"], r["scores"], r['masks'])
    APs.append(AP)

print("mAP: ", np.mean(APs))
