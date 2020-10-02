import os
import sys
import time
import numpy as np
import random
import imgaug  # https://github.com/aleju/imgaug (pip3 install imgaug)
from pycocotools.coco import COCO
from coco import COCO_PLUS
from pycocotools.cocoeval import COCOeval
from pycocotools import mask as maskUtils

import zipfile
import urllib.request
import shutil
import skimage
import matplotlib.pyplot as plt

# Root directory of the project
ROOT_DIR = os.path.abspath("../../")

sys.path.append(ROOT_DIR)  # To find local version of the library
from mrcnn.config import Config
from mrcnn import model as modellib, utils
from mrcnn import visualize
import os

# Path to trained weights file
COCO_MODEL_PATH = os.path.join(ROOT_DIR, "logs/mask_rcnn_coco_0024.h5")
DEFAULT_LOGS_DIR = os.path.join(ROOT_DIR, "logs")
############################################################
# Frozen graph config
############################################################
import tensorflow as tf
import keras.backend as K

PATH_TO_SAVE_FROZEN_PB = os.path.join(ROOT_DIR, "logs/")
FROZEN_NAME = 'frozen_graph.pb'
sess = tf.Session()
K.set_session(sess)
############################################################

# os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
# os.environ["CUDA_VISIBLE_DEVICES"] = "0, 1"
############################################################
#  Configurations
############################################################


class CocoConfig(Config):
    """Configuration for training on MS COCO.
    Derives from the base Config class and overrides values specific
    to the COCO dataset.
    """
    # Give the configuration a recognizable name
    NAME = "coco"

    # We use a GPU with 12GB memory, which can fit two images.
    # Adjust down if you use a smaller GPU.
    IMAGES_PER_GPU = 2

    IMAGE_RESIZE_MODE = "square"
#     IMAGE_MAX_DIM = 512#1024

    # Uncomment to train on 8 GPUs (default is 1)
    # GPU_COUNT = 8

    # Number of classes (including background)
    NUM_CLASSES = 1 + 6  # nuCOCO has 7(1 bg and 6 others) classes:::: dataset_train.class_names ['BG', 'person', 'bicycle', 'car', 'motorcycle', 'bus', 'truck']


############################################################
#  Dataset
############################################################

class CocoDataset(utils.Dataset):
    def load_coco(self, dataset_dir, subset, class_ids=None,
                  class_map=None, return_coco=False):
        """Load a subset of the COCO dataset.
        dataset_dir: The root directory of the COCO dataset.
        subset: What to load (train, val, minival, valminusminival)
        year: What dataset year to load (2014, 2017) as a string, not an integer
        class_ids: If provided, only loads images that have the given classes.
        class_map: TODO: Not implemented yet. Supports maping classes from
            different datasets to the same class ID.
        return_coco: If True, returns the COCO object.        
        """
        coco = COCO_PLUS("{}/annotations/instances_{}.json".format(dataset_dir, subset))
        image_dir = "{}/{}".format(dataset_dir, subset)

        # Load all classes or a subset?
        if not class_ids:
            # All classes
            class_ids = sorted(coco.getCatIds())

        # All images or a subset?
        if class_ids:
            image_ids = []
            for id in class_ids:
                image_ids.extend(list(coco.getImgIds(catIds=[id])))
            # Remove duplicates
            image_ids = list(set(image_ids))
        else:
            # All images
            image_ids = list(coco.imgs.keys())

        # Add classes
        for i in class_ids:
            self.add_class("coco", i, coco.loadCats(i)[0]["name"])

        # Add images
        for i in image_ids:
            self.add_image(
                "coco", image_id=i,
                path=os.path.join(image_dir, coco.imgs[i]['file_name']),
                width=coco.imgs[i]["width"],
                height=coco.imgs[i]["height"],
                annotations=coco.loadAnns(coco.getAnnIds(
                    imgIds=[i], catIds=class_ids, iscrowd=None)),
                pointclouds=coco.pointcls[i])

        if return_coco:
            return coco
        

    def image_reference(self, image_id):
        """Return a link to the image in the COCO Website."""
        info = self.image_info[image_id]
        if info["source"] == "coco":
            return "http://cocodataset.org/#explore?id={}".format(info["id"])
        else:
            super(CocoDataset, self).image_reference(image_id)

        
    def load_bbox(self, image_id):
        """
        load bounding box information from annotations
        :return:
        """
        # If not a COCO image, delegate to parent class.        
        image_info = self.image_info[image_id]
        annotations = self.image_info[image_id]["annotations"]
        instance_bbox = []
        class_ids = []

        for annotation in annotations:
            class_id = self.map_source_class_id(
                "coco.{}".format(annotation['category_id']))
            if class_id:
                bbox = annotation['bbox']
                tbox = []

                # y1,x1,y2,x2 format
                tbox[:] = bbox[1], bbox[0], round(bbox[1] + bbox[3], 2), round(bbox[0] + bbox[2], 2)                
                c = np.zeros((image_info["height"], image_info["width"]))
                if round(tbox[0]) == 0:
                    tbox[0] = 1
                if round(tbox[1]) == 0:
                    tbox[1] = 1
                if round(tbox[2]) == 0:
                    tbox[2] = 1
                if round(tbox[3]) == 0:
                    tbox[3] = 1

                # Set x1,y1 and x2,y2 as 1 rest are 0
                ##check it's 0,1 and  2,3 or 1,0 and 3,2
                # it's 0,1 and  2,3 or 1,0 and 3,2 // (row, column) -->(height, width)-->(y,x)
                c[round(tbox[0] - 1), round(tbox[1] - 1)] = 1
                c[round(tbox[2] - 1), round(tbox[3] - 1)] = 1  # 10000299 IMAGE WITH BBOX OF HEIGHT 0.44 PIXEL IndexError: index 1 is out of bounds for axis 0 with size 1
                instance_bbox.append(c)
                class_ids.append(class_id)

        # Pack instance masks into an array
        if class_ids:            
            bbox = np.stack(instance_bbox, axis=0).astype(np.bool)
            class_ids = np.array(class_ids, dtype=np.int32)
        return bbox, class_ids


    def annToRLE(self, ann, height, width):
        """
        Convert annotation which can be polygons, uncompressed RLE to RLE.
        :return: binary mask (numpy 2D array)
        """
        segm = ann['segmentation']
        if isinstance(segm, list):
            # polygon -- a single object might consist of multiple parts
            # we merge all parts into one mask rle code
            rles = maskUtils.frPyObjects(segm, height, width)
            rle = maskUtils.merge(rles)
        elif isinstance(segm['counts'], list):
            # uncompressed RLE
            rle = maskUtils.frPyObjects(segm, height, width)
        else:
            # rle
            rle = ann['segmentation']
        return rle

    def annToMask(self, ann, height, width):
        """
        Convert annotation which can be polygons, uncompressed RLE, or RLE to binary mask.
        :return: binary mask (numpy 2D array)
        """
        rle = self.annToRLE(ann, height, width)
        m = maskUtils.decode(rle)
        return m


############################################################
#  COCO Evaluation
############################################################

# def build_coco_results(dataset, image_ids, rois, class_ids, scores, masks):
def build_coco_results(dataset, image_ids, rois, class_ids, scores):
    """Arrange resutls to match COCO specs in http://cocodataset.org/#format
    """
    # If no results, return an empty list
    if rois is None:
        return []

    results = []
    for image_id in image_ids:
        # Loop through detections
        for i in range(rois.shape[0]):
            class_id = class_ids[i]
            score = scores[i]
            bbox = np.around(rois[i], 1)
            
            result = {
                "image_id": image_id,
                "category_id": dataset.get_source_class_id(class_id, "coco"),
                "bbox": [bbox[1], bbox[0], bbox[3] - bbox[1], bbox[2] - bbox[0]],
                "score": score
            }
            results.append(result)
    return results


def get_ax(rows=1, cols=1, size=16):
    """Return a Matplotlib Axes array to be used in
    all visualizations in the notebook. Provide a
    central point to control graph sizes.

    Adjust the size attribute to control how big to render images
    """
    _, ax = plt.subplots(rows, cols, figsize=(size * cols, size * rows))
    return ax


def evaluate_coco(model, dataset, coco, eval_type="bbox", net="BIRANet", augmentation=None, limit=50, image_ids=None):
    """Runs official COCO evaluation.
    dataset: A Dataset object with valiadtion data
    eval_type: "bbox" or "segm" for bounding box or segmentation evaluation
    limit: if not 0, it's the number of images to use for evaluation
    """
    # Pick COCO images from the dataset
    image_ids = image_ids or dataset.image_ids

    # Limit to a subset
    if limit:        
        image_ids = image_ids[:limit]

    # Get corresponding COCO image IDs.
    coco_image_ids = [dataset.image_info[id]["id"] for id in image_ids]    
    t_prediction = 0
    t_start = time.time()

    results = []
    for i, image_id in enumerate(image_ids):
        # Load image
        image = dataset.load_image(image_id)
        bbox, class_ids = dataset.load_bbox(image_id)
        gt_bbox = utils.extract_bbox_boxes(bbox)
        pointclouds = dataset.image_info[image_id]["pointclouds"]
        points = pointclouds['points']

        ranchor_index, rpoint_fmap, _ = utils.rpoint_image_mapping(points, image.shape[:2],
                                                   (config.IMAGE_MAX_DIM, config.IMAGE_MAX_DIM), config)

        # Run detection returns detection on input image size(900,1600)
        t = time.time()
        r = model.detect([image], rpoint_fmap, ranchor_index, net, verbose=0)[0]
        t_prediction += (time.time() - t)        

        ##visualize detection results on image
        visualize.save_image(image, i, r['rois'], r['class_ids'], dataset.class_names, r['scores'])
        
        # Convert results to COCO format
        # Cast masks to uint8 because COCO tools errors out on bool
        image_results = build_coco_results(dataset, coco_image_ids[i:i + 1],
                                           r["rois"], r["class_ids"],
                                           r["scores"])  # ,        
        results.extend(image_results)

    # Load results. This modifies results with additional attributes.
    coco_results = coco.loadRes(results)

    # Evaluate
    cocoEval = COCOeval(coco, coco_results, eval_type)
    cocoEval.params.imgIds = coco_image_ids
    cocoEval.evaluate()
    cocoEval.accumulate()
    cocoEval.summarize()

    print("Prediction time: {}. Average {}/image".format(
        t_prediction, t_prediction / len(image_ids)))
    print("Total time: ", time.time() - t_start)


############################################################
# Frozen Graph

def freeze_session(session, keep_var_names=None, output_names=None, clear_devices=True):
    graph = sess.graph

    with graph.as_default():
        freeze_var_names = list(set(v.op.name for v in tf.global_variables()).difference(keep_var_names or []))

        output_names = output_names or []
        input_graph_def = graph.as_graph_def()

        if clear_devices:
            for node in input_graph_def.node:
                node.device = ""

        frozen_graph = tf.graph_util.convert_variables_to_constants(
            session, input_graph_def, output_names, freeze_var_names)
        return frozen_graph


def freeze_model(model, name):
    frozen_graph = freeze_session(
        sess,
        output_names=[out.op.name for out in model.outputs][:4])
    directory = PATH_TO_SAVE_FROZEN_PB
    tf.train.write_graph(frozen_graph, directory, name , as_text=False)
    print("*"*80)
    print("Finish converting keras model to Frozen PB")
    print('PATH: ', PATH_TO_SAVE_FROZEN_PB)
    print("*" * 80)
############################################################  
#  Training
############################################################

if __name__ == '__main__':
    import argparse

    # Parse command line arguments
    parser = argparse.ArgumentParser(
        description='Train Mask R-CNN on MS COCO.')
    parser.add_argument("command",
                        metavar="<command>",
                        help="'train' or 'evaluate' or 'test'")
    parser.add_argument('--dataset', required=True,
                        metavar="/path/to/coco/",
                        help='Directory of the MS-COCO dataset')
    parser.add_argument('--resolution', required=False,
                        default=1024, #500
                        metavar="<image resolution>",
                        help='Images to use for evaluation 1024/512 (default=1024)')
    parser.add_argument('--model', required=True,
                        metavar="/path/to/weights.h5",
                        help="Path to weights .h5 file or 'coco'")
    parser.add_argument('--logs', required=False,
                        default=DEFAULT_LOGS_DIR,
                        metavar="/path/to/logs/",
                        help='Logs and checkpoints directory (default=logs/)')
    parser.add_argument('--limit', required=False,
                        default=710, #500
                        metavar="<image count>",
                        help='Images to use for evaluation (default=500)')
    parser.add_argument('--net', required=False,
                        default="BIRANet",  ##500
                        metavar="<network name>",
                        help='Net used BIRANet/RANet (default=BIRANet)')
    args = parser.parse_args()
    print("Command: ", args.command)
    print("Model: ", args.model)
    print("Dataset: ", args.dataset)
    print("Resolution: ", args.resolution)
    print("Logs: ", args.logs)    
        
    if args.resolution=='512':
        Config.IMAGE_MAX_DIM = 512
        Config.RPN_ANCHOR_SCALES = (8, 16, 32, 64, 128)
    else:
        Config.IMAGE_MAX_DIM = 1024
        Config.RPN_ANCHOR_SCALES = (16, 32, 64, 128, 256)    
    
    # Configurations
    if args.command == "train":
        config = CocoConfig()        
    else:
        class InferenceConfig(CocoConfig):
            GPU_COUNT = 1
            IMAGES_PER_GPU = 1
            DETECTION_MIN_CONFIDENCE = 0
            
            if args.resolution=='512':
                IMAGE_MAX_DIM = 512
                RPN_ANCHOR_SCALES = (8, 16, 32, 64, 128)
            else:
                IMAGE_MAX_DIM = 1024
                RPN_ANCHOR_SCALES = (16, 32, 64, 128, 256)
            
        config = InferenceConfig()    
    config.display()

    # Create model
    if args.command == "train":
        model = modellib.MaskRCNN(mode="training", config=config,
                                  model_dir=args.logs)
    else:        
        model = modellib.MaskRCNN(mode="inference", config=config,
                                  model_dir=args.logs)

    # Select weights file to load    
    if args.model.lower() == "coco":
        model_path = COCO_MODEL_PATH
    elif args.model.lower() == "last":
        # Find last trained weights
        model_path = model.find_last()
    elif args.model.lower() == "imagenet":
        # Start from ImageNet trained weights
        model_path = model.get_imagenet_weights()
    else:
        model_path = args.model

    # Load weights
    print("Loading weights ", model_path)
    model.load_weights(model_path, by_name=True)
    
    ########### Frozen graph #############
#     freeze_model(model.keras_model, FROZEN_NAME)
    ########### Frozen graph #############
    
    # Load weights by name
#     model.load_weights(model_path, by_name=True, exclude=[
#      "mrcnn_class_logits", "mrcnn_bbox_fc"  # , "mrcnn_mask"
#     ])

    # Train or evaluate
    if args.command == "train":
        # Training dataset. Use the training set and 35K from the
        # validation set, as as in the Mask RCNN paper.
        dataset_train = CocoDataset()
        dataset_train.load_coco(args.dataset, "train")
        dataset_train.prepare()
        print("dataset_train.class_names", dataset_train.class_names)

        # Validation dataset
        dataset_val = CocoDataset()
        val_type = "val"
        dataset_val.load_coco(args.dataset, val_type)
        dataset_val.prepare()

        # Image Augmentation
        augmentation = imgaug.augmenters.OneOf([
            # imgaug.augmenters.Fliplr(0.5),
            imgaug.augmenters.AdditiveGaussianNoise(scale=(0, 0.05 * 255)),  # 0.05 * 255
            # imgaug.augmenters.Flipud(1.0),
            imgaug.augmenters.ElasticTransformation(alpha=(0, 5.0), sigma=0.25),
            imgaug.augmenters.AverageBlur(k=5)
            # imgaug.augmenters.Affine(rotate=(-10, 10))
        ])
        
        # *** This training schedule is an example. Update to your needs ***

        # Training - Stage 1
        # Finetune layers from ResNet stage 2 and up
        print("Fine tune Resnet stage 2/4 and up")
        model.train(dataset_train, dataset_val,
                    learning_rate=config.LEARNING_RATE,
                    epochs=40,
                    layers='2+',
                    net=args.net,
                    augmentation=augmentation
                    )

        # Training - Stage 2
        # Fine tune all layers
        print("Fine tune all layers")
        model.train(dataset_train, dataset_val,
                    learning_rate=config.LEARNING_RATE,
                    epochs=180,
                    layers='all',
                    net=args.net,
                    augmentation=augmentation)

        # Training - Stage 3
        # Fine tune all layers
        print("Fine tune all layers")
        model.train(dataset_train, dataset_val,
                    learning_rate=config.LEARNING_RATE/10,
                    epochs=250,
                    layers='all',
                    net=args.net,
                    augmentation=augmentation)

    elif args.command == "evaluate":
        # Validation dataset
        dataset_val = CocoDataset()
        val_type = "val"
        coco = dataset_val.load_coco(args.dataset, val_type, return_coco=True)
        dataset_val.prepare()
        print("Running COCO evaluation on {} images.".format(args.limit))        
        evaluate_coco(model, dataset_val, coco, "bbox",  net=args.net, augmentation=None, limit=int(args.limit))
        
    elif args.command == "test":
        # Load COCO dataset
        dataset = CocoDataset()
        dataset.load_coco(args.dataset, "val")
        dataset.prepare()

        test_dir= '/home/ritu/Desktop/Ritu/RRLAB_data/Final_data'
        img_dir= os.path.join(test_dir,'Image_Data/')
        radar_dir= os.path.join(test_dir,'projected_radarPoints/')
        
        # Load Image
        # file_names = next(os.walk(test_dir))[2]
        # image = skimage.io.imread(os.path.join(test_dir, random.choice(file_names)))        
        image = skimage.io.imread(os.path.join(img_dir, 'I2018-08-07T15:03:44250586730+02:00.png'))
        orig_shape = image.shape       

        # If grayscale. Convert to RGB for consistency.
        if image.ndim != 3:
            image = skimage.color.gray2rgb(image)
        # If has an alpha channel, remove it for consistency
        if image.shape[-1] == 4:
            image = image[..., :3]

        t_prediction = 0
        
        # fetch radar points and filter
        rad_file= os.path.join(radar_dir, 'R2018-08-07T15:03:44222635534+02:00.txt')
        points=[]
        with open(rad_file, 'r') as filehandle:
            for line in filehandle:
                tmp =[]
                x=line.split(' ')
                tmp =[float(x[0]), float(x[1]), float(x[2])]
                if tmp[0]>(orig_shape[1]-1) or tmp[1]>(orig_shape[0]-1) or tmp[0]<=0 or tmp[1]<=0:
                    continue;
                else:
                    points.append(tmp)
        # test if no radar data is available
#         points = [[100.0, 100.0, 100.0]]

        ranchor_index, x_points, rpoint_fmap  = utils.rpoint_image_mapping(points, orig_shape[:2],
                                                   (config.IMAGE_MAX_DIM, config.IMAGE_MAX_DIM), config)

        # Run detection returns detection on input image
        t = time.time()
        r = model.detect([image], rpoint_fmap, ranchor_index, net=args.net, verbose=0)[0]
        t_prediction += (time.time() - t)        
        N = r['rois'].shape[0]
        for i in range(N):
            print("bbox, class name, score", r['rois'][i], dataset.class_names[r['class_ids'][i]], r['scores'][i])

        visualize.draw_boxes(image, x_points, r['class_ids'], dataset.class_names, boxes=r['rois'])
        visualize.display_instances(image, r['rois'], r['class_ids'], dataset.class_names, r['scores'])
    
    else:
        print("'{}' is not recognized. "
              "Use 'train' or 'evaluate'".format(args.command))
