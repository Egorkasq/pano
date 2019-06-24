import panorama
import os
import sys
import cv2
import time
import skimage.io
import timeit
ROOT_DIR = os.path.abspath(".")
sys.path.append(ROOT_DIR)
sys.path.append(os.path.join(ROOT_DIR, "mrcnn/"))
import mrcnn.model as modellib
from mrcnn import visualize
sys.path.append(os.path.join(ROOT_DIR, "depend/coco/"))
import coco
import object_detect

model_path = os.path.join(ROOT_DIR, "depend/logs")
head_path = os.path.join(ROOT_DIR, "depend/h5/mask_rcnn_object_0040.h5")
root_dir = os.path.abspath('./result')
video_path = os.path.abspath('./video')
image_path = os.path.abspath('./input_img')

"""""""""""""""""""""""""""""""""""""""""""""""
Create Map
"""""""""""""""""""""""""""""""""""""""""""""""

'''
#videofile = os.path.join(video_path, "ОС2014 Видео с БПЛА (Фотокамера) (online-video-cutter.com).mp4")
#panorama.screen_video(videofile, image_path)
panoram = panorama.create_panorama(image_path)
cv2.imwrite('{}.jpg'.format('card'), panoram)

card = panorama.Map(panoram, tile_size=256)
card.write_image_info()
card.create_tiles()
#print('the dist is ', panorama.get_distanse('1.JPG', 'IMG_0995.JPG'))
#print('size card is;', card.card_size1())

'''


"""""""""""""""""""""""""""""""""""""""""""""""
Detect objects
"""""""""""""""""""""""""""""""""""""""""""""""

class InferenceConfig(object_detect.BalloonConfig):
    GPU_COUNT = 1
    IMAGES_PER_GPU = 1
    NUM_CLASSES = 3
config = InferenceConfig()
model = modellib.MaskRCNN(mode="inference", model_dir=model_path, config=config)
model.load_weights(head_path, by_name=True)
#class_names = ['BG', 'tree', 'water', 'building', 'road', 'trial']
class_names = ['BG', 'tree', 'building']
file_names = 'IMG_1391.JPG'
image = skimage.io.imread(os.path.join(root_dir, file_names))
results = model.detect([image], verbose=1)
r = results[0]
visualize.display_instances(image, r['rois'], r['masks'], r['class_ids'],
                            class_names, r['scores'])
track = [[-40, 15], [-250, 220], [-10, 600]]
visualize.save_detect_info(image, r['rois'], r['masks'], track)
#visualize.save_detect_info1(image, r['rois'], r['masks'], track)

