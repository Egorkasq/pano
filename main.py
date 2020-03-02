import cv2

import mrcnn.model as modellib
import object_detect
import panorama
import os
import gdal
from mrcnn import visualize
import matplotlib.image as mpimg


if __name__ == '__main__':

    """""""""""""""""""""""""""""""""""""""""""""""
    Create Map
    """""""""""""""""""""""""""""""""""""""""""""""
    # panorama.screen_video('./video/DJI_0907-001.MP4', './image', fps=175)
    panoram = panorama.create_panorama('./Images/')
    cv2.imwrite('result/{}.jpg'.format('card'), panoram)
    panorama.georeferencer('result/card.jpg', 'result')
    card = panorama.Map(panoram)
    card.write_image_info('.result')
    # card.create_tiles()
    # print('the dist is ', panorama.get_distanse('1.JPG', 'IMG_0995.JPG'))
    # print('size card is;', card.card_size())

    """""""""""""""""""""""""""""""""""""""""""""""
    Detect objects
    """""""""""""""""""""""""""""""""""""""""""""""

    class InferenceConfig(object_detect.BalloonConfig):
        GPU_COUNT = 1
        IMAGES_PER_GPU = 1
        NUM_CLASSES = 3

    config = InferenceConfig()
    model = modellib.MaskRCNN(mode="inference", model_dir='./mrcnn/logs', config=config)
    model.load_weights('./mrcnn/heights/mask_rcnn_object_0241.h5', by_name=True)
    img_list = ['IMG_0912.JPG', 'IMG_1540.JPG']
    # img_list = ['odm_orthophoto1.jpg']

    for img in img_list:
        image = mpimg.imread('./test_img/' + str(img))
        # Run object detection
        results = model.detect([image], verbose=1)
        r = results[0]
        visualize.display_instances(img, image, r['rois'], r['masks'], r['class_ids'],
                                    ['BG', 'tree', 'building'])

    tif = './result/card.Gtiff'
    os.system('gdal.warp -s_srs EPSG:32632 -t_srs EPSG:4326 {} {}'.format(tif, 'result2.tif'))
    print(tif)
    track = [[-40, 15], [-250, 220], [-10, 600]]
    # visualize.save_detect_info(tif, r['rois'], r['masks'])
