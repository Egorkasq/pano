import mrcnn.model as modellib
import object_detect
from mrcnn import visualize
import panorama
import cv2
import os

if __name__ == '__main__':

    """""""""""""""""""""""""""""""""""""""""""""""
    Create Map
    """""""""""""""""""""""""""""""""""""""""""""""
    # panorama.screen_video(videofile)
    panoram = panorama.create_panorama('./Images/')
    cv2.imwrite('result/{}.jpg'.format('card'), panoram)
    tif = panorama.georeferencer('result/card.jpg', 'result')
    card = panorama.Map(panoram, tile_size=256)
    card.write_image_info('.result')
    card.create_tiles()

    #print('the dist is ', panorama.get_distanse('1.JPG', 'IMG_0995.JPG'))
    #print('size card is;', card.card_size1())

    """""""""""""""""""""""""""""""""""""""""""""""
    Detect objects
    """""""""""""""""""""""""""""""""""""""""""""""

    class InferenceConfig(object_detect.BalloonConfig):
        GPU_COUNT = 1
        IMAGES_PER_GPU = 1
        NUM_CLASSES = 3


    config = InferenceConfig()
    model = modellib.MaskRCNN(mode="inference", model_dir='./mrcnn/logs', config=config)
    model.load_weights('./mrcnn/h5/mask_rcnn_object_0089.h5', by_name=True)
    img_list = ['IMG_0912.JPG', 'IMG_0913.JPG', 'IMG_1390.JPG', 'IMG_1391.JPG', 'IMG_1392.JPG']
    # img_list = ['odm_orthophoto1.jpg']
    import matplotlib.image as mpimg

    for img in img_list:
        image = mpimg.imread('./test_img/' + str(img))
        # Run object detection
        results = model.detect([image], verbose=1)
        r = results[0]
        visualize.display_instances(image, r['rois'], r['masks'], r['class_ids'],
                                    ['BG', 'tree', 'building'])
    tif = './result2.tif'
    # os.system('gdalwarp -s_srs EPSG:32632 -t_srs EPSG:4326 {} {}'.format(tif, 'result2.tif'))
    track = [[-40, 15], [-250, 220], [-10, 600]]
    visualize.save_detect_info(tif, r['rois'], r['masks'])
