import cv2
import gdal
import mrcnn.model as modellib
import object_detect
import panorama
import os
import pyodm

from mrcnn import visualize
import matplotlib.image as mpimg

if __name__ == '__main__':

    """""""""""""""""""""""""""""""""""""""""""""""
    Create Map
    """""""""""""""""""""""""""""""""""""""""""""""
    # panorama.screen_video('./video/DJI_0907-001.MP4', './image', fps=175)
    '''
    panoram = panorama.create_panorama('./image/')
    cv2.imwrite('result/{}.jpg'.format('card'), panoram)

    panorama.georeferencer('result/card.jpg', 'result')
    card = panorama.Map(panoram)
    card.write_image_info('.result')
    os.system('pwd')
    os.system(sudo docker run -it --rm \
                -v "$(pwd)/odm/images:/code/images" \
                -v "$(pwd)/odm/odm_georeferencing:/code/odm_georeferencing" \
                -v "$(pwd)/odm/odm_meshing:/code/odm_meshing" \
                -v "$(pwd)/odm/odm_orthophoto:/code/odm_orthophoto" \
                -v "$(pwd)/odm/odm_texturing:/code/odm_texturing" \
                -v "$(pwd)/odm/opensfm:/code/opensfm" \
                opendronemap/odm
              )
    '''

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
    model.load_weights('./mrcnn/heights/mask_rcnn_object_0250.h5', by_name=True)
    img_list = ['DSC07663.JPG']
    # img_list = ['odm_orthophoto.original.tif']

    for img in img_list:
        image = mpimg.imread('./test_img/' + str(img))
        # Run object detection
        results = model.detect([image], verbose=1)
        r = results[0]
        visualize.display_instances(img, image, r['rois'], r['masks'], r['class_ids'],
                                    ['BG', 'tree', 'building'])
    # tif = gdal.Open(', 1)
    visualize.save_detect_info('/home/error/PycharmProjects/panorama/result/odm_orthophoto.original.tif', r['rois'], r['masks'])
    # track = [[-40, 15], [-250, 220], [-10, 600]]


