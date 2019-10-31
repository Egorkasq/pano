import mrcnn.model as modellib
from mrcnn import visualize
import object_detect
import os
import sys
import skimage
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import matplotlib.image as mpimg
import geojson2shp

ROOT_DIR = os.path.abspath(".")
sys.path.append(ROOT_DIR)
sys.path.append(os.path.join(ROOT_DIR, "mrcnn/"))
sys.path.append(os.path.join(ROOT_DIR, "depend/coco/"))
model_path = os.path.join(ROOT_DIR, "depend/logs")
head_path = os.path.join(ROOT_DIR, "depend/h5/mask_rcnn_object_0030.h5")
root_dir = os.path.abspath('./result')
video_path = os.path.abspath('./video')
image_path = os.path.abspath('./input_img')


def get_ax(rows=1, cols=1, size=16):
    """Return a Matplotlib Axes array to be used in
    all visualizations in the notebook. Provide a
    central point to control graph sizes.

    Adjust the size attribute to control how big to render images
    """
    _, ax = plt.subplots(rows, cols, figsize=(size * cols, size * rows))
    return ax


if __name__ == '__main__':

    """""""""""""""""""""""""""""""""""""""""""""""
    Create Map
    """""""""""""""""""""""""""""""""""""""""""""""
    #panorama.screen_video(videofile, image_path)
    '''
    panoram = panorama.create_panorama(image_path)
    cv2.imwrite('{}.jpg'.format('card'), panoram)
    tif = panorama.georeferencer('card.jpg', root_dir)
    card = panorama.Map(panoram, tile_size=256)
    card.write_image_info()
    card.create_tiles()

    #print('the dist is ', panorama.get_distanse('1.JPG', 'IMG_0995.JPG'))
    #print('size card is;', card.card_size1())

    """""""""""""""""""""""""""""""""""""""""""""""
    Detect objects
    """""""""""""""""""""""""""""""""""""""""""""""
    '''
    class InferenceConfig(object_detect.BalloonConfig):
        GPU_COUNT = 1
        IMAGES_PER_GPU = 1
        NUM_CLASSES = 3
    config = InferenceConfig()
    model = modellib.MaskRCNN(mode="inference", model_dir=model_path, config=config)
    model.load_weights(head_path, by_name=True)
    # class_names = ['BG', 'tree', 'water', 'building', 'road', 'trial']

    img_list = ['IMG_1151.JPG', 'IMG_0912.JPG', 'IMG_0931.JPG', 'IMG_1415.JPG', 'IMG_1174.JPG']

    import matplotlib.image as mpimg

    for img in img_list:
        image = mpimg.imread('test_image/' + str(img))
        # Run object detection
        print(len([image]))
        results = model.detect([image], verbose=1)

        ax = get_ax(1)
        r = results[0]
        visualize.display_instances(image, r['rois'], r['masks'], r['class_ids'],
                                    ['BG', 'tree', 'building'], r['scores'], ax=ax,
                                    title="")

    tif = 'odm_orthophoto.original.tif'
    track = [[-40, 15], [-250, 220], [-10, 600]]
    visualize.save_detect_info(tif, r['rois'], r['masks'])
    # Create an object from the GeoJ class
    gJ = geojson2shp.GeoJ('input/lines.geojson')

    # Creating a shapefile from the geoJSON object
    gJ.toShp('output/lines')
