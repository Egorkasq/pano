import copy

import numpy as np
import cv2
import os
import json
import shutil
import math
from geopandas import GeoSeries
import codecs
from PIL import Image
from PIL.ExifTags import TAGS, GPSTAGS
from osgeo import gdal
from geojson import Polygon, MultiPolygon
from osgeo import gdal, osr
from skimage.measure import find_contours
import geopandas as gpd
titles_path = os.path.abspath('.result/tiles')


class Map:

    def __init__(self, image, zoom=0, tile_size=256):
        self.image = image
        self.zoom = zoom
        self.tile_size = tile_size
        self.column, self.line = image.shape[:2]

    def write_image_info(self, path):
        """
        save info about image in json format
        """
        data = {"tileSize": self.tile_size, "mapWidth": self.line,
                "mapHeight": self.column, "maxZoom": self.zoom}
        with open(path + '/info.json', 'w') as f:
            json.dump(data, f)

    def create_tiles(self):
        """
        create tiles, and save it in (y_x.jpg) format
        """
        if os.path.exists(titles_path):
            shutil.rmtree(titles_path)
        os.makedirs(titles_path)

        for i in range(0, self.column, self.tile_size):
            for j in range(0, self.line, self.tile_size):
                temp_img = self.image[i: i + self.tile_size, j: j + self.tile_size]
                name = str(i // self.tile_size) + '_' + str(j // self.tile_size)
                cv2.imwrite('/tiles/{}.jpg'.format(str(name)), temp_img)

        return 0

    def card_size(self, image_path):
        folder = []
        map_size = []
        average = 0
        os.chdir(image_path)
        paths = list(os.walk(image_path))
        for image in paths[0][2][0:]:
            temp = cv2.imread(image, 1)
            folder.append(image)
            meta_data = ImageMetaData(image)
            latlng = meta_data.get_lat_lng()
            average = average + latlng[2]
        average = average / len(folder)
        map_size.append(math.tan(math.radians(60.3 / 2)) * 2 * 162 / temp.shape[0] * self.column)
        map_size.append(math.tan(math.radians(70.42 / 2)) * 2 * 162 / temp.shape[1] * self.line)
        return map_size


class ImageMetaData(object):
    """
    Extract the exif data from any image. Data includes GPS coordinates,
    Focal Length, Manufacture, and more.
    """
    exif_data = None
    image = None

    def __init__(self, img_path):
        self.image = Image.open(img_path)
        self.get_exif_data()
        super(ImageMetaData, self).__init__()

    def get_exif_data(self):
        """Returns a dictionary from the exif data of an PIL Image item. Also converts the GPS Tags"""
        exif_data = {}
        info = self.image._getexif()
        if info:
            for tag, value in info.items():
                decoded = TAGS.get(tag, tag)
                if decoded == "GPSInfo":
                    gps_data = {}
                    for t in value:
                        sub_decoded = GPSTAGS.get(t, t)
                        gps_data[sub_decoded] = value[t]

                    exif_data[decoded] = gps_data
                else:
                    exif_data[decoded] = value
        self.exif_data = exif_data
        return exif_data

    @staticmethod
    def get_if_exist(data, key):
        if key in data:
            return data[key]
        return None

    @staticmethod
    def convert_to_degress(value):

        """Helper function to convert the GPS coordinates
        stored in the EXIF to degress in float format"""
        d0 = value[0][0]
        d1 = value[0][1]
        d = float(d0) / float(d1)

        m0 = value[1][0]
        m1 = value[1][1]
        m = float(m0) / float(m1)

        s0 = value[2][0]
        s1 = value[2][1]
        s = float(s0) / float(s1)

        return d + (m / 60.0) + (s / 3600.0)

    def get_lat_lng(self, height_=True):
        """
        Returns the latitude and longitude, if available,
        from the provided exif_data (obtained through get_exif_data above)
        """
        lat = None
        lng = None
        height = None
        exif_data = self.get_exif_data()
        if "GPSInfo" in exif_data:
            gps_info = exif_data["GPSInfo"]
            gps_latitude = self.get_if_exist(gps_info, "GPSLatitude")
            gps_latitude_ref = self.get_if_exist(gps_info, 'GPSLatitudeRef')
            gps_longitude = self.get_if_exist(gps_info, 'GPSLongitude')
            gps_longitude_ref = self.get_if_exist(gps_info, 'GPSLongitudeRef')
            gps_altitude = self.get_if_exist(gps_info, 'GPSAltitude')
            gps_altitude_ref = self.get_if_exist(gps_info, 'GPSAltitudeRef')
            # height = self.convert_to_degress(gps_altitude)
            # height = gps_altitude[0] // gps_altitude[1]

            if gps_latitude and gps_latitude_ref and gps_longitude and gps_longitude_ref:
                lat = self.convert_to_degress(gps_latitude)
                if gps_latitude_ref != "N":
                    lat = 0 - lat
                lng = self.convert_to_degress(gps_longitude)
                if gps_longitude_ref != "E":
                    lng = 0 - lng

        # if height_:
        #    return lat, lng, height
        # else:
        return lat, lng


def screen_video(video_file, image_path, fps=15):
    """
    This func create tiles from videofile
    :return: return every 'fps' frame
    """
    c = 0
    name = 0
    cap = cv2.VideoCapture(video_file)

    if os.path.exists(image_path):
        shutil.rmtree(image_path)
    os.makedirs(image_path)
    flag, img = cap.read()
    print(flag, len(img))
    '''
    for i in img:
        if flag != 0:
            print(i)
            cv2.imwrite(image_path + '/{:06d}.jpg'.format(name), i)
            name += 1
            print(name)
    '''
    while True:
        flag, img = cap.read()
        if flag == 0:
            break

        if c % fps == 0:
            cv2.imwrite(image_path + '/{:06d}.jpg'.format(name), img)
            print(len(img))
            print(img)
            name += 1
        c = c + 1
    return 0


def create_panorama(image_path, write_info=False):
    paths = list(os.walk(image_path))
    folder = paths[0][2][0:]
    folder.sort()
    print('find {} images for stitching:{}'.format(len(folder), folder))
    base_image = cv2.imread(image_path + folder[0], 1)

    img_data = ImageMetaData(image_path + folder[0])
    cent_y = int(base_image.shape[0] // 2)
    cent_x = int(base_image.shape[1] // 2)
    data = [
        str(folder[0]),
        cent_x,
        cent_y,
        img_data.get_lat_lng()
        ]
    folder.pop(0)

    for file in folder:
        next_image = cv2.imread(image_path + file, 1)
        # next_image_resize = cv2.resize(next_image, (500, 500))
        next_image_resize = next_image
        # base_image_resize = cv2.resize(base_image, (500, 500))
        base_image_resize = base_image
        orb = cv2.ORB_create()
        kp1, des1 = orb.detectAndCompute(next_image_resize, None)
        kp2, des2 = orb.detectAndCompute(base_image_resize, None)
        bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True, )
        matches = bf.match(des1, des2)
        matches = sorted(matches, key=lambda x: x.distance)
        matches = matches[:int(len(matches) * 0.5)]
        assert len(matches) > 10

        dst_pts = np.float32([kp1[m.queryIdx].pt for m in matches]).reshape(-1, 1, 2)
        src_pts = np.float32([kp2[m.trainIdx].pt for m in matches]).reshape(-1, 1, 2)
        M, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)
        h1, w1 = next_image.shape[:2]
        h2, w2 = base_image.shape[:2]
        pts1 = np.float32([[0, 0], [0, h1], [w1, h1], [w1, 0]]).reshape(-1, 1, 2)
        pts2 = np.float32([[0, 0], [0, h2], [w2, h2], [w2, 0]]).reshape(-1, 1, 2)
        pts2 = cv2.perspectiveTransform(pts2, M)
        pts = np.concatenate((pts1, pts2), axis=0)
        [xmin, ymin] = np.int32(pts.min(axis=0).ravel() - 0.5)
        [xmax, ymax] = np.int32(pts.max(axis=0).ravel() + 0.5)
        t = [-xmin, -ymin]
        Ht = np.array([[1, 0, t[0]], [0, 1, t[1]], [0, 0, 1]])
        result = cv2.warpPerspective(base_image, Ht.dot(M), (xmax - xmin, ymax - ymin))

       # next_image = next_image[:, :next_image.shape[1] // 4 * 3] if t[0] > 0 else \
       #     next_image[:, next_image.shape[1] // 4:]
       # next_image = next_image[:next_image.shape[0] // 4 * 3, :] if t[0] > 0 else \
       #     next_image[next_image.shape[0] // 4:, :]

        result[t[1]:next_image.shape[0] + t[1], t[0]:next_image.shape[1] + t[0]] = next_image
        img_data = ImageMetaData(image_path + file)
        cent_x = cent_x + t[1]
        cent_y = cent_y + t[0]
        img_data = [
            str(file),
            int(cent_x),
            int(cent_y),
            img_data.get_lat_lng()
        ]
        data = data + img_data
        base_image = result


    result = crop(result)
    result = resize(result, 256)
    print('map created {}'.format(result.shape))
    json.dump(data, codecs.open('result/geo_info.txt', 'w', encoding='utf-8'), separators=(',', ':'), indent=4)

    return result


def georeferencer(image_temp, geo_json_dir):
    """
    :param image_temp:
    :param geo_json_dir:
    :return:
    x, y, long, lat
    """
    src_ds = gdal.Open(image_temp)
    data = list()
    json_info = json.load(open(os.path.join(geo_json_dir, 'geo_info.txt')))
    for i in range(0, len(json_info), 4):
        print(json_info[i], json_info[i + 2], json_info[i + 1], 0, json_info[i + 2], json_info[i + 3][0])
        # gcplist = [gdal.GCP(json_info[i + 3][0], json_info[i + 3][1], json_info[i + 3][2], json_info[i + 2], json_info[i + 1])]
        gcplist = [gdal.GCP(json_info[i + 3][0], json_info[i + 3][1], 0, json_info[i + 2], json_info[i + 1])]
        data = data + gcplist
    print(type(data), data)

    gdal.Translate('{}.Gtiff'.format(image_temp.split('.')[0]), src_ds, format="GTiff")
    gdal.Translate('{}_georeference.Gtiff'.format(image_temp.split('.')[0]), src_ds, outputSRS='EPSG:32632', format="GTiff", GCPs=data)
    format = "GTiff"
    driver = gdal.GetDriverByName(format)
    dst_ds = driver.CreateCopy(image_temp[:-4] + '_georeference2', src_ds, 0)
    gt = data
    dst_ds.SetGeoTransform(gt)
    epsg = 32632
    srs = osr.SpatialReference()
    srs.ImportFromEPSG(epsg)
    dest_wkt = srs.ExportToWkt()
    dst_ds.SetProjection(dest_wkt)
    # Close files
    dst_ds = None
    src_ds = None

    # gdal.WarpOptions(image, xRes=image.shape[1], yRes=image.shape[0], 'image1.gtiff')
    # gdal.wrapper_GDALWarpDestDS("EPSG:4326")
    # warpImage = gdal.Warp('warp_Image.tif', image)
    # cv2.imwrite('{}.tif'.format(str(warpImage)), warpImage)


def coord2pixelOffset(rasterfn, x, y):
    raster = gdal.Open(rasterfn)
    geotransform = raster.GetGeoTransform()
    originX = geotransform[0]
    originY = geotransform[3]
    pixelWidth = geotransform[1]
    pixelHeight = geotransform[5]
    xOffset = int((x - originX)/pixelWidth)
    yOffset = int((y - originY)/pixelHeight)
    result = [xOffset, yOffset]
    return result


def pixelOffset2coord(rasterfn, xOffset, yOffset):
    raster = gdal.Open(rasterfn)
    geotransform = raster.GetGeoTransform()
    originX = geotransform[0]
    originY = geotransform[3]
    pixelWidth = geotransform[1]
    pixelHeight = geotransform[5]
    coordX = originX+pixelWidth*xOffset
    coordY = originY+pixelHeight*yOffset
    result = (coordX, coordY)
    return result


def resize(image, tile_size):
    """
    Makes the image multiple tile_size
    """
    column, line = image.shape[:2]
    temp_col = column % tile_size
    temp_line = line % tile_size
    blank_image = np.zeros((column + tile_size - temp_col, line + tile_size - temp_line, 3), np.uint8)
    blank_image[:column, :line] = image
    return blank_image


def crop(image, threshold=0):
    """
    To crop all black zone around image
    """
    if len(image.shape) == 3:
        flatImage = np.max(image, 2)
    else:
        flatImage = image
    assert len(flatImage.shape) == 2
    rows = np.where(np.max(flatImage, 0) > threshold)[0]
    if rows.size:
        cols = np.where(np.max(flatImage, 1) > threshold)[0]
        image = image[cols[0]: cols[-1] + 1, rows[0]: rows[-1] + 1]
    else:
        image = image[:1, :1]
    return image


def get_distanse(image1, image2):
    """
    :param image1:
    :param image2:
    :return: distance between two images
    """
    meta_data_1 = ImageMetaData(image1)
    meta_data_2 = ImageMetaData(image2)
    latlng_1 = meta_data_1.get_lat_lng()
    latlng_2 = meta_data_2.get_lat_lng()
    print('coord first img{}'.format(latlng_1))
    print('coord second img{}'.format(latlng_2))

    if latlng_1 and latlng_2 is not None:
        dist = math.acos(math.sin(math.radians(latlng_1[0])) * math.sin(math.radians(latlng_2[0])) +
                         math.cos(math.radians(latlng_1[0])) * math.cos(math.radians(latlng_2[0])) *
                         math.cos(math.radians(latlng_1[1] - latlng_2[1]))) * 6371 * 1000
    return dist


def tif2jpg(image):
    if os.path.splitext(os.path.join(image))[1].lower() == ".tif":
        outfile = os.path.splitext(os.path.join(image))[0] + ".jpg"
        im = Image.open(os.path.join(image))
        print("Generating jpeg for %s" % image)
        im.thumbnail(im.size)
        im.save(outfile, "JPG", quality=100)
    else:
        print("can't find {}".format(image))


def save_detect_info(image, boxes, masks):
    N = boxes.shape[0]
    pix = []
    point = []
    geo_coord = []
    for i in range(N):
        mask = masks[:, :, i]
        padded_mask = np.zeros((mask.shape[0] + 2, mask.shape[1] + 2), dtype=np.uint8)
        padded_mask[1:-1, 1:-1] = mask
        contours = find_contours(padded_mask, 0.5)
        for verts in contours:
            for j in verts:
                pix.append((j[0], j[1]))

        if len(pix) <= 100:
            c = 4
        elif 100 < len(pix) <= 200:
            c = 6
        elif 200 < len(pix) <= 400:
            c = 8
        elif 400 < len(pix) <= 1000:
            c = 10
        elif 1000 < len(pix) <= 1500:
            c = 14
        elif 1500 < len(pix) <= 2000:
            c = 16
        elif 2000 < len(pix) <= 4000:
            c = 18
        elif 4000 < len(pix) <= 6000:
            c = 20
        elif 6000 < len(pix) <= 8000:
            c = 22
        elif 6000 < len(pix):
            c = 30
        pix = pix[::len(pix) // c]
        for k in pix:
            geo_coord.append(pixelOffset2coord(image, k[1], k[0]))
        point.append(copy.deepcopy(geo_coord))
        geo_coord.clear()
    print(point)
    poly = []
    for i in point:
        l = Polygon(i)
        poly.append(l)

    e = GeoSeries(poly)
    e.to_file('shape.shp')
    print("detect info created")

    '''
    w = shapefile.Writer(str(image) + 'shapefile')
    for k in point:
        temp += 1
        print(k)
        w.field('F_FLD', 'C', '10')
        print(type(k), k)
        # w.poly(k)
        w.record('polygon_{}'.format(temp))
        w.close()

        epsg = 'GEOGCS["WGS 84",'
        epsg += 'DATUM["WGS_1984",'
        epsg += 'SPHEROID["WGS 84",6378137,298.257223563]]'
        epsg += ',PRIMEM["Greenwich",0],'
        epsg += 'UNIT["degree",0.0174532925199433]]'

    '''


def split_list(seq, num):
    avg = len(seq) / float(num)
    out = []
    last = 0.0

    while last < len(seq):
        out.append(seq[int(last):int(last + avg)])
        last += avg
    return out

'''
def save_detect_info1(image, boxes, masks, track):
    os.chdir(root_dir)
    coord = track
    temp = []
    N = boxes.shape[0]
    for i in range(N):
        print(N)
        if N == 0:
            print("no instances to save ")
            break
        mask = masks[:, :, i]
        for i in range(mask.shape[0]):
            for j in range(1, mask.shape[1]):
                if mask[i, j - 1] == True and mask[i, j] == False or mask[i, j - 1] == False and mask[i, j] == True:
                    temp.append(i - image.shape[0])
                    temp.append(j)

    temp = split_list(temp, len(temp) / 2)
    data = {
        "impassableAreas": [{
            "type": "Area",
            "coordinates":
                temp
        }],

        "platformTracks": [{
            "type": "Point",
            "coordinates":
                coord
        }]
    }
    json.dump(data, codecs.open('detect_info1.json', 'w', encoding='utf-8'), separators=(',', ':'), indent=4)
    print("detect info created1")
'''

