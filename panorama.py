import numpy as np
import cv2
import os
import json
import numpy
import shutil
import math
import random
from PIL import Image
from PIL.ExifTags import TAGS, GPSTAGS


root_dir = os.path.abspath('./result')
video_path = os.path.abspath('./video')
titles_path = os.path.abspath('./tiles')
image_path = os.path.abspath('./input_img')
image_path1 = os.path.abspath('./temp')


class Map:
    def __init__(self, image, zoom=0, tile_size=256):
        self.image = image
        self.zoom = zoom
        self.tile_size = tile_size
        self.column, self.line = image.shape[:2]

    def write_image_info(self):
        """
        save info about image in json format
        """
        data = {"tileSize": self.tile_size, "mapWidth": self.line,
                "mapHeight": self.column, "maxZoom": self.zoom}
        with open('info.json', 'w') as f:
            json.dump(data, f)

    def create_tiles(self):
        """
        create tiles, and save it in (y_x.jpg) format
        """
        if os.path.exists(titles_path):
            shutil.rmtree(titles_path)
        os.makedirs(titles_path)
        os.chdir(titles_path)
        for i in range(0, self.column, self.tile_size):
            for j in range(0, self.line, self.tile_size):
                tempImg = self.image[i: i + self.tile_size, j: j + self.tile_size]
                name = str(i // self.tile_size) + '_' + str(j // self.tile_size)

                cv2.imwrite(str(name) + '.jpg', tempImg)
        os.chdir(root_dir)
        return 0

    def convert_to_map(self, x, y, revert=False):
        """
        False: transform img's coordinates to map coord
        True: transform map's coord to img
        :param x: coord x
        :param y: coord y
        """
        height_map, wight_map = self.card_size()
        kef_x = height_map / self.line
        kef_y = wight_map / self.column
        if not revert:
            x = round(x * kef_x)
            y = round(y * kef_y)
        else:
            x = round(x / kef_x)
            y = round(y / kef_y)
        return x, y

    def find_tile(self, x, y):
        """
        Return tiles, witch contains coord x and y
        """
        folder = []
        paths = list(os.walk(titles_path))
        for images in paths[0][2]:
            folder.append(images)
        folder.sort()
        print(folder)
        temp = 0
        x = int(x / self.line)
        y = int(y / self.column)
        mat_size_x = self.line // self.tile_size
        mat_size_y = self.column // self.tile_size
        matrix = numpy.array(range(mat_size_y * mat_size_x))
        matrix.shape = (mat_size_x, mat_size_y)
        #for i in range(mat_size_y):
        #    for j in range(mat_size_x):
        #        matrix[i][j] = folder[temp]
        #        temp += 1
        print(matrix)
        return matrix[x][y]
    '''
    @staticmethod
    def card_size():
        folder = []
        map_size = []
        average = 0
        os.chdir(image_path)
        paths = list(os.walk(image_path))
        for image in paths[0][2][0:]:
            folder.append(image)
            meta_data = ImageMetaData(image)
            latlng = meta_data.get_lat_lng()
            if latlng is True:
                average = average + latlng[2]
        average = average / len(folder)
        map_size[0] = math.tan(70.42) * average[2]
        map_size[1] = math.tan(43.3) * average[2]
        return map_size
    '''

    def card_size(self):
        folder = []
        map_size = []
        average = 0
        os.chdir(image_path1)
        paths = list(os.walk(image_path1))
        for image in paths[0][2][0:]:
            folder.append(image)
            meta_data = ImageMetaData(image)
            latlng = meta_data.get_lat_lng()
            average = average + latlng[2]
        average = average / len(folder)
        print('height ', average)
        map_size.append(math.tan(math.radians(70.42 / 2)) * average * 2 * self.line / self.tile_size)
        map_size.append(math.tan(math.radians(43.3 / 2)) * average * 2 * self.column / self.tile_size)
        return map_size

    def card_size1(self):
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
    '''
    Extract the exif data from any image. Data includes GPS coordinates,
    Focal Length, Manufacture, and more.
    '''
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

    def get_if_exist(self, data, key):
        if key in data:
            return data[key]
        return None

    def convert_to_degress(self, value):

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

    def get_lat_lng(self):
        """
        Returns the latitude and longitude, if available,
        from the provided exif_data (obtained through get_exif_data above)
        """
        gps_altitude = None
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

            #height = self.convert_to_degress(gps_altitude)
            height = gps_altitude[0] // gps_altitude[1] / 10

            if gps_latitude and gps_latitude_ref and gps_longitude and gps_longitude_ref:
                lat = self.convert_to_degress(gps_latitude)
                if gps_latitude_ref != "N":
                    lat = 0 - lat
                lng = self.convert_to_degress(gps_longitude)
                if gps_longitude_ref != "E":
                    lng = 0 - lng

        return lat, lng, height


def screen_video(video_file, fps=25):
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
    os.chdir(image_path)
    while True:
        flag, img = cap.read()
        if flag == 0:
            break
        if c % fps == 0:                        # каждый 25 кадр
            cv2.imwrite('{}/{:06d}.jpg'.format(image_path, name), img)
            name += 1
        c = c + 1
    os.chdir(root_dir)
    return 0


def create_panorama():
    folder = []
    os.chdir(image_path)
    paths = list(os.walk(image_path))
    for image in paths[0][2][0:]:
        folder.append(image)
    folder.sort()
    print(folder)
    img2 = cv2.imread(folder[0], 1)
    folder.pop(0)
    print('find', len(folder) + 1, 'images for steaching')
    for file in folder:
        img1 = cv2.imread(file, 1)
        orb = cv2.ORB_create()
        kp1, des1 = orb.detectAndCompute(img1, None)
        kp2, des2 = orb.detectAndCompute(img2, None)
        bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
        matches = bf.match(des1, des2)
        matches = sorted(matches, key=lambda x: x.distance)
        dst_pts = np.float32([kp1[m.queryIdx].pt for m in matches]).reshape(-1, 1, 2)
        src_pts = np.float32([kp2[m.trainIdx].pt for m in matches]).reshape(-1, 1, 2)
        M, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)
        h1, w1 = img1.shape[:2]
        h2, w2 = img2.shape[:2]
        pts1 = np.float32([[0, 0], [0, h1], [w1, h1], [w1, 0]]).reshape(-1, 1, 2)
        pts2 = np.float32([[0, 0], [0, h2], [w2, h2], [w2, 0]]).reshape(-1, 1, 2)
        pts2 = cv2.perspectiveTransform(pts2, M)
        pts = np.concatenate((pts1, pts2), axis=0)
        [xmin, ymin] = np.int32(pts.min(axis=0).ravel() - 0.5)
        [xmax, ymax] = np.int32(pts.max(axis=0).ravel() + 0.5)
        t = [-xmin, -ymin]
        Ht = np.array([[1, 0, t[0]], [0, 1, t[1]], [0, 0, 1]])
        result = cv2.warpPerspective(img2, Ht.dot(M), (xmax - xmin, ymax - ymin))
        result[t[1]:h1 + t[1], t[0]:w1 + t[0]] = img1
        img2 = result

    #if os.path.exists(root_dir):
    #    shutil.rmtree(root_dir)
    #os.makedirs(root_dir)
    os.chdir(root_dir)
    result = crop(result)
    result = resize(result, 256)
    print('map created', result.shape[:2])
    return result


def create_panorama1():
    folder = []
    os.chdir(image_path)
    paths = list(os.walk(image_path))
    for image in paths[0][2][0:]:
        folder.append(image)
    folder.sort()
    print(folder)
    img2 = cv2.imread(folder[0], 1)
    folder.pop(0)
    print('find', len(folder) + 1, 'images for steaching')
    for file in folder:
        img1 = cv2.imread(file, 1)
        orb = cv2.ORB_create()
        kp1, des1 = orb.detectAndCompute(img1, None)
        kp2, des2 = orb.detectAndCompute(img2, None)
        bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
        matches = bf.match(des1, des2)
        matches = sorted(matches, key=lambda x: x.distance)
        dst_pts = np.float32([kp1[m.queryIdx].pt for m in matches]).reshape(-1, 1, 2)
        src_pts = np.float32([kp2[m.trainIdx].pt for m in matches]).reshape(-1, 1, 2)
        M, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)
        h1, w1 = img1.shape[:2]
        h2, w2 = img2.shape[:2]
        pts1 = np.float32([[0, 0], [0, h1], [w1, h1], [w1, 0]]).reshape(-1, 1, 2)
        pts2 = np.float32([[0, 0], [0, h2], [w2, h2], [w2, 0]]).reshape(-1, 1, 2)
        pts2 = cv2.perspectiveTransform(pts2, M)
        pts = np.concatenate((pts1, pts2), axis=0)
        [xmin, ymin] = np.int32(pts.min(axis=0).ravel() - 0.5)
        [xmax, ymax] = np.int32(pts.max(axis=0).ravel() + 0.5)
        t = [-xmin, -ymin]
        Ht = np.array([[1, 0, t[0]], [0, 1, t[1]], [0, 0, 1]])
        result = cv2.warpPerspective(img2, Ht.dot(M), (xmax - xmin, ymax - ymin))
        result[t[1]:h1 + t[1], t[0]:w1 + t[0]] = img1
        img2 = result

    #if os.path.exists(root_dir):
    #    shutil.rmtree(root_dir)
    #os.makedirs(root_dir)
    os.chdir(root_dir)
    result = crop(result)
    result = resize(result, 256)
    print('map created', result.shape[:2])
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
    :return: distance between tho img
    """
    meta_data_1 = ImageMetaData(image1)
    meta_data_2 = ImageMetaData(image2)
    latlng_1 = meta_data_1.get_lat_lng()
    latlng_2 = meta_data_2.get_lat_lng()
    print(latlng_1)
    print(latlng_2)
    if latlng_1 and latlng_2 is not None:
        dist = math.acos(math.sin(math.radians(latlng_1[0])) * math.sin(math.radians(latlng_2[0])) +
                         math.cos(math.radians(latlng_1[0])) * math.cos(math.radians(latlng_2[0])) *
                         math.cos(math.radians(latlng_1[1] - latlng_2[1]))) * 6371 * 1000
        return dist

