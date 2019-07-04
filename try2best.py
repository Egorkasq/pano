import cv2
import numpy as np
import os
import gdal
import math
#import sift.client

root_dir = os.path.abspath('./result')
image_path = os.path.abspath('./input_img')
image_path1 = os.path.abspath('./input_img')


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


def color_align(image_path):
    paths = list(os.walk(image_path))
    n = 0
    for image in paths[0][2][0:]:
        image = cv2.imread(image, 1)
        hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        h, s, v = cv2.split(hsv)
        hsv[:, :, 2] += v

        img = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)
        cv2.imwrite('{}/{}.jpg'.format(image_path1, n), img)
        n += 1
    return 0


def create_panorama(image_path):

    os.chdir(image_path)
    paths = list(os.walk(image_path))
    folder = paths[0][2][0:]
    folder.sort()
    print('find {} images for stitching:{}'.format(len(folder), folder))
    base_image = cv2.imread(folder[0], 1)
    folder.pop(0)

    for file in folder:

        next_image = cv2.imread(file, 1)
        orb = cv2.ORB_create()
        kp1, des1 = orb.detectAndCompute(next_image, None)
        kp2, des2 = orb.detectAndCompute(base_image, None)
        bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
        matches = bf.match(des1, des2)
        matches = sorted(matches, key=lambda x: x.distance)
        matches = matches[:int(len(matches) * 0.2)]
        print(len(matches))

        dst_pts = np.float32([kp1[m.queryIdx].pt for m in matches]).reshape(-1, 1, 2)
        src_pts = np.float32([kp2[m.trainIdx].pt for m in matches]).reshape(-1, 1, 2)
        M, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)
        h1, w1 = next_image.shape[:2]
        h2, w2 = base_image.shape[:2]
        pts1 = np.float32([[0, 0], [0, h1], [w1, h1], [w1, 0]]).reshape(-1, 1, 2)
        pts2 = np.float32([[0, 0], [0, h2], [w2, h2], [w2, 0]]).reshape(-1, 1, 2)
        pts2 = cv2.perspectiveTransform(pts2, M)
        print('yes', M[0][2], M[1][2])
        #print(math.degrees(M[0][1]))
        pts = np.concatenate((pts1, pts2), axis=0)
        [xmin, ymin] = np.int32(pts.min(axis=0).ravel() - 0.5)
        [xmax, ymax] = np.int32(pts.max(axis=0).ravel() + 0.5)
        #print(xmin, xmax, ymin, ymax)
        #print(xmin, ymin, xmax, ymax)
        t = [-xmin, -ymin]

        Ht = np.array([[1, 0, t[0]], [0, 1, t[1]], [0, 0, 1]])

        if t[0] > 0:
            next_image = next_image[:int(h1 - 0.5 * h1 + t[0] / 2), :w1]
        elif t[0] < 0:
            next_image = next_image[int(w1 - 0.5 * w1 + t[0] / 2):, :w1]
        if t[1] > 0:
            next_image = next_image[:h1, int(w1 - 0.5 * w1 + t[0] / 2):w1]
        elif t[1] < 0:
            next_image = next_image[:h1, :int(w1 - 0.5 * w1 + t[0] / 2)]

        h1, w1 = next_image.shape[:2]
        result = cv2.warpPerspective(base_image, Ht.dot(M), (xmax - xmin, ymax - ymin))

        result[t[1]:h1 + t[1], t[0]:w1 + t[0]] = next_image
        crop(result)
        base_image = result

    os.chdir(root_dir)
    cv2.imwrite('res2.jpg', result)
    return 0




if __name__ == '__main__':
    map = create_panorama(image_path)
    os.system("gdal_translate -of GTiff -a_srs EPSG:4326", )