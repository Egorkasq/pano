import cv2
import numpy as np
import os
import math

root_dir = os.path.abspath('./result')
image_path = os.path.abspath('./input_img')

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

def create_panorama(image_path):
    folder = []
    good_point = []
    os.chdir(image_path)
    paths = list(os.walk(image_path))
    for image in paths[0][2][0:]:
        folder.append(image)
    folder.sort()
    print(folder)
    image1 = cv2.imread(folder[0], 1)
    #image1 = cv2.GaussianBlur(cv2.cvtColor(image1, cv2.COLOR_BGR2GRAY), (5, 5), 0)
    folder.pop(0)
    print('find {} images for steaching'.format(len(folder) + 1))
    for file in folder:
        image2 = cv2.imread(file, 1)

        #image2 = cv2.GaussianBlur(cv2.cvtColor(image2, cv2.COLOR_BGR2GRAY), (5, 5), 0)
        orb = cv2.ORB_create()
        kp1, des1 = orb.detectAndCompute(image1, None)
        kp2, des2 = orb.detectAndCompute(image2, None)
        bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
        matches = bf.match(des1, des2)
        matches = sorted(matches, key=lambda x: x.distance)

        #matches = matches[:11]
        matches = matches[:int(len(matches) * 0.4)]
        assert len(matches) > 10
        dst_pts = np.float32([kp1[m.queryIdx].pt for m in matches]).reshape(-1, 1, 2)
        src_pts = np.float32([kp2[m.trainIdx].pt for m in matches]).reshape(-1, 1, 2)
        M, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)
        h1, w1 = image1.shape[:2]
        h2, w2 = image2.shape[:2]
        pts1 = np.float32([[0, 0], [0, h1], [w1, h1], [w1, 0]]).reshape(-1, 1, 2)
        pts2 = np.float32([[0, 0], [0, h2], [w2, h2], [w2, 0]]).reshape(-1, 1, 2)
        pts2 = cv2.perspectiveTransform(pts2, M)
        pts = np.concatenate((pts1, pts2), axis=0)
        [xmin, ymin] = np.int32(pts.min(axis=0).ravel() - 0.5)
        [xmax, ymax] = np.int32(pts.max(axis=0).ravel() + 0.5)
        t = [-xmin, -ymin]
        Ht = np.array([[1, 0, t[0]], [0, 1, t[1]], [0, 0, 1]])
        result = cv2.warpPerspective(image2, Ht.dot(M), (xmax - xmin, ymax - ymin))
        result[t[1]:h1 + t[1], t[0]:w1 + t[0]] = image1
        crop(result)
        #print(math.acos(M[0][1]))
        #if t[0] > 0:
        #    result = result[t[1]:result.shape[0], t[0]:int(result.shape[1])]
        #else:
        #    result = result[:int(result.shape[0] - t[0]), :int(result.shape[1] - t[1])]

        image1 = result


    os.chdir(root_dir)
    cv2.imwrite('card.jpg', result)


panoram = create_panorama(image_path)