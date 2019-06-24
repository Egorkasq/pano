import cv2
import numpy as np
import os
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

    folder = []
    os.chdir(image_path)
    #color_align(image_path1)
    paths = list(os.walk(image_path))
    for image in paths[0][2][0:]:
        folder.append(image)
    folder.sort()
    print('find {} images for stitching:{}'.format(len(folder), folder))
    img2 = cv2.imread(folder[0], 1)
    folder.pop(0)
    for file in folder:
        base_img = cv2.GaussianBlur(cv2.cvtColor(base_img_rgb, cv2.COLOR_BGR2GRAY), (5, 5), 0)

        img1 = cv2.imread(file, 1)
        orb = cv2.ORB_create()
        kp1, des1 = orb.detectAndCompute(img1, None)
        kp2, des2 = orb.detectAndCompute(img2, None)
        bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
        matches = bf.match(des1, des2)
        matches = sorted(matches, key=lambda x: x.distance)
        matches = matches[:int(len(matches) * 0.4)]
        assert len(matches) > 10

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
        print(t)
        Ht = np.array([[1, 0, t[0]], [0, 1, t[1]], [0, 0, 1]])
        result = cv2.warpPerspective(img2, Ht.dot(M), (xmax - xmin, ymax - ymin))
        if t[0] > 0:
            img1 = img1[:int(w1 - 0.5 * w1 + t[0] / 2), :result.shape[1]]
        elif t[0] > 0:
            img1 = img1[int(w1 - 0.5 * w1 + t[0] / 2):, :result.shape[1]]

        h1, w1 = img1.shape[:2]
        result[t[1]:h1 + t[1], t[0]:w1 + t[0]] = img1
        crop(result)
        img2 = result

    os.chdir(root_dir)
    cv2.imwrite('res.jpg', result)
    return 0


if __name__ == '__main__':
    create_panorama(image_path)
