from osgeo import gdal


image_dir = './result'

def geo_point(x, y):
    ds = gdal.Open('result1.tif')
    xoffset, px_w, rot1, yoffset, px_h, rot2 = ds.GetGeoTransform()

    posX = px_w * x + rot1 * y + xoffset
    posY = rot2 * x + px_h * y + yoffset

    print('for x {} y {} coord is:\n{} {}'.format(x, y, posX, posY))
    return posX, posY

#geo_point(5376 / 2, 5888 / 2)
#geo_point(0, 0)
