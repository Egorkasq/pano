from osgeo import gdal
from geojson import Polygon


def pixelOffset2coord(rasterfn,xOffset,yOffset):
    raster = gdal.Open(rasterfn)
    geotransform = raster.GetGeoTransform()
    originX = geotransform[0]
    originY = geotransform[3]
    pixelWidth = geotransform[1]
    pixelHeight = geotransform[5]
    coordX = originX+pixelWidth*xOffset
    coordY = originY+pixelHeight*yOffset
    return coordX, coordY


def coord2pixelOffset(rasterfn, x, y):
    raster = gdal.Open(rasterfn)
    geotransform = raster.GetGeoTransform()
    originX = geotransform[0]
    originY = geotransform[3]
    pixelWidth = geotransform[1]
    pixelHeight = geotransform[5]
    xOffset = int((x - originX)/pixelWidth)
    yOffset = int((y - originY)/pixelHeight)
    return xOffset, yOffset


if __name__ == '__main__':
    print(coord2pixelOffset('odm_orthophoto.original.tif', 758515.608914, 4298315.87526))
    print(pixelOffset2coord('odm_orthophoto.original.tif', 0, 0))

    with open('track', 'r') as f:
        with open('track_with_geo.txt', 'w') as e:
            for i in f:
                for k in range(len(i)):
                    if i[k] == ' ':
                        x = int(i[:k])
                        y = int(i[k:])
                        break
                e.write(str(pixelOffset2coord('odm_orthophoto.original.tif', int(x), int(y))) + '\n')
    print(Polygon([(2.38, 57.322), (23.194, -20.28), (-120.43, 19.15), (2.38, 57.322)]))