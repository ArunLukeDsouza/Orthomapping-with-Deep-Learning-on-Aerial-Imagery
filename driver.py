# Exif Driver Import
import os
from PIL import Image, ExifTags
from pymap3d import ecef2enu, geodetic2ecef
import numpy as np

from PIL.ExifTags import GPSTAGS
from PIL import Image
from PIL.ExifTags import TAGS

# Orthomapping Driver Import
import utilities as util
import Combiner
import cv2

# import os
# cwd = os.getcwd()  # Get the current working directory (cwd)
# files = os.listdir(cwd)  # Get all the files in that directory
# print("Files in %r: %s" % (cwd, files))

# Exif Driver
def get_exif(filename):
    image = Image.open(filename)
    image.verify()
    return image._getexif()

def get_geotagging(exif):
    if not exif:
        raise ValueError("No EXIF metadata found")

    geotagging = {}
    for (idx, tag) in TAGS.items():
        if tag == 'GPSInfo':
            if idx not in exif:
                raise ValueError("No EXIF geotagging found")

            for (key, val) in GPSTAGS.items():
                if key in exif[idx]:
                    geotagging[val] = exif[idx][key]

    return geotagging

def get_decimal_from_dms(dms, ref):

    degrees = dms[0]
    minutes = dms[1] / 60.0
    seconds = dms[2] / 3600.0

    if ref in ['S', 'W']:
        degrees = -degrees
        minutes = -minutes
        seconds = -seconds

    return round(degrees + minutes + seconds, 5)

def get_LLA(geotags):
    lat = get_decimal_from_dms(geotags['GPSLatitude'], geotags['GPSLatitudeRef'])

    lon = get_decimal_from_dms(geotags['GPSLongitude'], geotags['GPSLongitudeRef'])

    alt = geotags['GPSAltitude'] # altitude in meters

    return lat, lon, alt


def get_data_1(path_name1):
    exif = get_exif(path_name1)
    geotags = get_geotagging(exif)

    return get_LLA(geotags)

def get_data(path):
    lat0 = None
    lon0 = None
    h0 = 0
    filepath = []
    for root, dirs, files in os.walk(path):
        for filename in sorted(filter(lambda x: os.path.splitext(x)[1].lower() == '.jpg', files)):
            filepath.append(os.path.join(root, filename))
            for im in filepath:
                pitch = 0
                yaw = 0
                roll = 0
                lat, lon, alt = get_data_1(im)
                if lat0 is None:
                    lat0 = lat
                    lon0 = lon
                x, y, z = geodetic2ecef(lat, lon, alt)
                x, y, z = ecef2enu(x, y, z, lat0, lon0, h0)
            yield filename, '{:f}'.format(x), '{:f}'.format(y), '{:f}'.format(z), yaw, pitch, roll

def main():
    # image_data = []
    data = [d for d in get_data('datasets/images')]
    data = sorted(data, key=lambda x: x[0])
    x = np.array(map(lambda d: d[1], data))
    y = np.array(map(lambda d: d[2], data))

    with open('datasets/imageData.txt', 'w+') as f:
        for datum in data:
            f.write(','.join([str(d) for d in datum]) + '\n')

    # for datum in data:
    #     image_data.append(','.join([str(d) for d in datum]) + '\n')
    # print(image_data)
    # return image_data

main()

# Orthomapping Driver
path = os.getcwd()
print(path)

fileName = os.path.join(path, "datasets/imageData.txt")
imageDirectory = os.path.join(path, "datasets/images", "")

allImages, dataMatrix = util.importData(fileName, imageDirectory)
myCombiner = Combiner.Combiner(allImages, dataMatrix)
result = myCombiner.createMosaic()
util.display("RESULT", result)
cv2.imwrite("results/finalResult.png", result)
