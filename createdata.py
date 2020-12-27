#!/usr/bin/env python

import numpy
from osgeo import gdal

IMG_XSIZE = 2000
IMG_YSIZE = 2000
IMG_NUM = 3
IMG_MIN = 1
IMG_MAX = 9

def main():

    driver = gdal.GetDriverByName('HFA')
    with open('imagelist.csv', 'w') as f:

        for n in range(IMG_NUM):
            name = 'image_{}.img'.format(n)
            print(name)
            ds = driver.Create(name, IMG_XSIZE, IMG_YSIZE, 1, gdal.GDT_Byte)
            data = numpy.random.randint(IMG_MIN, IMG_MAX, (IMG_YSIZE, IMG_XSIZE),
                        dtype=numpy.uint8)
            band = ds.GetRasterBand(1)
            band.WriteArray(data)
            band.SetNoDataValue(0)
            
            column = 'column {}'.format(n)
            f.write('{},{}\n'.format(name, column))

if __name__ == '__main__':
    main()
