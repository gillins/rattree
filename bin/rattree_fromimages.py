#!/usr/bin/env python

# This file is part of RATTree
# Copyright (C) 2020  Sam Gillingham
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with this program.  If not, see <http://www.gnu.org/licenses/>.

from __future__ import print_function, division

import io
import sys
import csv
import argparse
import numpy
from osgeo import gdal
from rios import applier
from rios import rat
from rios import calcstats
from rios.imageio import GDALTypeToNumpyType
from rios.fileinfo import ImageInfo
from rios.cuiprogress import GDALProgressBar

from rattree import RATTree

DFLT_OUTPUT_DRIVER = 'KEA'

def get_cmdargs():
    """     
    Get the command line arguments.
    """
    descstr = ("Given a CSV file containing a list of thematic image files " +
            "and column names, create a new image with a RAT and each pixel" +
            "as a row into the RAT that corresponds to the pixel values in" +
            "the input images")
    parser = argparse.ArgumentParser(description=descstr,
            formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("-i", "--inputs", help="path to input csv")
    parser.add_argument("-o", "--output", help="name of output KEA file")
    parser.add_argument("-f", "--format", default=DFLT_OUTPUT_DRIVER, 
            help="name of output GDAL format")
    parser.add_argument("-g", "--gdalstats", default=False,
            action="store_true",
            help="Use GDAL to calculate Histogram and statistics. " +
                "Default is to use histogram from the RAT and estimate " +
                "statistics from that.")
    parser.add_argument("-n", "--nocolourtable", default=False,
            action="store_true",
            help="Don't write a random colour table to the file")
    
    args = parser.parse_args()
    if args.inputs is None or args.output is None:
        parser.print_help()
        sys.exit()
        
    return args
    
def estimateStatsFromHisto(ds, hist):
    """
    As a shortcut to calculating stats with GDAL, use the histogram 
    that we already have from calculating the RAT and calc the stats
    from that. 
    """
    # https://stackoverflow.com/questions/47269390/numpy-how-to-find-first-non-zero-value-in-every-column-of-a-numpy-array
    mask = hist > 0
    nVals = hist.sum()
    minVal = mask.argmax()
    maxVal = hist.shape[0] - numpy.flip(mask).argmax() - 1
    
    values = numpy.arange(hist.shape[0])
    
    meanVal = (values * hist).sum() / nVals
    
    stdDevVal = (hist * numpy.power(values - meanVal, 2)).sum() / nVals
    stdDevVal = numpy.sqrt(stdDevVal)
    
    modeVal = numpy.argmax(hist)
    # estimate the median - bin with the middle number
    middlenum = hist.sum() / 2
    gtmiddle = hist.cumsum() >= middlenum
    medianVal = gtmiddle.nonzero()[0][0]
    
    band = ds.GetRasterBand(1)
    band.SetMetadataItem("STATISTICS_MINIMUM", repr(minVal))
    band.SetMetadataItem("STATISTICS_MAXIMUM", repr(maxVal))
    band.SetMetadataItem("STATISTICS_MEAN", repr(meanVal))
    band.SetMetadataItem("STATISTICS_STDDEV", repr(stdDevVal))
    band.SetMetadataItem("STATISTICS_MODE", repr(modeVal))
    band.SetMetadataItem("STATISTICS_MEDIAN", repr(medianVal))
    band.SetMetadataItem("STATISTICS_SKIPFACTORX", "1")
    band.SetMetadataItem("STATISTICS_SKIPFACTORY", "1")
    band.SetMetadataItem("STATISTICS_HISTOBINFUNCTION", "direct")
    band.SetMetadataItem("STATISTICS_HISTOMIN", "0")
    band.SetMetadataItem("STATISTICS_HISTOMAX", repr(maxVal))
    band.SetMetadataItem("STATISTICS_HISTONUMBINS", repr(len(hist)))
    
def readCSV(inputs):
    """
    Read the .csv file and return 2 list. One for input filenames
    and one for output column names
    """
    filenames = []
    columnNames = []

    # use io.open as this has the newline param on python 2 and 3
    with io.open(inputs, newline='') as f:
        reader = csv.reader(f)
        for row in reader:
            filenames.append(row[0])
            columnNames.append(row[1])
            
    return filenames, columnNames
    
def buildImageAndTree(info, inputs, outputs, otherArgs):
    """
    Called from RIOS. Does the actual work.
    """
    # first stack the inputs into one array
    block = numpy.vstack(inputs.inputs)
    
    outputs.output = otherArgs.tree.addfromRIOS(block, otherArgs.nodatas)
    
    
def main(cmdargs):
    """
    Main function
    """
    filenames, columnNames = readCSV(cmdargs.inputs)
    
    nodatas = []
    for fname in filenames:
        info = ImageInfo(fname)
        if len(info.nodataval) > 1:
            raise SystemExit("Inputs need to be single band")
            
        val = info.nodataval[0]
        if val is None:
            raise SystemExit("No data values must be set on inputs")
            
        # now convert to the type of the image so we 
        # are doing comparisons on the same type
        numpyType = GDALTypeToNumpyType(info.dataType)
        val = numpyType(val)
            
        nodatas.append(val)
    
    otherargs = applier.OtherInputs()
    otherargs.tree = RATTree()
    # this will convert to the 'smallest' type that holds all the data types
    # - hopefully they all the same
    otherargs.nodatas = numpy.array(nodatas)
    
    inputs = applier.FilenameAssociations()
    inputs.inputs = filenames
    
    outputs = applier.FilenameAssociations()
    outputs.output = cmdargs.output
    
    controls = applier.ApplierControls()
    progress = GDALProgressBar()
    controls.setProgress(progress)
    controls.setThematic(True)
    controls.setOutputDriverName(cmdargs.format)
    # we calculate the stats from the Histogram
    # (which we create)
    # pyramid layers are calculated separately
    controls.setCalcStats(cmdargs.gdalstats)
    
    applier.apply(buildImageAndTree, inputs, outputs, otherargs, 
                        controls=controls)
                        
    print()
    print('rows', otherargs.tree.currow - 1)
    outputRAT = otherargs.tree.dumprat()
                        
    print()
    print('writing rat')
    ds = gdal.Open(cmdargs.output, gdal.GA_Update)

    if not cmdargs.gdalstats:
        # now histogram
        hist = otherargs.tree.dumphist()
        rat.writeColumn(ds, 'Histogram', hist, colUsage=gdal.GFU_PixelCount,
                                colType=gdal.GFT_Integer)

        estimateStatsFromHisto(ds, hist)
        
        calcstats.addPyramid(ds, progress)
        
    # else stats/pyramids are handled by RIOS calcstats
    
    # colour table if requested
    if not cmdargs.nocolourtable:
        # NOTE: rios.rat colour table stuff uses the old RAT API
        # so let's use the faster one
        colNames = ["Blue", "Green", "Red"]
        colUsages = [gdal.GFU_Blue, gdal.GFU_Green, gdal.GFU_Red]
        numEntries = outputRAT.shape[0]
        for band in range(3):
            data = numpy.random.randint(0, 255, size=numEntries)
            data[0] = 0
            rat.writeColumn(ds, colNames[band], data, colUsage=colUsages[band],
                    colType=gdal.GFT_Integer)
            
        # alpha 
        alpha = numpy.full((numEntries,), 255, dtype=numpy.uint8)
        alpha[0] = 0
        rat.writeColumn(ds, 'Alpha', alpha, colUsage=gdal.GFU_Alpha,
                    colType=gdal.GFT_Integer)

    # now all the rat columns
    for idx, colName in enumerate(columnNames):
        rat.writeColumn(ds, colName, outputRAT[..., idx], 
                            colType=gdal.GFT_Integer)
                            
    ds.FlushCache()

if __name__ == '__main__':

    cmdargs = get_cmdargs()
    main(cmdargs)
