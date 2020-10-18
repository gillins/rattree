#!/usr/bin/env python

from __future__ import print_function, division

import sys
import csv
import argparse
import numpy
from osgeo import gdal
from rios import applier
from rios import rat
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
    
    args = parser.parse_args()
    if args.inputs is None or args.output is None:
        parser.print_help()
        sys.exit()
        
    return args
    
def readCSV(inputs):
    """
    Read the .csv file and return 2 list. One for input filenames
    and one for output column names
    """
    filenames = []
    columnNames = []

    with open(inputs, newline='') as f:
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
    
    outputs.output = otherArgs.tree.addfromRIOS(block)
    
    
def main(cmdargs):
    """
    Main function
    """
    filenames, columnNames = readCSV(cmdargs.inputs)
    
    otherargs = applier.OtherInputs()
    otherargs.tree = RATTree()
    
    inputs = applier.FilenameAssociations()
    inputs.inputs = filenames
    
    outputs = applier.FilenameAssociations()
    outputs.output = cmdargs.output
    
    controls = applier.ApplierControls()
    progress = GDALProgressBar()
    controls.setProgress(progress)
    controls.setThematic(True)
    controls.setOutputDriverName(cmdargs.format)
    
    applier.apply(buildImageAndTree, inputs, outputs, otherargs, 
                        controls=controls)
                        
    print()
    print('rows', otherargs.tree.currow)
    print('building rat')
    outputRAT = otherargs.tree.dumprat()
                        
    print()
    print('writing rat')
    ds = gdal.Open(cmdargs.output, gdal.GA_Update)
    for idx, colName in enumerate(columnNames):
        rat.writeColumn(ds, colName, outputRAT[..., idx], 
                            colType=gdal.GFT_Integer)
    
    ds.FlushCache()

if __name__ == '__main__':

    cmdargs = get_cmdargs()
    main(cmdargs)