#!/usr/bin/env python

"""
Code for quickly creating a Raster Attribute Table (RAT). The original idea 
is that you have a bunch of thematic images that need to be converted to a 
new thematic image with a RAT and the RAT contains one column per input image. 

This attempts to speed up the process of deciding whether to create a new RAT
row when you get a new set of pixels. To do this it uses a tree. There is a 
level of the tree for each input file. Each node of the tree has an image value, 
a pointer to any sibling nodes, and an optional row value for leaf nodes. 

When you add a new row, the code moves through each node following siblings
who have matching image values (progressing through each value in the new row). 
This is best demonstrated using a sketch:

1. First an initial row is added and a simple tree with no siblings is created
    for the values in the row. Say [5, 3, 9] is added:
    
head --> imgval=5 ---> imgval=3 ---> imgval=9, row=1

2. Another row is added ([5, 4, 8]). A sibling for 3 at the second level
    is added. Because this creates a new leaf node, a new row is created.
    
head --> imgval=5 ---> imgval=3 ---> imgval=9, row=1
                           |
                       imgval=4 ---> imgval=8, row=2
         
3. If data that already in the tree is provided the tree is traversed, but 
    because a leaf row is found for the last value the existing row value 
    is returned.
    
4. Another row is added ([5, 4, 7]) and this requires just a new leaf node:

head --> imgval=5 ---> imgval=3 ---> imgval=9, row=1
                           |
                       imgval=4 ---> imgval=8, row=2
                                        |
                                     imgval=7, row=3
    

5. Another row ([9, 0, 1]) is added that requires a new branch at the start 
    of the tree:

head --> imgval=5 ---> imgval=3 ---> imgval=9, row=1
            |              |
            |          imgval=4 ---> imgval=8, row=2
            |                           |
            |                        imgval=7, row=3
            |
         imgval=9 ---> imgval=0 ---> imgval=1, row=4
 
For speed, the RAT is filled in for the new row when a new leaf is created. 
This ended up being much faster (10x) than traversing the tree later to build the 
RAT. The downside is that we don't know the size of the RAT so the RAT
is periodically grown by the size given by RAT_GROW_SIZE, and truncated
by RATTree.dumprat()

row 1 = [5, 3, 9]
row 2 = [5, 4, 8]
row 3 = [5, 4, 7]
row 4 = [9, 0, 1]
                            

"""
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

from collections import OrderedDict
import numpy
import numba
from numba import njit
try:
    # support older versions of numba
    from numba.experimental import jitclass
except ImportError:
    from numba import jitclass
    
"Initial size of the RAT and size to grow it as"
RAT_GROW_SIZE = 2**19 # ~500k

# Tell Numba we will define this type later
# this is to work around circular reference - LinkedNode
# points to itself
node_type = numba.deferred_type()

# Specify the types of the attributes in LinkedNode
node_spec = OrderedDict()
node_spec['row'] = numba.uint32 # 0 unless a leaf node
node_spec['imgval'] = numba.uint32 # value taken from input image that will be written to RAT
node_spec['next'] = numba.optional(node_type)   # sibling node. None if last sibling
node_spec['child'] = numba.optional(node_type)  # pointer to next level of the tree for this combo of values

@jitclass(node_spec)
class LinkedNode(object):
    """
    Main object in this module. Is an element in the tree.
    See node_spec for description of attributes. 
    """
    def __init__(self, imgval, next):
        self.row = 0
        self.imgval = imgval
        self.next = next
        self.child = None

    def insert(self, imgval):
        # inserts a new node (with given imgval) as a sibling
        # straight after this node in the list.
        newnode = LinkedNode(imgval, self.next)
        self.next = newnode
        return newnode
        
    def setChild(self, child):
        # given a new LinkedNode insert this as a child node
        # to this node. 
        self.child = child
        
    def setRow(self, row):
        # is a leaf node (row should be != 0)
        self.row = row

    def adddata(self, data, currow, rat):
        # Given an array of image values (data) and the current row number
        # we are up to add this to the tree 
        # returns tuple with new current row and row number for the data
        # these may be different when the data is already in the tree.
        # also passed the RAT which is filled in. Up to caller to ensure RAT
        # is large enough to handle the data.
        idx = 0
        return adddata_tonode(self, data, idx, currow, rat)

# now we have the full definition of the class we can set as the type.
node_type.define(LinkedNode.class_type.instance_type)
        
@njit
def adddata_tonode(node, data, idx, currow, rat):
    """
    Numba does support static class members, but for some reason these don't
    work with recursion. So we have this as a module level function that
    does seem fine with recursion. 
    
    This function takes an existing node in the tree, and tries to add the
    elements in data to it. For each level in the tree it tries to find a sibling
    that matches the first element in data. If it finds one and the node is a 
    leaf node then it returns the row for that node. Otherwise it attempts to
    add the next element in data as a child to the existing node using recursion.
    
    idx is the current into data that we are processing.
    
    rat is the RAT to update as we go. Caller must ensure this is large enough.
    
    If the first element in data was not found, then it is added as a sibling to
    the current node. The remaining elements in data are then added as children
    and currow is then set as the row of the leaf element. Currow is incremented
    before being returned.
    
    This function returns and (possibly) updated value of currow, and the row
    of the data (which maybe new, or the value of a row already added to the
    tree if data as been added before).
    """
    sibling = node
    while sibling is not None:
        # process this and all siblings
        if sibling.imgval == data[idx]:
            # matching value found
            if sibling.row != 0:
                # is leaf node, return this row
                return currow, sibling.row

            # process child with remaining data elements
            currow, row = adddata_tonode(sibling.child, data, idx+1, currow, rat)
            return currow, row

        # go to next sibling
        sibling = sibling.next

    # got here, not one of the siblings, add it
    newsibling = node.insert(data[idx])
    child = newsibling
    # now go through and add the remaining data elements all as children
    # to this node
    for n in range(idx+1, data.shape[0]):
        newchild = LinkedNode(data[n], None)
        child.setChild(newchild)
        child = newchild
    # set row for leaf node
    child.setRow(currow)
    # copy into rat
    for n in range(data.shape[0]):
        rat[currow, n] = data[n]
    
    row = currow
    # update currow
    currow += 1

    return currow, row

# specify the types of the attributes of RATTree
tree_spec = OrderedDict()
tree_spec['currow'] = numba.uint32  # value of the row that will be added next
tree_spec['head'] = numba.optional(node_type) # first node in the tree
tree_spec['ncols'] = numba.uint32   # number of columns in the RAT (same as number of input images)
tree_spec['rat'] = numba.optional(numba.uint32[:,:])  # RAT as it is being built (rows, cols)
tree_spec['histo'] = numba.optional(numba.uint32[:])  # Histogram as it is bring built (rows)

@jitclass(tree_spec)
class RATTree(object):
    """
    Class that represents a tree. Holds pointer to first node in the tree.
    Also keeps track of the number of the next row to be added.
    """
    def __init__(self):
        self.currow = 1
        self.head = None
        self.ncols = 0
        self.rat = None
        self.histo = None
        
    def adddata(self, data):
        """
        Add an array of image values to the tree. Return the row
        these values will have in the final RAT.
        """
        if self.head is None:
            # new tree
            self.ncols = data.shape[0]
            # add first element as head. 
            self.head = LinkedNode(data[0], None)
            child = self.head
            # remaining elements as children
            for idx in range(1, data.shape[0]):
                newchild = LinkedNode(data[idx], None)
                child.setChild(newchild)
                child = newchild

            row = self.currow
            child.setRow(self.currow)
            self.currow += 1
            
            # create RAT
            # note i64 cast for older numba...
            self.rat = numpy.empty((numpy.int64(RAT_GROW_SIZE), 
                        numpy.int64(self.ncols)), dtype=numpy.uint32)
                        
            # create histogram
            self.histo = numpy.zeros((numpy.int64(RAT_GROW_SIZE),), 
                                dtype=numpy.uint32)
            
            # first and second row
            for idx2 in range(self.ncols):
                self.rat[0, idx2] = 0 # first row all zeros
                self.rat[row, idx2] = data[idx2] # first row of actual rat (row=1)
            
        else:
            if data.shape[0] != self.ncols:
                raise ValueError('inconsistent number of columns')
        
            # otherwise try and add this to the exiting tree
            self.currow, row = self.head.adddata(data, self.currow, self.rat)
            
            if self.currow >= self.rat.shape[0]:
                #print('growing RAT')
                newRAT = numpy.empty((numpy.int64(self.rat.shape[0] + RAT_GROW_SIZE), 
                        numpy.int64(self.ncols)), dtype=numpy.uint32)
                for n in range(self.currow):
                    for j in range(self.ncols):
                        newRAT[n, j] = self.rat[n, j]
                self.rat = newRAT
                
                # new histo
                newHisto = numpy.zeros((numpy.int64(self.histo.shape[0] + RAT_GROW_SIZE),),
                                        dtype=numpy.uint32)
                for n in range(self.currow):
                    newHisto[n] = self.histo[n]
                self.histo = newHisto
                
        # update histo
        self.histo[row] += 1
        
        return row
        
    def addfromRIOS(self, block, nodatas):
        """
        When passed a (nlayers, ysize, xsize) shape block from RIOS, add 
        all the values to the tree.
        
        nodatas should be a 1d array of the nodata values - same length
        as the number. If the data for any of the layers matches the 
        corresponding nodata the pixel is ignored.
        """
        nlayers, ysize, xsize = block.shape
        # you might think it would be faster to transpose() the data
        # so we can tightloop the nlayers, but it ends up being slower
        # overall...
        
        data = numpy.empty((nlayers,), dtype=numpy.uint32)
        output = numpy.empty((1, ysize, xsize), dtype=numpy.uint32)
        
        allok = True
        for y in range(ysize):
            for x in range(xsize):
                for n in range(nlayers):
                    val = block[n, y, x]
                    if val == nodatas[n]:
                        allok = False
                        break
                    data[n] = val
                    
                if allok:
                    output[0, y, x] = self.adddata(data)
                else:
                    output[0, y, x] = 0
                    # don't need to update histogram in this case as 
                    # we are ignoring zeros
                
        return output
        
    def dump(self):
        """
        For debugging. Dump the tree in a readable format.
        """
        # note i64 cast for older numba...
        data = numpy.empty((numpy.int64(self.ncols),), dtype=numpy.uint32)
        idx = 0
        dumprow(self.head, data, idx)
        
    def dumprat(self):
        """
        Return the RAT built from the tree. Shape is (rows, cols).
        Truncates to the used space of the RAT.
        """
        return self.rat[:self.currow]
        
    def dumphist(self):
        """
        Return the Histogram built from the tree.
        Truncates to the used space of the histogram
        """
        return self.histo[:self.currow]

@njit
def dumprow(node, data, idx):
    """
    See comments above about recursion and static class functions.
    node is the current node to process, data is an array that is populated
    with values for the current row and idx is the index into data for
    the current level of the tree being processed
    """
    if node.row != 0:
        # leaf nodes. Print out all siblings with the contents of data
        next = node
        while next is not None:
            data[idx] = next.imgval
            print('row', next.row, 'data', data)
            next = next.next
        
    else:
        # any siblings
        next = node.next
        while next is not None:
            # process this sibling
            dumprow(next, data, idx)
            next = next.next
            
        # process child of this node at next location in data
        data[idx] = node.imgval
        idx += 1
        dumprow(node.child, data, idx)

@njit
def main():
    """
    Basic test code
    """
    tree = RATTree()
    row = tree.adddata(numpy.array([5, 3, 9]))
    print(row)

    row = tree.adddata(numpy.array([5, 4, 8]))
    print(row)

    row = tree.adddata(numpy.array([5, 4, 8]))
    print(row)

    row = tree.adddata(numpy.array([5, 4, 7]))
    print(row)

    row = tree.adddata(numpy.array([9, 0, 1]))
    print(row)
    
    tree.dump()
    print(tree.dumprat())
    
@njit
def mainStressTest():
    """
    Try adding 1million rows and see how fast this is.
    """
    tree = RATTree()
    for n in range(1000000):
        data = numpy.random.randint(0, 5, 10)
        row = tree.adddata(data)
    
    print(tree.currow)
    rat = tree.dumprat()
    
def mainRIOSTest():
    """
    Simple test for RIOS interface
    """
    tree = RATTree()
    block = numpy.random.randint(0, 10, (10, 256, 256), 
                dtype=numpy.uint32)
    
    result = tree.addfromRIOS(block)
    print(result)
    
    
if __name__ == '__main__':
    import time
    main()
    
    a = time.time()
    mainRIOSTest()
    b = time.time()
    print(b - a)
    
    #a = time.time()
    #mainStressTest()
    #b = time.time()
    #print(b - a)
    
