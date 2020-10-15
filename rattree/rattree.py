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
         
The last step is to convert the tree back into a RAT. The tree is traversed and
when a leaf node is found, the various image values for that path through the
tree are copied into the row of the output row as specified by that leaf node:

row 1 = [5, 3, 9]
row 2 = [5, 4, 8]
row 3 = [5, 4, 7]
row 4 = [9, 0, 1]
                            

"""

import time
from collections import OrderedDict
import numpy
from numba import njit
from numba import uint32, deferred_type, optional, jitclass
try:
    # support older versions of numba
    from numba.experimental import jitclass
except ImportError:
    from numba import jitclass
from numba.typed import List

node_type = deferred_type()

spec = OrderedDict()
spec['row'] = uint32
spec['imgval'] = uint32
spec['next'] = optional(node_type)
spec['child'] = optional(node_type)


@jitclass(spec)
class LinkedNode(object):
    def __init__(self, imgval, next):
        self.row = 0
        self.imgval = imgval
        self.next = next
        self.child = None

    def insert(self, imgval):
        newnode = LinkedNode(imgval, self.next)
        self.next = newnode
        return newnode
        
    def setChild(self, child):
        self.child = child
        
    def setRow(self, row):
        self.row = row

    def adddata(self, data, currow):
        return adddata_tonode(self, data, currow)
        

@njit
def make_linked_node(imgval):
    return LinkedNode(imgval, None)

@njit
def adddata_tonode(node, data, currow):
    sibling = node
    while sibling is not None:
        if sibling.imgval == data[0]:
            if sibling.row != 0:
                return currow, sibling.row

            currow, row = adddata_tonode(sibling.child, data[1:], currow)
            return currow, row

        sibling = sibling.next

    # got here, not one of the siblings, add it
    newsibling = node.insert(data[0])
    child = newsibling
    for n in range(1, data.shape[0]):
        newchild = LinkedNode(data[n], None)
        child.setChild(newchild)
        child = newchild
    child.setRow(currow)
    row = currow
    currow += 1

    return currow, row

node_type.define(LinkedNode.class_type.instance_type)

treespec = OrderedDict()
treespec['currow'] = uint32
treespec['head'] = optional(node_type)
treespec['ncols'] = uint32

@jitclass(treespec)
class RATTree(object):
    def __init__(self):
        self.currow = 1
        self.head = None
        self.ncols = 0
        
    def adddata(self, data):
        if self.head is None:
            # new tree
            self.ncols = data.shape[0]
            self.head = make_linked_node(data[0])
            child = self.head
            for n in range(1, data.shape[0]):
                newchild = make_linked_node(data[n])
                child.setChild(newchild)
                child = newchild

            row = self.currow
            child.setRow(self.currow)
            self.currow += 1
            
        else:
            if data.shape[0] != self.ncols:
                raise ValueError('inconsistent number of columns')
        
            self.currow, row = self.head.adddata(data, self.currow)
        
        return row
        
    def dump(self):
        data = numpy.empty((self.ncols,), dtype=numpy.uint32)
        idx = 0
        dumprow(self.head, data, idx)
        
    def dumprat(self):
        # for speed
        rat = numpy.empty((self.currow, self.ncols), dtype=numpy.uint32)
        rat[0] = 0
        data = numpy.empty((self.ncols,), dtype=numpy.uint32)
        col = 0
        dumprowtorat(self.head, col, data, rat)
        return rat

@njit
def dumprow(node, data, idx):
    if node.row != 0:
        # leaf nodes
        next = node
        while next is not None:
            data[idx] = next.imgval
            print('row', next.row, 'data', data)
            next = next.next
        
    else:
        # any siblings
        next = node.next
        while next is not None:
            dumprow(next, data, idx)
            next = next.next
            
        # process child of this
        data[idx] = node.imgval
        idx += 1
        dumprow(node.child, data, idx)

@njit
def dumprowtorat(node, col, data, rat):
    if node.row != 0:
        # leaf nodes
        next = node
        while next is not None:
            data[col] = next.imgval
            # loop for speed
            for n in range(data.shape[0]):
                rat[next.row, n] = data[n]
            next = next.next
        
    else:
        # any siblings
        next = node.next
        while next is not None:
            dumprowtorat(next, col, data, rat)
            next = next.next
            
        # process child of this
        data[col] = node.imgval
        col += 1
        #print('imgval', head.imgval, 'has child', head.child is not None, 'has next', head.next is not None)
        dumprowtorat(node.child, col, data, rat)
        
@njit
def main():
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
    tree = RATTree()
    for n in range(1000000):
        data = numpy.random.randint(0, 5, 10)
        row = tree.adddata(data)
    
    print(tree.currow)
    rat = tree.dumprat()
    

if __name__ == '__main__':
    #main()
    a = time.time()
    mainStressTest()
    b = time.time()
    print(b - a)
    
