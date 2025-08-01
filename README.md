# rattree

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
