#!/usr/bin/env python2.3
#
#  signature.py
#  
#
#  Created by Andrew BYRD on 17/08/09.
#  Copyright (c) 2009 __MyCompanyName__. All rights reserved.
#

from graphserver.ext.gtfs.gtfsdb import GTFSDatabase
from graphserver.graphdb import GraphDatabase
from graphserver.core import Graph, Street, State
from pylab import *
from numpy import *
from random import shuffle
import time 
from socket import *
import geotools
from math import ceil

#gtfsdb = GTFSDatabase  ( '../gsdata/bart.gtfsdb' )
#gdb    = GraphDatabase ( '../gsdata/bart.linked.gsdb' )
gtfsdb = GTFSDatabase  ( '../gsdata/trimet_13sep2009.gtfsdb' )
gdb    = GraphDatabase ( '../gsdata/trimet_13sep2009.linked.gsdb' )
g = gdb.incarnate()
grid = geotools.create_grid( g, gtfsdb)

def get_times(g, grid, origin, t0 = 1253820000) :
    print "Performing path search..."
    v_label = 'grid-%i-%i' % (origin)
    g.spt_in_place(v_label, None, State(1, t0))
    # this is the slow line !
    return array( [ [ (c[2].payload.time - t0) for c in r] for r in grid ] )

def cplot(Z) :
    imshow(Z, cmap=cm.gray)
    levels = array([15, 30, 60, 90])
    CS = contour(Z, levels, linewidths=2)
    clabel(CS, inline=1, fmt='%d', fontsize=14)
    show()

def transmit(result) :
    result = ''.join(result)
    udp.sendto ( result, ('localhost', 6000) )
    print "sent %d values" % len(result)

def visu_times(tt) :
    max_disp = 60 * 80  # transmit everything under max_disp seconds (normalized)
    visu_dim = (100, 100)
    # downsample the matrix for visualization
    row_scale = (tt.shape[0] / float(visu_dim[0]))
    col_scale = (tt.shape[1] / float(visu_dim[1]))
    tt = tt[::row_scale, ::col_scale]
    rows = tt.shape[0]
    cols = tt.shape[1]
    # cplot(tt)
    result = ['T'] 
    for row in range(rows) :
        for col in range(cols) :
            if len(result) >= 8000 : 
                transmit(result)
                result = ['S']
            t = tt[row][col] * 255 / max_disp
            if t > 255 : t = 255            
            t = 255 - t
            if t != 0 :
                result.append(chr(row))
                result.append(chr(col))
                result.append(chr(t))
    transmit(result)

def visu_pos(coords) :
    visu_dim = (100, 100)
    # downsample the matrix for visualization
    row_scale = (coords.shape[0] / float(visu_dim[0]))
    col_scale = (coords.shape[1] / float(visu_dim[1]))
    coords = coords[::row_scale, ::col_scale]
    rows = coords.shape[0]
    cols = coords.shape[1]
    result = ['P'] 
    maxes  = coords.max(axis=0).max(axis=0) 
    mins   = coords.min(axis=0).min(axis=0)
    ranges = maxes - mins
    scales = 255 / ranges
    # print mins, maxes, ranges, scales
    coords = (coords - mins) * scales
    # print coords
    print rows, cols
    for row in range(rows) :
        for col in range(cols) :
            if len(result) >= 8000 : 
                transmit(result)
                result = ['P']
            result.append(chr(row))
            result.append(chr(col))
            coord = coords[row][col]
            result.append(chr(int(coord[0])))
            result.append(chr(int(coord[1])))
            result.append(chr(int(coord[2])))
    transmit(result)

    
grid_rows = len(grid)
grid_cols = len(grid[1])

count = 0
# initialize planar
coords = random.random((grid_rows, grid_cols, 5))
for row in range(grid_rows) :
    for col in range(grid_cols) :
        coords[row][col][0] = row
        coords[row][col][1] = col     

# initialize coords

point_list = [(286, 163)] # chosen to be close to a bart station
for i in range (grid_rows) : 
    for j in range (grid_cols) :
        if grid[i][j][2].degree_out > 4 :
            point_list.append((i, j))

udp = socket( AF_INET, SOCK_DGRAM )
# to test real time of path search portion
# tt = random.random((grid_rows, grid_cols))
print "Number of interesting grid points :", len(point_list)
t_start = time.time()
while (1) :
    shuffle(point_list)
    for o in range(len(point_list)) :
        # point = point_list.pop()
        point = point_list[o]
        tt       = get_times(g, grid, point) # call graphserver
        visu_times(tt)
        # cplot(tt)
        print "Performing signature algorithm..."
        vectors  = coords - coords[point]           # OK - all dimensions to add are indices to the left (tested)
        norms    = sqrt(sum(vectors**2, axis = 2))  # OK for finding norms, tested
        # print coords
        # print norms    
        
        # tt[point] = 0          # point distance to self should be 0 in real data
        adjust     = tt - norms  # both are same dim, ok for element-wise subtraction
        stress     = sqrt(sum(adjust**2))
        stress_rms = sqrt(sum(adjust**2) / adjust.size) 
        print "Stress relative to current point is : %f / %f" % (stress, stress_rms)
        # when norms fall to zero, division gives NaN
        # use nan_to_num() or myarr[np.isnan(myarr)] = 0
        uvectors = vectors  /  norms[:,:,newaxis]  # divide every 3vector element-wise by norms duplicated into direction 2
        dvectors = uvectors * adjust[:,:,newaxis]  # same strategy
        coords  += nan_to_num(dvectors)            # filter out NaNs before backpropagating to coords
        visu_pos(coords)
        # broadcasting always goes up dimensions
        # so shape 3, 3, 2 can be divided by a shape 3, 2 but not a 3, 3
        # this implies that you must put the most general dimesnsion first, like in math notation:
        # X sub 1, X sub 1 prime, Y sub 1, Y sub 1 prime
        #
        # broadcasting can extend an existing axis, or create a new 'higer order axis'
        # to avoid ambiguity, you have to explicitly create a new axis to show which way to broadcast.
        # broadcasting just means implicit, space-saving copy by reference into the inexistent 
        # indices of another dimension.
        count += 1
        if count % 5 == 0 : 
            print "%i iterations averaging %f seconds" % (count, (time.time() - t_start) / count)
            

