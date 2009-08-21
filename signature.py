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
import struct


def get_times(g, grid, origin, t0) :
    # print "Performing path search..."
    v_label = 'grid-%i-%i' % (origin)
    g.spt_in_place(v_label, None, State(1, t0))
    # this is the slow line ! must make static grid of references...
    # return array( [ [ (c[2].payload.time - t0) for c in r] for r in grid ] )

def cplot(Z) :
    imshow(Z, cmap=cm.gray)
    levels = array([15, 30, 60, 90])
    CS = contour(Z, levels, linewidths=2)
    clabel(CS, inline=1, fmt='%d', fontsize=14)
    show()

def transmit(result) :
    udp.sendto( result, ('localhost', 6000) )
    # print "sent %d values" % len(result)

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
    result = 'T' 
    for row in range(rows) :
        for col in range(cols) :
            if len(result) >= 8000 : 
                transmit(result)
                result = 'T'
            t = tt[row][col] * 255 / max_disp
            if t > 255 : t = 255            
            t = 255 - t
            if t != 0 :
                # be sure to put in network byte order (exclamation point in format string) 
                # so it can be read by Java or anyone else. Though with only 1 byte chars it's not important.
                result += struct.pack('!ccc', chr(row), chr(col), chr(t))
    transmit(result)

def visu_pos(coords) :
    visu_dim = (100, 100)
    # downsample the matrix for visualization
    row_scale = (coords.shape[0] / float(visu_dim[0]))
    col_scale = (coords.shape[1] / float(visu_dim[1]))
    coords = coords[::row_scale, ::col_scale]
    rows = coords.shape[0]
    cols = coords.shape[1]
    result = 'P' 
    maxes  = coords.max(axis=0).max(axis=0) 
    mins   = coords.min(axis=0).min(axis=0)
    ranges = maxes - mins
    scales = 2000.0 / ranges
    # print mins, maxes, ranges, scales
    coords = (coords - mins) * scales - 1000.0
    # print coords
    # print rows, cols
    for row in range(rows) :
        for col in range(cols) :
            if len(result) >= 8000 : 
                transmit(result)
                result = 'P'
            coord = coords[row][col]
            # print (chr(row), chr(col), coord[0], coord[1], coord[2])
            # be sure to put in network byte order (exclamation point in format string) 
            # so it can be read by Java or anyone else.
            result += struct.pack('!ccfff', chr(row), chr(col), coord[0], coord[1], coord[2])
    transmit(result)

def visu_pos_b(coords) :
    cnum = 0
    result = 'S' 
    maxes  = coords.max(axis=0).max(axis=0) 
    mins   = coords.min(axis=0).min(axis=0)
    ranges = maxes - mins
    scales = 2000.0 / ranges
    coords = (coords - mins) * scales - 1000.0
    for coord in coords :
        if len(result) >= 8000 : 
            transmit(result)
            result = 'S'
        result += struct.pack('!ifff', cnum, coord[0], coord[1], coord[2])
        cnum += 1
    transmit(result)

    
gtfsdb = GTFSDatabase  ( '../gsdata/bart.gtfsdb' )
gdb    = GraphDatabase ( '../gsdata/bart.linked.gsdb' )
#gtfsdb = GTFSDatabase  ( '../gsdata/trimet_13sep2009.gtfsdb' )
#gdb    = GraphDatabase ( '../gsdata/trimet_13sep2009.linked.gsdb' )
g = gdb.incarnate()
# doubling radius from 500 to 1000 meters roughly doubles path search time and quadruples number of points in use
# setting radius to 100 works quickly on small set to test convergence and set parameters
grid, point_list = geotools.create_grid( g, gtfsdb, link_radius = 500)

# point_list = [(286, 163)] # chosen to be close to a bart station
print "Number of interesting grid points :", len(point_list)
            
# initialize coords to random
coords = random.random((len(point_list), 4)) # * 200
# initialize first two dimensions to planar coords
# to shuffle at every iteration, coords and velocities must be shuffled identically to point list
shuffle(point_list)
for i in range(len(point_list)) :
    coords[i][0] = point_list[i][0]
    coords[i][1] = point_list[i][1]
print "Coordinates initialized."

# points start with zero velocity
vels = zeros(coords.shape)

# fix travel times in order to test array copy time
# tt = ones(len(point_list)) * 100

t0 = 1253820000
udp = socket( AF_INET, SOCK_DGRAM )
TIMESTEP = 1.0
PASSES = 10
# 0.5 seems to converge best (10 passes), 0.3 about the same 
NORMALIZE_FACTOR = 0.5/len(point_list)
n_pass  = 0
while (1) :
    pass_err = 0
    # shuffle(point_list)
    t_start = time.time()
    n_iter = 0
    accels = zeros(coords.shape) 
    # what if velocities are nullified every cycle, i.e. vels = accels? YES! converges.
    for o in range(len(point_list)) :
        # point = point_list.pop()
        point = point_list[o][0:2]
        tt = zeros(len(point_list))
        get_times(g, grid, point, t0) # call graphserver
        # this copy operation takes 38 percent of the time
        # it should be eliminated
        for i in range(len(point_list)) : 
            tt[i] = point_list[i][2].payload.time - t0
        # visu_times(tt)
        # cplot(tt)
        # print "Performing signature algorithm..."
        vectors  = coords - coords[o]               # OK - all dimensions to add are indices to the left (tested)
        norms    = sqrt(sum(vectors**2, axis = 1))  # OK for finding norms, tested
        # print coords
        # print norms    
        
        # tt[point] = 0          # point distance to self should be 0 in real data
        adjust     = tt - norms  # both are same dim, ok for element-wise subtraction
        # stress     = sqrt(sum(adjust**2))
        # stress_rms = sqrt(sum(adjust**2) / adjust.size) 
        # print "Stress relative to current point is : %f / %f / %f" % (stress, stress_rms, stress_b)
        # when norms fall to zero, division gives NaN
        # use nan_to_num() or myarr[np.isnan(myarr)] = 0
        uvectors = vectors  /  norms[:,newaxis]  # divide every 3vector element-wise by norms duplicated into direction 2
        accels  += nan_to_num(uvectors * adjust[:,newaxis]) * NORMALIZE_FACTOR  # same strategy. filter out NaNs before using.
        # should this also be done after each full pass?
        # vels += accels  # instantaneous acceleration vectors added to existing velocities
        # vels[point] -= accels.sum(axis=0).sum(axis=0)  # they should also push back collectively on the origin point 
        # update pos here or after a full pass? after a pass looks much more stable and predictable.
        # coords  += vels * TIMESTEP  # timestep * velocity applied to move points
        # visu_pos(coords)
        # broadcasting always goes up dimensions
        # so shape 3, 3, 2 can be divided by a shape 3, 2 but not a 3, 3
        # this implies that you must put the most general dimesnsion first, like in math notation:
        # X sub 1, X sub 1 prime, Y sub 1, Y sub 1 prime
        #
        # broadcasting can extend an existing axis, or create a new 'higer order axis'
        # to avoid ambiguity, you have to explicitly create a new axis to show which way to broadcast.
        # broadcasting just means implicit, space-saving copy by reference into the inexistent 
        # indices of another dimension.
        pass_err += sum(abs(adjust))
        n_iter += 1
        if n_iter % 50 == 0 : 
            print "%i%% (%i iterations averaging %f seconds)" % (n_iter * 100 / len(point_list), n_iter, (time.time() - t_start) / n_iter)
        #    stress = sqrt(sum(adjust**2) / sum(norms**2) )
        #    print "Stress relative to current point is : %f" % (stress)
    # update positions here or during loop? here seems better
    n_pass += 1
    vels = accels  * TIMESTEP  # instantaneous acceleration vectors added to existing velocities
    pass_err = pass_err / float(len(point_list) ** 2)
    k_energy = sum( sqrt( (vels**2).sum(axis=1) ) )
    avg_k_energy = k_energy / len(point_list)
    print "Average absolute error %i." % pass_err
    print "Kinetic energy total %i average %i." % (k_energy, avg_k_energy)
    if avg_k_energy < 20 :
        break
    coords  += vels * TIMESTEP  # timestep * velocity applied to move points
    visu_pos_b(coords)
    print "End of pass number %i." % n_pass
    
for c in coords:
    for e in c :
        print e, ',',
    print