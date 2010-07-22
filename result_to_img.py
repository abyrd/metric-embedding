#!/usr/bin/env python

from PIL import Image
import numpy as np
import random
import cPickle as pickle

result = np.load('./result.npy')
xmax   = result.shape[0]
ymax   = result.shape[1] 
MAX_T  = 90 * 60.

infile = open('', 'rb')
stations = pickle.load(infile)
matrix   = np.array(pickle.load(infile), dtype=np.float32) # must be float, because cannot cast None to int32
dists    = pickle.load(infile)



i = 0
while(1) :
    x = int(random.random() * xmax)
    y = int(random.random() * ymax)

    coords = result[x, y]
    if np.isnan(coords[0]) : continue
    print coords
    diff = result - coords
    diff = np.sum(diff ** 2, axis=2)
    dist = np.sqrt(diff)

    dist /= MAX_T
    dist[dist > 1]       = 1
    dist[np.isnan(dist)] = 0
    dist = 1 - dist
    dist *= 255
    im = Image.fromarray(dist.astype(np.uint8), 'L') # greyscale mode, must use UINT8 array!
    im.save('tt_%d.png' % i)
    i += 1
    print i
