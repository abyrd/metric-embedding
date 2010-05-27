#!/usr/bin/env python

from graphserver.ext.osm.osmdb   import OSMDB
from PIL import Image
from math import log, sqrt

osmdb  = OSMDB ('/home/andrew/data/pdx/testgrid.osmdb'  )

print "Fetching grid from OSMDB..."
grid = list(osmdb.execute("SELECT g.x, g.y, gp.surf FROM grid AS g, grid_pop AS gp WHERE g.rowid = gp.rowid"))
max_x, max_y = osmdb.execute("SELECT max(x), max(y) FROM grid").next()

max_pop, = osmdb.execute("SELECT max(surf) FROM grid_pop").next()

max_x += 1
max_y += 1
print max_pop
#max_pop = 200.
print "saving image..."
im = Image.new("L", (max_x, max_y))
for (x, y, pop) in grid :
    if pop == 0: continue
    c = int((sqrt(pop)/sqrt(max_pop)) * 255)
    im.putpixel((x, max_y - y - 1), c)

im.save('population.png')

