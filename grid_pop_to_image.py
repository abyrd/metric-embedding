#!/usr/bin/env python

from graphserver.ext.osm.osmdb   import OSMDB
from PIL import Image
from math import log, sqrt

osmdb  = OSMDB ('/home/andrew/data/pdx/testgrid.osmdb'  )

print "Fetching grid from OSMDB..."
grid = list(osmdb.execute("SELECT g.x, g.y, gp.surf, gp.pop FROM grid AS g, grid_pop AS gp WHERE g.rowid = gp.rowid"))
max_x, max_y = osmdb.execute("SELECT max(x), max(y) FROM grid").next()

max_pop,  = osmdb.execute("SELECT max(pop)  FROM grid_pop").next()
max_surf, = osmdb.execute("SELECT max(surf) FROM grid_pop").next()

max_x += 1
max_y += 1
print max_surf, max_pop

print "saving image..."
im_surf = Image.new("L", (max_x, max_y))
im_pop  = Image.new("L", (max_x, max_y))
for (x, y, surf, pop) in grid :
    # s = int((sqrt(surf)/sqrt(max_surf)) * 255)
    s = int((surf/max_surf) * 255)
    im_surf.putpixel((x, max_y - y - 1), s)
    # p = int((sqrt(pop)/sqrt(max_pop)) * 255)
    p = int((pop / max_pop) * 255)
    im_pop.putpixel((x, max_y - y - 1), p)

im_pop.save('population.png')
im_surf.save('residential_surface.png')

