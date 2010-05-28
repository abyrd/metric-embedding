#!/usr/bin/env python

from graphserver.ext.osm.osmdb import OSMDB
osmdb = OSMDB('/home/andrew/data/pdx/testgrid.osmdb')

c = osmdb.conn.cursor()
# this is great, you can retrieve a geometry and send it back to SQLite as a parameter!
c.execute( "SELECT *, mbrminx(g), mbrminy(g), mbrmaxx(g), mbrmaxy(g) from (select pk_uid, pop00, transform(geom, 4326) as g from blockgrp)" )
# no need to make an index to join grid and grid_pop because they are being joined on primary keys
for grpid, pop, geom, minx, miny, maxx, maxy in c.fetchall():
    # you could use the grid mbr cache for grid_pop because they have the same row ids,    
    # except that you want to compare exact geometry in another clause
    c.execute( """SELECT group_concat(g.rowid), sum(gp.surf) FROM grid as g, grid_pop as gp 
                  WHERE g.rowid = gp.rowid
                  AND g.rowid IN (
                  SELECT rowid FROM cache_grid_geometry WHERE mbr = FilterMBRWithin(?, ?, ?, ?))
                  AND within(g.geometry, ?)""", 
                  (minx, miny, maxx, maxy, geom) )
    grid_ids, surf = c.next()
    if not grid_ids :    
        print "No gridpoints within block group with population", pop
    else :
        if surf == 0 :
            print "Gridpoints inside block group have 0 surface on which to place %d people." % pop 
            # spread them out
            n = float(len(grid_ids.split(',')))
            c.execute( "UPDATE grid_pop SET pop = pop + %f WHERE rowid IN (%s)" % (pop/n, grid_ids) )            
        else:       
            print "Distribute %d people among %f square feet: %f sqft per person." % (pop, surf, surf/pop)
            c.execute( "UPDATE grid_pop SET pop = pop + surf * %f WHERE rowid IN (%s)" % (pop/surf, grid_ids) )

osmdb.conn.commit()
