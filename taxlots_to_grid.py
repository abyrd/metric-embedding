#!/usr/bin/env python

from graphserver.ext.osm.osmdb import OSMDB
osmdb = OSMDB('/home/andrew/data/pdx/testgrid.osmdb')

c = osmdb.conn.cursor()
c.execute( "DROP TABLE IF EXISTS grid_pop" )
c.execute( "CREATE TABLE grid_pop (id integer primary key, surf FLOAT, pop FLOAT)" )
c.execute( "SELECT AddGeometryColumn('grid_pop', 'geom', 4326, 'POINT', 2)" )
c.execute( "INSERT INTO grid_pop SELECT rowid, 0, 0, geometry FROM grid" )

# 100 meter steps at this latitude: lat 0.000900 lon 0.001279
xstep = 0.001279
ystep = 0.000900
print "Getting all residential building sizes from taxlots table..."
c.execute("""select bldgsqft, x(c), y(c) from (
             select pk_uid, bldgsqft, transform(centroid(geom), 4326) as c 
             from taxlots where (landuse = 'SFR' or landuse = 'MFR') and bldgsqft > 0 )""")

for sqft, x, y in c.fetchall() :
    c.execute( """SELECT rowid, Distance(MakePoint(?, ?, 4326), geometry) AS d 
                  FROM grid WHERE rowid IN (
                  SELECT rowid FROM cache_grid_geometry 
                  WHERE mbr = FilterMBRWithin(?, ?, ?, ?) )
                  ORDER BY d LIMIT 1""", 
                  (x, y, x-xstep, y-ystep, x+xstep, y+ystep) )    
    try:
        rowid, dist = c.next()
    except StopIteration:
        print "No close grid point found."
        c.execute( "INSERT INTO grid_pop (surf, pop, geom) VALUES (-8, 0, MakePoint(?, ?, 4326))", (x, y) )
    else:
        print x, y, sqft, rowid, dist    
        c.execute( "UPDATE grid_pop SET surf = surf + ? WHERE id = ?", (sqft, rowid) )

osmdb.conn.commit()
