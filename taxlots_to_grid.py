#!/usr/bin/env python

from graphserver.ext.osm.osmdb import OSMDB
osmdb = OSMDB('/home/andrew/data/pdx/testgrid.osmdb')

c = osmdb.conn.cursor()
c.execute( "DROP TABLE IF EXISTS grid_pop" )
c.execute( "CREATE TABLE grid_pop (id integer primary key, surf FLOAT, pop FLOAT)" )
c.execute( "SELECT AddGeometryColumn('grid_pop', 'geom_pt', 4326, 'POINT', 2)" )
c.execute( "SELECT AddGeometryColumn('grid_pop', 'geom_poly', 4326, 'POLYGON', 2)" )
c.execute( "INSERT INTO grid_pop SELECT rowid, 0, 0, geom_pt, geom_poly FROM grid" )
c.execute( "SELECT CreateMBRCache('grid_pop', 'geom_poly')" )

print "Getting all residential buildings from taxlots table..."
c.execute("""select pk_uid from taxlots 
             where landuse in ('SFR', 'MFR', 'AGR', 'RUR', 'FOR') 
             and bldgsqft > 0 """)

print "Rasterizing residential surfaces..."
m2_over_ft2 = 0.09290304 # conversion factor (1 m**2 / 1 ft**2)
for i, (lotid,) in enumerate(c.fetchall()) :
    if i % 10000 == 0 : 
        print 'Taxlot', i
        # occasionally commit, otherwise memory corruption errors happen
        # Nope, still getting double free errors in gaiaFreePolygon
        # errors do not occur if results are printed - some kind of timing issue
        osmdb.conn.commit()
    # for each interesting taxlot, get its housing area, lot shape and size
    c.execute("""select bldgsqft, geom, area(geom), mbrminx(g), mbrminy(g), mbrmaxx(g), mbrmaxy(g)
                 from ( select bldgsqft, geom, transform(geom, 4326) as g 
                 from taxlots where pk_uid = ? )""", (lotid,))
    bldg_surf, lot_geom, lot_area, x0, y0, x1, y1 = c.next()
    print lotid, bldg_surf, lot_area, x0, y0, x1, y1
    # convert housing area to square metres
    bldg_surf *= m2_over_ft2 
    #print "%f m2 housing on %f ft2 lot" % (bldg_surf, lot_area)

    # for this taxlot, get all cells that intersect it 
    # as well as what proportion of the lot's area they intersect
    c.execute( """SELECT rowid, area(intersection(transform(geom_poly, 2269), ?)) AS a
                  FROM grid_pop WHERE rowid IN (
                  SELECT rowid FROM cache_grid_pop_geom_poly 
                  WHERE mbr = FilterMBRIntersects(?, ?, ?, ?) )""", 
                  (lot_geom, x0, y0, x1, y1) )
    for gridid, area in c.fetchall():
        print "  -", gridid, area
        # what if there are no points found? should those points be marked?
        if area:
            #print "   -- %f%% (%f ft2)" % (area/lot_area * 100, area) 
            c.execute( "UPDATE grid_pop SET surf = surf + ? WHERE id = ?", (bldg_surf * (area / lot_area), gridid) )

osmdb.conn.commit()
