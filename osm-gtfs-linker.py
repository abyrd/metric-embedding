#!/usr/bin/env python2.6
#

from graphserver.core import Graph
from graphserver.ext.gtfs.gtfsdb import GTFSDatabase
from graphserver.ext.osm.osmdb import OSMDB, Node, cons

import sys, time
import ligeos as lg
from optparse import OptionParser


def linker(osmdb, gtfsdb, split_threshold = 50, range = 0.005): # meters (max distance to link node), degrees (intersection box size)
    c = osmdb.conn
    c.enable_load_extension(True)
    try :
        c.execute("SELECT load_extension('libspatialite.so.2')")
    except :
        if reporter: reporter.write("You need libspatialite.so.2, and pysqlite must be compiled with enable_load_extension.\n")
        return
    cur = c.cursor()
    cur.execute( "CREATE TABLE way_segments (id INTEGER, way TEXT, dist FLOAT, start_vertex TEXT, end_vertex TEXT)" )
    cur.execute( "SELECT AddGeometryColumn ( 'way_segments', 'GEOMETRY', 4326, 'LINESTRING', 2 )" )
    # can cache be made after loading segments like this?
    # cur.execute( "SELECT CreateMbrCache('way_segments', 'GEOMETRY')" )
    c.commit()

    key = 0
    for i, way in enumerate( osmdb.ways() ) :
        if i % 5000 == 0: print "Way", i
        dist = 0
        for sid, eid in zip(way.nds[:-1], way.nds[1:]) :
            cur.execute( "SELECT lat, lon FROM nodes WHERE id = ?", (sid,) )
            slat, slon = cur.next()
            cur.execute( "SELECT lat, lon FROM nodes WHERE id = ?", (eid,) )
            elat, elon = cur.next()
            # query preparation automatically quotes string for you, no need to explicitly put quotes in the string
            wkt = "LINESTRING(%f %f, %f %f)" % (slon, slat, elon, elat)
            cur.execute( "INSERT INTO way_segments VALUES (?, ?, ?, ?, ?, LineFromText(?, 4326))", (key, way.id, dist, sid, eid, wkt) )
            #length += vincenty(slat, slon, elat, elon) #maybe store in table also
            length = 1
            dist += length
            key += 1
    c.commit()
    EARTH_RADIUS = 6367000
    PI_OVER_180 =  0.017453293    
    # actually don't need dist in meters since you're just looking for closest
    sql = "select *, min(%f * distance(GEOMETRY, makepoint(?, ?))) from (select * from way_segments where MbrIntersects(BuildCircleMbr( ?, ?, ?), GEOMETRY))" % (EARTH_RADIUS * PI_OVER_180)
    for sid, name, lat, lon in gtfsdb.stops() :
        print "stop :", sid, name, lat, lon
        cur.execute( sql, (lon, lat, lon, lat, range) )
        for e in cur.fetchall() : print "    ", e

def main(osm_db, gtfs_db): 
    
    osmdb  = OSMDB( osm_db ) # no overwrite parameter means load existing
    gtfsdb = GTFSDatabase (gtfs_db)
    linker (osmdb, gtfsdb)
    
    print "DONE."
    sys.exit(0)    
            

if __name__=='__main__':
    usage = """usage: python zzzz.py <osm_database> <gtfs_database>"""
    parser = OptionParser(usage=usage)
    
    (options, args) = parser.parse_args()
    
    if len(args) != 2:
        parser.print_help()
        exit(-1)
        
    osm_db  = args[0]
    gtfs_db = args[1]
    
    main(osm_db, gtfs_db)
