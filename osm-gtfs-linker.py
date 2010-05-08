#!/usr/bin/env python2.6
#

from graphserver.core import Graph
from graphserver.ext.gtfs.gtfsdb import GTFSDatabase
from graphserver.ext.osm.osmdb import OSMDB, Node, cons

from geotools import *
import sys, time
import ligeos as lg
from optparse import OptionParser


def linker(osmdb, gtfsdb, split_threshold = 50, range = 0.005): # meters (max distance to link node), degrees (intersection box size)
    c = osmdb.conn
    cur = c.cursor()
    cur.execute( "DROP TABLE IF EXISTS way_segments" )
    cur.execute( "CREATE TABLE way_segments (id INTEGER, way INTEGER, dist FLOAT, length FLOAT, start_vertex TEXT, end_vertex TEXT)" )
    cur.execute( "SELECT AddGeometryColumn ( 'way_segments', 'geometry', 4326, 'LINESTRING', 2 )" )
    # can cache be made after loading segments like this?
    # cur.execute( "SELECT CreateMbrCache('way_segments', 'geometry')" )
    c.commit()

    key = 0
    for i, way in enumerate( osmdb.ways() ) :
        if i % 5000 == 0: print "Way", i
        dist = 0
        for sid, eid in zip(way.nds[:-1], way.nds[1:]) :
            try:
                cur.execute( "SELECT lat, lon, alias FROM nodes WHERE id = ?", (sid,) )
                slat, slon, sv = cur.next()
                cur.execute( "SELECT lat, lon, alias FROM nodes WHERE id = ?", (eid,) )
                elat, elon, ev = cur.next()
                # print "SUCCEED!"
            except:
                # print "FAIL!"
                # i.e. the referenced node was not found in our database
                continue
            # query preparation automatically quotes string for you, no need to explicitly put quotes in the string
            wkt = "LINESTRING(%f %f, %f %f)" % (slon, slat, elon, elat)
            length = geoid_dist(slat, slon, elat, elon) # in meters. spatialite 3.4 will have this function built in
            if length <= 0 : print "zero or negative length way segment?!"
            cur.execute( "INSERT INTO way_segments VALUES (?, ?, ?, ?, ?, ?, LineFromText(?, 4326))", (key, way.id, dist, length, sv, ev, wkt) )
            dist += length
            key += 1
    cur.execute( "CREATE INDEX way_segments_id ON way_segments (id)" )
    c.commit()
    
    # This whole method should really be in a osmdb.find_or_make_link_vertex() method
    # Called from gtfsdb.link_to_osmdb(osmdb)
    # Which cycles through all stops, calling the osmdb function and saving references.
    
    EARTH_RADIUS = 6367000
    PI_OVER_180 =  0.017453293    
    # to make unique ids for new segments (can't sql already do this?)
    # actually don't need dist in meters since you're just looking for closest
    sql = "SELECT *, distance(geometry, makepoint(?, ?)) AS d FROM (SELECT * FROM way_segments WHERE MbrContains(BuildCircleMbr( ?, ?, ?), geometry)) ORDER BY d LIMIT 1"
    for stopid, name, lat, lon in gtfsdb.stops() :
        print "stop :", stopid, name, lat, lon
        cur.execute( sql, (lon, lat, lon, lat, range) )
        seg = cur.next() 
        segid, way, off, length, sv, ev, geom, d = seg
        print "    ", seg
        cur.execute( "SELECT *, distance(geometry, makepoint(?, ?)) AS d FROM vertices WHERE id IN (?, ?) ORDER BY d LIMIT 1", (lon, lat, sv, ev) )
        end = cur.next()
        vid, refs, geom, d = end
        print "    ", end
        d *= EARTH_RADIUS * PI_OVER_180
        print d, "meters"
        if d < split_threshold :
            print "link to existing vertex", vid
        else :
            # split!
            print "existing vertex is beyond threshold. splitting way segment."
            cur.execute( "SELECT x(geometry), y(geometry) FROM vertices WHERE id = ?", (sv,) )
            slon, slat = cur.next()
            cur.execute( "SELECT x(geometry), y(geometry) FROM vertices WHERE id = ?", (ev,) )
            elon, elat = cur.next()
            ls  = lg.LineString([(slon, slat), (elon, elat)], geographic=True)
            # dist   = ls.distance_pt(stop)
            # find ideal place to link
            stop = lg.GPoint(lon, lat)
            pt   = ls.closest_pt(stop)
            loc  = ls.locate_point(pt)            
            # make new vertex named wWAYdOFFSET
            new_v_name = "w%do%d" % (way, off + loc)
            cur.execute( "INSERT INTO vertices (id, refs, geometry) VALUES (?, 1, MakePoint(?, ?, 4326))", (new_v_name, pt.x, pt.y) )
            cur.execute( "SELECT max(id) FROM way_segments" )
            (max_segid,) = cur.next()
            # make 2 new segments
            wkt = "LINESTRING(%f %f, %f %f)" % (slon, slat, pt.x, pt.y)
            cur.execute( "INSERT INTO way_segments (id, way, dist, length, start_vertex, end_vertex, geometry) VALUES (?, ?, ?, ?, ?, ?, LineFromText(?, 4326))", 
                         (max_segid + 1, way, off, loc, sv, new_v_name, wkt) )
            wkt = "LINESTRING(%f %f, %f %f)" % (pt.x, pt.y, elon, elat)
            cur.execute( "INSERT INTO way_segments (id, way, dist, length, start_vertex, end_vertex, geometry) VALUES (?, ?, ?, ?, ?, ?, LineFromText(?, 4326))", 
                         (max_segid + 2, way, off + loc, length - loc, new_v_name, ev, wkt) )
            # drop old segment 
            cur.execute( "DELETE FROM way_segments WHERE id = ?", (segid,) )
            print "link to new vertex: ", new_v_name

        c.commit()
    ### OSM edges are never stored, but both edges and vertices (for final graph) are simply built from an ordered way segment table.
    
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
