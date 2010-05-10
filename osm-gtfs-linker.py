#!/usr/bin/env python2.6
#

from graphserver.core import Graph
from graphserver.ext.gtfs.gtfsdb import GTFSDatabase
from graphserver.ext.osm.osmdb import OSMDB, Node, cons

from geotools import *
import sys, time
import ligeos as lg
from optparse import OptionParser


def linker(osmdb, gtfsdb, split_threshold = 50, range = 0.01): # meters (max distance to link node), degrees (spatial index box size)

    # This whole method should really be in a osmdb.find_or_make_link_vertex() method
    # Called from gtfsdb.link_to_osmdb(osmdb)
    # Which cycles through all stops, calling the osmdb function and saving references.
    c = osmdb.conn
    cur = c.cursor()
    
    gcur = gtfsdb.get_cursor()
    gcur.execute( "DROP TABLE IF EXISTS osm_links" )
    gcur.execute( "CREATE TABLE osm_links (gtfs_stop TEXT, osm_vertex TEXT)" )
    gtfsdb.conn.commit() 
        
    EARTH_RADIUS = 6367000
    PI_OVER_180 =  0.017453293    
    # here you don't need distance in meters since you're just looking for closest
    sql = """SELECT *, distance(geometry, makepoint(?, ?)) AS d FROM way_segments WHERE id IN 
             (SELECT pkid FROM idx_way_segments_geometry where xmin > ? and xmax < ? and ymin > ? and ymax < ?) 
             ORDER BY d LIMIT 1"""
    nstops = gtfsdb.count_stops()
    # for every stop in the GTFS database
    for (i, (stopid, name, lat, lon)) in enumerate(gtfsdb.stops()) :
        print "stop %i / %i :" % (i, nstops), stopid, name, lat, lon
        # get the closest way segment
        cur.execute( sql, (lon, lat, lon - range, lon + range, lat - range, lat + range) )
        seg = cur.next() 
        segid, way, off, length, sv, ev, geom, d = seg
        print "    Found segment %d. way %d offset %d." % (segid, way, off)
        # get the closest endpoint of this segment and its distance
        cur.execute( "SELECT *, distance(geometry, makepoint(?, ?)) AS d FROM vertices WHERE id IN (?, ?) ORDER BY d LIMIT 1", (lon, lat, sv, ev) )
        end = cur.next()
        vid, refs, geom, d = end
        d *= (EARTH_RADIUS * PI_OVER_180)
        print "    Closest endpoint vertex %s at %d meters" % (vid, d)
        if d < split_threshold :
            print "    Link to existing vertex."
            # should return or save vertex id in gtfsdb
            cur.execute( "UPDATE vertices SET refs = refs + 1 WHERE id = ?", (vid,) )
            gcur.execute( "INSERT INTO osm_links VALUES (?, ?)", (stopid, vid) )
        else :
            # split the way segment in pieces to make a better linking point
            print "    Existing vertex beyond threshold. Splitting way segment."
            # get the segment start vertex coordinates
            cur.execute( "SELECT x(geometry), y(geometry) FROM vertices WHERE id = ?", (sv,) )
            slon, slat = cur.next()
            # get the segment end vertex coordinates
            cur.execute( "SELECT x(geometry), y(geometry) FROM vertices WHERE id = ?", (ev,) )
            elon, elat = cur.next()
            # make a linestring for the existing segment
            ls  = lg.LineString([(slon, slat), (elon, elat)], geographic=True)
            # find the ideal place to link
            stop = lg.GPoint(lon, lat)
            dist = ls.distance_pt(stop)
            pt   = ls.closest_pt(stop)
# SHOULD CHECK THAT NEW POINT IS NOT FARTHER THAN threshold, otherwise you get useless splittings.
            # and its distance along the segment (float in range 0 to 1)
            pos  = ls.locate_point(pt) 
            pos *= length 
            print "    Ideal link point %d meters away, %d meters along segment." % (dist, pos)
            # make new vertex named wWAYdOFFSET
            new_v_name = "w%do%d" % (way, off + pos)
            cur.execute( "INSERT INTO vertices (id, refs, geometry) VALUES (?, 2, MakePoint(?, ?, 4326))", (new_v_name, pt.x, pt.y) )
            # DEBUG make a new vertex to show stop location
            cur.execute( "INSERT INTO vertices (id, refs, geometry) VALUES ('gtfs_stop', 0, MakePoint(?, ?, 4326))", (lon, lat) )
            cur.execute( "SELECT max(id) FROM way_segments" )
            (max_segid,) = cur.next()
            # make 2 new segments
            wkt = "LINESTRING(%f %f, %f %f)" % (slon, slat, pt.x, pt.y)
            cur.execute( "INSERT INTO way_segments (id, way, dist, length, start_vertex, end_vertex, geometry) VALUES (?, ?, ?, ?, ?, ?, LineFromText(?, 4326))", 
                         (max_segid + 1, way, off, pos, sv, new_v_name, wkt) )
            wkt = "LINESTRING(%f %f, %f %f)" % (pt.x, pt.y, elon, elat)
            cur.execute( "INSERT INTO way_segments (id, way, dist, length, start_vertex, end_vertex, geometry) VALUES (?, ?, ?, ?, ?, ?, LineFromText(?, 4326))", 
                         (max_segid + 2, way, off + pos, length - pos, new_v_name, ev, wkt) )
            # drop old segment 
            cur.execute( "DELETE FROM way_segments WHERE id = ?", (segid,) )
            print "    Link to new vertex:", new_v_name
            gcur.execute( "INSERT INTO osm_links VALUES (?, ?)", (stopid, new_v_name) )

        c.commit()
        gtfsdb.conn.commit()
        print ""
    
def make_edges(osmdb): 
    print "Converting way segments into graph edges..."
    c = osmdb.conn
    cur = c.cursor()
    cur.execute( "DROP TABLE IF EXISTS edges" )
    cur.execute( "CREATE TABLE edges (id TEXT, start_vertex TEXT, end_vertex TEXT)" )
    cur.execute( "SELECT AddGeometryColumn ( 'edges', 'geometry', 4326, 'LINESTRING', 2 )" )
    c.commit()
    def insert_edge( way, idx, sv, ev, geom ):
        wkt = "LINESTRING( %s )" % ( ','.join(geom) )
        id  = "w%d-%d" % (way, idx)
        cur.execute( "INSERT INTO edges (id, start_vertex, end_vertex, geometry) VALUES (?, ?, ?, LinestringFromText(?, 4326))", (id, sv, ev, wkt) )
        # print "inserted", id
        
    last_way = -1
    edge_length = 0
    cur.execute( "SELECT way, dist, length, start_vertex, end_vertex FROM way_segments ORDER BY way, dist" )
    way_count = 0
    for way, dist, length, sv, ev in cur.fetchall() :
        try :
            # get the segment start vertex coordinates
            cur.execute( "SELECT x(geometry), y(geometry), refs FROM vertices WHERE id = ?", (sv,) )
            slon, slat, srefs = cur.next()
            # get the segment end vertex coordinates
            cur.execute( "SELECT x(geometry), y(geometry), refs FROM vertices WHERE id = ?", (ev,) )
            elon, elat, erefs = cur.next()
        except :
            # print "error fetching info on vertices", sv, ev
            # this is a real problem! missing vertices!
            continue
            
        if way != last_way :
            if edge_length > 0 :
                insert_edge( last_way, idx, edge_sv, last_ev, edge_geom )
            if last_way != -1 : 
                # print "Inserted %d edges for way %d." % (idx, last_way)
                way_count += 1
                if way_count % 5000 == 0 : print "Way", way_count
            idx = 0
            edge_sv = sv
            edge_length = 0
            edge_geom = ["%f %f" % (slon, slat)]
        edge_geom.append("%f %f" % (elon, elat))
        edge_length += length
        if erefs > 1 :
            insert_edge( way, idx, edge_sv, ev, edge_geom )
            idx += 1
            edge_sv = ev
            edge_length = 0
            edge_geom = ["%f %f" % (elon, elat)]
        last_way = way 
        last_ev  = ev        

    c.commit()

    # remove all vertices with less than 2 refs
    ### actually, OSM edges are never stored, but both edges and vertices (for final graph) are simply built from an ordered way segment table.

def main(osm_db, gtfs_db): 
    
    osmdb  = OSMDB( osm_db ) # no overwrite parameter means load existing
    gtfsdb = GTFSDatabase( gtfs_db )
    linker( osmdb, gtfsdb )
    make_edges( osmdb )
    
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
