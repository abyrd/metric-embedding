#!/usr/bin/env python2.6

from graphserver.ext.gtfs.gtfsdb import GTFSDatabase
from graphserver.ext.osm.osmdb import OSMDB, Node

import sys, time
import ligeos as lg
from optparse import OptionParser
from rtree import Rtree

from geotools import geoid_dist

def main(split_threshold = 50, range = 0.005): # meters (max distance to link node), degrees (intersection box size)
    split_idx = 1000000000 # base node/edge number. above usual OSM ids, but still leaves half of 32bit int range.
    usage = """usage: python stationsplit_osmdb.py <osmdb_filename> <gtfsdb_filename>"""
    parser = OptionParser(usage=usage)
    
    (options, args) = parser.parse_args()
    
    if len(args) != 2:
        parser.print_help()
        exit(-1)
        
    osmdb_filename   = args[0]
    gtfsdb_filename  = args[1]
    
    osmdb = OSMDB( osmdb_filename )
    gtfsdb = GTFSDatabase( gtfsdb_filename )
    
    n_stops = gtfsdb.count_stops()
    #edge_index = Rtree(osmdb_filename+'-edges')
    edge_index = Rtree()

    def index_edge(edge_rid, edge_id) :
        start_nd, end_nd = osmdb.execute( "SELECT start_nd, end_nd FROM edges WHERE id = ?", (edge_id,) ).next()
        s_lat, s_lon = osmdb.execute("SELECT lat, lon FROM nodes WHERE id = ?", (start_nd,) ).next()
        e_lat, e_lon = osmdb.execute("SELECT lat, lon FROM nodes WHERE id = ?", (end_nd,) ).next()
        # why float casting?
        #s_lat = float(s_lat)
        #s_lon = float(s_lon)
        #e_lat = float(e_lat)
        #e_lon = float(e_lon)
        box = (min(s_lon, e_lon), min(s_lat, e_lat), max(s_lon, e_lon), max(s_lat, e_lat))
        # print parent_id, box, edge_id
        # Need our own index number to remove edges as we go along.
        # This index could ideally be generated automatically, with collision detection.
        edge_index.add( edge_rid, box, obj = edge_id )

    print 'indexing OSM edges into rtree' 
    for rid, edge in enumerate(osmdb.execute("SELECT id FROM edges")) :
        index_edge(rid, edge[0]) # take first element because db response is a tuple
        
    def nearest_point_on_edge((edge_rid, edge_id, edge_bbox), stop_lat, stop_lon) :
        # print "finding nearest point on edge %s (rtree id %d)" % (edge_id, edge_rid)
        way_id, start_nd, end_nd, length = osmdb.execute( "SELECT parent_id, start_nd, end_nd, dist FROM edges WHERE id = ?", (edge_id,) ).next()
        s_lat, s_lon = osmdb.execute("SELECT lat, lon FROM nodes WHERE id = ?", (start_nd,) ).next()
        e_lat, e_lon = osmdb.execute("SELECT lat, lon FROM nodes WHERE id = ?", (end_nd,) ).next()
        # print (s_lat, s_lon, e_lat, e_lon, stop_lat, stop_lon)
        stop   = lg.GPoint(stop_lon, stop_lat)
        ls     = lg.LineString([(s_lon, s_lat), (e_lon, e_lat)], geographic=True)
        dist   = ls.distance_pt(stop)
        pt     = ls.closest_pt(stop)
        loc    = ls.locate_point(pt)
        return edge_rid, edge_id, edge_bbox, way_id, start_nd, end_nd, dist, pt, loc, length

    print 'finding nodes near stations...'
    for i, (stop_id, stop_name, stop_lat, stop_lon) in enumerate( gtfsdb.stops() ):
        print "GTFS stop %d/%d (id %s)"%(i,n_stops,stop_id)
        
        # 1. Find closest existing node        
        nd_id, nd_lat, nd_lon, nd_dist = osmdb.nearest_node( stop_lat, stop_lon )
        nd_dist = geoid_dist(nd_lat, nd_lon, stop_lat, stop_lon)
        print "Closest existing node:  %s at %08.5f %09.5f (%3.1f meters away)" % (nd_id, nd_lat, nd_lon, nd_dist)
       
        # 2. Find closest possible place to make a node on an existing edge (the ideal link point)
        nearby_items = edge_index.intersection( (stop_lon - range, stop_lat - range, stop_lon + range, stop_lat + range), objects=True )
        rtree_items = [(i.id, i.object, i.bbox) for i in nearby_items]
        candidates = [nearest_point_on_edge(rtree_item, stop_lat, stop_lon) for rtree_item in rtree_items]
        edge_rid, edge_id, edge_bbox, way_id, start_nd, end_nd, dist, pt, loc, length = min(candidates, key=lambda x:x[6])
        print "Ideal linking location: edge %s position %04.2f (%d meters away)" % (edge_id, loc, dist)
        # 3. Check if closest existing node is an endpoint of the way containing the ideal point
        #    Also check if the distance is over the split threshold
        if (start_nd != nd_id and end_nd != nd_id) or nd_dist > split_threshold :
            # The existing point is not a good candidate. Make a new one by splitting the edge.
            print 'Closest existing node rejected. Splitting OSM edge %s.' % (edge_id,)
            # make a new node at the right place
            split_idx += 1
            print 'Adding new node: %d' % split_idx
            # id of new node must be numeric because nearest node index uses it as a long int
            new_nd = Node( split_idx, pt.x, pt.y )
            osmdb.add_node(new_nd) # pass cursor parameter, will not close cursor
            # add the new node to the rtree index
            osmdb.index.add( split_idx, (pt.x, pt.y) ) # no need to repeat coords? in new Rtree lib
            # remove the old edge from the database
            c = osmdb.cursor()
            c.execute("DELETE FROM edges WHERE id = ?", (edge_id,))
            # remove the old edge from the edges r-tree
            # cannot use the same range, since the find operation is an intersection
            # but in this case the edge must be _within_ the bounding box
            # print 'deleting rtree entry for edge with rid %d bounds %s' % (edge_rid, edge_bounds)
            edge_index.delete(edge_rid, (edge_bbox))
            
            # add 2 new edges
            c.execute( "INSERT INTO edges VALUES (?, ?, ?, ?, ?, ?)", ('to-%d' % split_idx,
                                                                       way_id,
                                                                       start_nd,
                                                                       split_idx,
                                                                       loc * length,
                                                                       '') )
            c.execute( "INSERT INTO edges VALUES (?, ?, ?, ?, ?, ?)", ('from-%d' % split_idx,
                                                                       way_id,
                                                                       split_idx,
                                                                       end_nd,
                                                                       (1-loc) * length,
                                                                       '') )
            osmdb.conn.commit()
            c.close()
            # print osmdb.edge(split_name+'_A')
            # print osmdb.edge(split_name+'_B')
            # index the new edges into the r-tree
            index_edge(split_idx, 'to-%d' % (split_idx,))
            # advance the split index so edges have different ids in rtree
            split_idx += 1                                                               
            index_edge(split_idx, 'from-%d' % (split_idx - 1, ))                                                             
            
    # the all-important commit call (not necessary if it's called at the end of each loop.
    # osmdbs don't have a commit function, so must use the connection member directly
    # osmdb.conn.commit()
    # c.close()
            
if __name__=='__main__':
    main()
