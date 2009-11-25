#!/usr/bin/env python2.6

from graphserver.ext.gtfs.gtfsdb import GTFSDatabase
from graphserver.ext.osm.osmdb import OSMDB, Node

import sys, time
import ligeos as lg
from optparse import OptionParser
from rtree import Rtree

def main(split_threshold = 0.001):
    split_idx = 1000000000 # above usual OSM ids, but half max 32bit int
    range = 0.005
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
        print "%d/%d"%(i,n_stops)
        nd_id, nd_lat, nd_lon, nd_dist = osmdb.nearest_node( stop_lat, stop_lon )
        # if node is too far away, split a nearby edge and make a closer one
        if nd_dist > split_threshold :
            nearby_items = edge_index.intersection( (stop_lon - range, stop_lat - range, stop_lon + range, stop_lat + range), objects=True )
            rtree_items = [(i.id, i.object, i.bbox) for i in nearby_items]
            candidates = [nearest_point_on_edge(rtree_item, stop_lat, stop_lon) for rtree_item in rtree_items]
            edge_rid, edge_id, edge_bbox, way_id, start_nd, end_nd, dist, pt, loc, length = min(candidates, key=lambda x:x[6])
            print 'splitting OSM edge %s to make point at dist %fm loc %f.' % (edge_id, dist, loc)
            # make a new node at the right place
            split_idx += 1
            split_name = "%d" % split_idx
            new_nd = Node(split_name, pt.x, pt.y)
            osmdb.add_node(new_nd) # pass cursor parameter, will not close cursor
            # add the new node to the rtree index
            osmdb.index.add( split_idx, (pt.x, pt.y, pt.x, pt.y) )
            # remove the old edge from the database
            c = osmdb.cursor()
            c.execute("DELETE FROM edges WHERE id = ?", (edge_id,))
            # remove the old edge from the edges r-tree
            # cannot use the same range, since the find operation is an intersection
            # but in this case the edge must be _within_ the bounding box
            # print 'deleting rtree entry for edge with rid %d bounds %s' % (edge_rid, edge_bounds)
            edge_index.delete(edge_rid, (edge_bbox))
            
            # add 2 new edges
            c.execute( "INSERT INTO edges VALUES (?, ?, ?, ?, ?, ?)", (split_name+'_A',
                                                                           way_id,
                                                                           start_nd,
                                                                           split_name,
                                                                           loc * length,
                                                                           '') )
            c.execute( "INSERT INTO edges VALUES (?, ?, ?, ?, ?, ?)", (split_name+'_B',
                                                                           way_id,
                                                                           split_name,
                                                                           end_nd,
                                                                           (1-loc) * length,
                                                                           '') )
            osmdb.conn.commit()
            c.close()
            # print osmdb.edge(split_name+'_A')
            # print osmdb.edge(split_name+'_B')
            # index the new edges into the r-tree
            index_edge(split_idx, split_name+'_A')
            # advance the split index so edges have different ids in rtree
            split_idx += 1                                                               
            index_edge(split_idx, split_name+'_B')                                                             
            
    # the all-important commit call
    # osmdbs don't have a commit function, so must use the connection member directly
    osmdb.conn.commit()
    c.close()
            
if __name__=='__main__':
    main()