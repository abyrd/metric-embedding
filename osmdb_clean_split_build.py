#!/usr/bin/env python2.6
#

from graphserver.core import Graph
from graphserver.ext.gtfs.gtfsdb import GTFSDatabase
from graphserver.ext.osm.osmdb import OSMDB, Node, cons
from graphserver.ext.osm.osmfilters import StitchDisjunctGraphs, PurgeDisjunctGraphsFilter

from graphserver.compiler.gdb_import_gtfs import gdb_load_gtfsdb_to_boardalight
from graphserver.compiler.gdb_import_osm  import gdb_import_osm

from geotools import geoid_dist, angular_dist

import sys, time
import ligeos as lg
from optparse import OptionParser
from rtree import Rtree


def station_split(osmdb, gtfsdb, split_threshold = 50, range = 0.005): # meters (max distance to link node), degrees (intersection box size)
    split_idx = 1000000000 # base node/edge number. above usual OSM ids, but still leaves half of 32bit int range.
    
    n_stops = gtfsdb.count_stops()
    n_edges = osmdb.count_edges()
    #edge_index = Rtree(osmdb_filename+'-edges')
    edge_index = Rtree()
        
    def index_subedge(rid, pair, edge_id) :
        ((s_lon, s_lat), (e_lon, e_lat)) = pair
        box = (min(s_lon, e_lon), min(s_lat, e_lat), max(s_lon, e_lon), max(s_lat, e_lat))
        # Need our own index number to remove edges as we go along.
        # This index could ideally be generated automatically, with collision detection.
        edge_index.add( rid, box, obj = (edge_id, pair) )
        
    new_nodes = {} # for keeping track of where new nodes are added
    print 'indexing OSM sub-way sub-edges into rtree' 
    progress = 0
    for edge_id, way_id, from_nd, to_nd, dist, coords, tags in osmdb.edges() :
        new_nodes [edge_id] = [] # an empty placeholder list to be filled in when new nodes are created along this edge
        for subedge_number, pair in enumerate(cons(coords)) :
            # make id the subedge number along the parent edge 
            index_subedge(subedge_number, pair, edge_id) # take first element because db response is a tuple
        if progress % 1000 == 0 : 
            sys.stdout.write('\rEdge %d/%d' % (progress, n_edges))
            sys.stdout.flush()
        progress += 1
    print
    
    def nearest_point_on_subedge((subedge_rid, (edge_id, pair), subedge_bbox), stop_lat, stop_lon) :
        # print "finding nearest point on edge %s (rtree id %d)" % (edge_id, edge_rid)
        ((s_lon, s_lat), (e_lon, e_lat)) = pair
        # print (s_lat, s_lon, e_lat, e_lon, stop_lat, stop_lon)
        stop   = lg.GPoint(stop_lon, stop_lat)
        ls     = lg.LineString([(s_lon, s_lat), (e_lon, e_lat)], geographic=True)
        dist   = ls.distance_pt(stop)
        pt     = ls.closest_pt(stop)
        loc    = ls.locate_point(pt)
        return subedge_rid, edge_id, pair, subedge_bbox, dist, pt, loc

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
        candidates = [nearest_point_on_subedge(rtree_item, stop_lat, stop_lon) for rtree_item in rtree_items]
        subedge_rid, edge_id, pair, subedge_bbox, dist, pt, loc = min(candidates, key=lambda x:x[4])
        way_id, start_nd, end_nd, length = osmdb.execute( "SELECT parent_id, start_nd, end_nd, dist FROM edges WHERE id = ?", (edge_id,) ).next()
        print "Ideal linking location: edge %s (%d meters away)" % (edge_id, dist)
        # 3. Check if closest existing node is an endpoint of the way containing the ideal point
        #    Or if it is a new node on the same way
        #    Also check if the distance is over the split threshold
        new_nodes_on_edge = [x[0] for x in new_nodes[edge_id]]
        if (start_nd != nd_id and end_nd != nd_id and (nd_id not in new_nodes_on_edge)) or nd_dist > split_threshold :
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
            # keep track of this new node so edges can be split later
            new_nodes[edge_id].append((split_idx, subedge_rid, loc)) # associates edge with (new node id, subedge index, location along subedge)
    
    # New nodes have been created as necessary. Delete old edges and add new ones between the new nodes.
    print 'Removing old edges and adding new ones...'        
    progress=0
    for edge_id, node_list in new_nodes.items() :
        progress += 1
        if progress % 100 == 0 : 
            sys.stdout.write('\rEdge %d/%d' % (progress, n_edges))
            sys.stdout.flush()
        if len(node_list) == 0 : continue 
        # get information about the edge to be replaced
        edge_id, way_id, from_nd, to_nd, edge_length, coords, tags = osmdb.edge(edge_id)
        # find distance and length for each sub-edge
        subedges = []
        c_dist = 0
        for (lon1, lat1), (lon2, lat2) in cons(coords) :
            dist = geoid_dist(lat1, lon1, lat2, lon2) 
            subedges.append((c_dist, dist))
            c_dist += dist
        print '\nedge: ', edge_id
        print 'subedges: ', subedges
        print 'new node list: ', node_list
        
        # sort new nodes by subedge index and location within subedge
        node_list.sort( key = lambda x: float(x[1]) + x[2] ) 
        c = osmdb.cursor()
        last_node_id = from_nd
        dist = 0
        subedge_idx = 0
        while len(node_list) > 0 :
            node_id, node_subedge, node_loc = node_list.pop(0) # take from beginning not end of list - otherwise sort is reversed.
            length = subedges[node_subedge][0] + subedges[node_subedge][1] * node_loc - dist
            print 'Adding edge to node %s on subedge %d offset %f (length %f)' % (node_id, node_subedge, node_loc, length)
            dist += length
            c.execute( "INSERT INTO edges VALUES (?, ?, ?, ?, ?, ?)", ('%s-split-%d' % (edge_id, subedge_idx),
                                                                       way_id,
                                                                       last_node_id,
                                                                       node_id,
                                                                       length,
                                                                       '') ) # <-- TODO: preserve coordinate lists for linestring geometries
            subedge_idx += 1
            last_node_id = node_id
            
        # add a final edge to connect to the original edge's endpoint
        c.execute( "INSERT INTO edges VALUES (?, ?, ?, ?, ?, ?)", ('%s-split-%d' % (edge_id, subedge_idx),
                                                                   way_id,
                                                                   last_node_id,
                                                                   to_nd,
                                                                   edge_length - dist,
                                                                   '') )
            
        # remove the old edge from the database
        c.execute("DELETE FROM edges WHERE id = ?", (edge_id,))
    print

    # the all-important commit call (not necessary if it's called at the end of each loop.
    # osmdbs don't have a commit function, so must use the connection member directly
    osmdb.conn.commit()
    c.close()


def main(osm_xml, osm_db, gtfs_db): 
    
    print "CREATING AND POPULATING OSM DATABASE"
    osmdb =  OSMDB( osm_db, overwrite=True )
    osmdb.populate( osm_xml, reporter=sys.stdout )

    print "CLEANING UP OSM DATABASE"
    StitchDisjunctGraphs().run( osmdb )
    osmdb.create_and_populate_edges_table( tolerant=True )
    PurgeDisjunctGraphsFilter().run( osmdb ) #, threshold = 1000 )
    
    print "SPLITTING OSM EDGES FOR BETTER STATION LINKING"
    gtfsdb = GTFSDatabase (gtfs_db)
    station_split (osmdb, gtfsdb)
    
    print "DONE."
    sys.exit(0)    
            

if __name__=='__main__':
    usage = """usage: python build-graph.py <input_osm_xml> <new_osm_database> <input_gtfs_database>"""
    parser = OptionParser(usage=usage)
    
    (options, args) = parser.parse_args()
    
    if len(args) != 3:
        parser.print_help()
        exit(-1)
        
    osm_xml = args[0]
    osm_db  = args[1]
    gtfs_db = args[2]
    
    main(osm_xml, osm_db, gtfs_db)
