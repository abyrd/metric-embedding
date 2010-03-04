#!/usr/bin/env python2.6

from graphserver.ext.gtfs.gtfsdb import GTFSDatabase
from graphserver.ext.osm.osmdb import OSMDB
from graphserver.graphdb import GraphDatabase
from graphserver.core import Link

import sys
from optparse import OptionParser

def main():
    usage = """usage: python wktize_gdb.py <graphdb_filename> <osmdb_filename> <gtfsdb_filename>"""
    parser = OptionParser(usage=usage)
    
    (options, args) = parser.parse_args()
    
    if len(args) != 3:
        parser.print_help()
        exit(-1)
        
    graphdb_filename = args[0]
    osmdb_filename   = args[1]
    gtfsdb_filename  = args[2]
    
    gtfsdb = GTFSDatabase( gtfsdb_filename )
    osmdb = OSMDB( osmdb_filename )
    gdb = GraphDatabase( graphdb_filename )
    
    def vertex_interesting(vlabel) :
        return vlabel[0:4] == 'sta-' or vlabel[0:4] == 'osm-'  
    
    def vertex_lookup(vlabel) :
        if vlabel[0:4] == 'sta-' :
            id, name, lat, lon = gtfsdb.stop(vlabel[4:])
            vclass = 'GTFS Stop'
        elif vlabel[0:4] == 'osm-' :
            id, tags, lat, lon, endnode_refs = osmdb.node(vlabel[4:])
            vclass = 'OSM Node'
        else : 
            lat = None
            lon = None
            vclass = None
        return vclass, lat, lon

    c = gdb.get_cursor()
    c.execute( "CREATE TABLE IF NOT EXISTS geom_vertices (label TEXT UNIQUE ON CONFLICT IGNORE, class TEXT, WKT_GEOMETRY TEXT)" )
    c.execute( "CREATE TABLE IF NOT EXISTS geom_edges (class TEXT, WKT_GEOMETRY TEXT)" )
    gdb.commit()

    num_vertices = gdb.num_vertices()
    curr_vertex = 0
    for vlabel in gdb.all_vertex_labels() :
        curr_vertex += 1
        if curr_vertex % 1000 == 0 : 
            sys.stdout.write( '\rVertex %i/%i' % (curr_vertex, num_vertices) )
            sys.stdout.flush()
        if not vertex_interesting(vlabel) : continue
        vclass, lat, lon = vertex_lookup(vlabel)
        c.execute("INSERT INTO geom_vertices VALUES (?, ?, ?)", (vlabel, vclass, "POINT(%s %s)" % (lon, lat)))
    gdb.commit()
        
    print ' '
    num_edges = gdb.num_edges()
    curr_edge = 0
    edges = gdb.execute( "SELECT vertex1, vertex2, edgetype, edgestate FROM edges" )
    for vertex1, vertex2, edgetype, edgestate in edges :
        curr_edge += 1
        if curr_edge % 1000 == 0 : 
            sys.stdout.write( '\rEdge %i/%i' % (curr_edge, num_edges) )
            sys.stdout.flush()
        if not (vertex_interesting(vertex1) and vertex_interesting(vertex2)) : continue
        vclass1, lat1, lon1 = vertex_lookup(vertex1)
        vclass2, lat2, lon2 = vertex_lookup(vertex2)
        c.execute("INSERT INTO geom_edges VALUES (?, ?)", (edgetype, "LINESTRING(%s %s, %s %s)" % (lon1, lat1, lon2, lat2)))
    gdb.commit()

    print '\nIndexing...'
    gdb.execute( "CREATE INDEX IF NOT EXISTS geom_vertices_label ON geom_vertices (label)" )
    gdb.execute( "CREATE INDEX IF NOT EXISTS geom_vertices_class ON geom_vertices (class)" )
    gdb.execute( "CREATE INDEX IF NOT EXISTS geom_edges_class ON geom_edges (class)" )
    
if __name__=='__main__':
    main()