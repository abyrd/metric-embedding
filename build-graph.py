#!/usr/bin/env python2.6
#
# Given a GTFS file and an OSM file, do all steps to prepare a Graphserver graph.
#

from graphserver.core import Graph
from graphserver.ext.gtfs.gtfsdb import GTFSDatabase
from graphserver.ext.osm.osmdb import OSMDB, Node
from graphserver.ext.osm.osmfilters import StitchDisjunctGraphs, PurgeDisjunctGraphsFilter

from graphserver.compiler.gdb_import_gtfs import gdb_load_gtfsdb_to_boardalight
from graphserver.compiler.gdb_import_osm  import gdb_import_osm

from geotools import geoid_dist, angular_dist

import sys, time
import ligeos as lg
from optparse import OptionParser
from rtree import Rtree


def main(output_basename, osm_filename, gtfs_filename): 
    
    print "CREATING AND POPULATING GTFS DATABASE"
    gtfsdb = GTFSDatabase( output_basename + '.gtfsdb', overwrite=True )
    gtfsdb.load_gtfs( gtfs_filename, reporter=sys.stdout )

    print "CREATING AND POPULATING OSM DATABASE"
    osmdb  = OSMDB( output_basename + '.osmdb', overwrite=True )
    osmdb.populate( osm_filename, reporter=sys.stdout )

    print "CLEANING UP OSM DATABASE"
    StitchDisjunctGraphs().run( osmdb )
    osmdb.create_and_populate_edges_table( tolerant=True )
    PurgeDisjunctGraphsFilter().run( osmdb ) #, threshold = 1000 )
    
    print "SPLITTING OSM EDGES FOR BETTER STATION LINKING"
    from stationsplit_osmdb import station_split
    station_split (osmdb, gtfsdb)
    
    print "IMPORTING GTFS INTO GRAPH DATABASE"
    gdb = GraphDatabase( output_basename + '.gdb', overwrite=True )
    agency_id = gtfsdb.execute('SELECT agency_id FROM agency LIMIT 1').next()[0]
    gdb_load_gtfsdb_to_boardalight(gdb, "0", gtfsdb, agency_id, gdb.get_cursor(), maxtrips=None)
    gdb.commit()

    print "IMPORTING OSM INTO GRAPH DATABASE"
    namespace = "osm"
    slogs = {}
    gdb_import_osm(gdb, osmdb, namespace, slogs);

    print "LINKING GTFS STATIONS TO OSM NODES IN GRAPH DATABASE"
    print "ADDING WKT STRINGS TO GRAPH DATABASE"

    sys.exit(0)    
            
if __name__=='__main__':
    usage = """usage: python build-graph.py <output_basename> <osm_filename> <gtfs_filename>"""
    parser = OptionParser(usage=usage)
    
    (options, args) = parser.parse_args()
    
    if len(args) != 3:
        parser.print_help()
        exit(-1)
        
    output_basename = args[0]
    osm_filename    = args[1]
    gtfs_filename   = args[2]
    
    main(output_basename, osm_filename, gtfs_filename)
