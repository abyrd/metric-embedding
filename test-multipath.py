#!/usr/bin/env python

from graphserver.core import Graph, State, Street, Link, WalkOptions
from graphserver.graphdb import GraphDatabase
import time, os

os.environ['TZ'] = 'US/Pacific'
time.tzset()
t0s = "Mon Jan 20 08:50:00 2010"
t0t = time.strptime(t0s)
d0s = time.strftime('%a %b %d %Y', t0t)
t0  = time.mktime(t0t)
print 'search date: ', d0s
print 'search time: ', time.ctime(t0), t0

wo = WalkOptions() 
wo.max_walk = 1600 
wo.walking_overage = 0.1
wo.walking_speed = 0.8 # trimet uses 0.03 miles / 1 minute
wo.transfer_penalty = 600
wo.walking_reluctance = 2

g = Graph()

# intersections in a grid
for r in range(10):
    for c in range(10):
        g.add_vertex( "%d-%d" % (r, c) )

# streets connecting intersections in a grid
for r in range(9):
    for c in range(9):
        g.add_edge( "%d-%d" % (r, c),   "%d-%d" % (r, c+1), Street("R%dA" % r, 500) )
        g.add_edge( "%d-%d" % (r, c),   "%d-%d" % (r+1, c), Street("C%dA" % c, 400) )
        g.add_edge( "%d-%d" % (r, c+1), "%d-%d" % (r, c),   Street("R%dB" % r, 500) )
        g.add_edge( "%d-%d" % (r+1, c), "%d-%d" % (r, c),   Street("C%dB" % c, 400) )

# a train station at one extremity of the grid
g.add_vertex( "sta-tlocal" )
g.add_edge( "9-9", "sta-tlocal", Link() )
g.add_edge( "sta-tlocal", "9-9", Link() )

# a train station very far away
g.add_vertex( "DistantStation" )

# train service
g.add_vertex( "psv-T-1" )
g.add_vertex( "psv-T-2" )
c = Crossing()
c.
g.add_edge( "psv-T-1", "psv-T-2", c )


# bus stops on the grid
g.add_vertex( "sta-B-1" )
g.add_edge( "1-1", "sta-B-1", Link() )
g.add_edge( "sta-B-1", "1-1", Link() )
g.add_vertex( "sta-B-2" )
g.add_edge( "9-1", "sta-B-2", Link() )
g.add_edge( "sta-B-2", "9-1", Link() )
g.add_vertex( "sta-B-3" )
g.add_edge( "9-8", "sta-B-3", Link() )
g.add_edge( "sta-B-3", "9-8", Link() )

# bus service on the grid
g.add_vertex( "psv-B-1" )
g.add_vertex( "psv-B-2" )
g.add_vertex( "psv-B-3" )

spt = gg.shortest_path_tree( "0-0", "DistantStation", State(1, t0) )
