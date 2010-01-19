from graphserver.core import Graph, State, Street
gg = Graph()
gg.add_vertex("A")
gg.add_vertex("B")
gg.add_vertex("C")
gg.add_vertex("D")
gg.add_vertex("E")
gg.add_vertex("F")
gg.add_edge( "A", "B", Street("1", 100) )
gg.add_edge( "A", "B", Street("2", 50) )
gg.add_edge( "B", "C", Street("3", 500) )
gg.add_edge( "B", "C", Street("4", 505) )
gg.add_edge( "C", "D", Street("5", 20) )
gg.add_edge( "D", "E", Street("6", 30) )
gg.add_edge( "D", "E", Street("7", 40) )
gg.add_edge( "E", "F", Street("8", 50) )
gg.add_edge( "E", "F", Street("9", 60) )
gg.add_edge( "B", "E", Street("A", 70) )
spt = gg.shortest_path_tree( "A", "C", State(1,0) )
spt.get_vertex("E").best_state.narrate()

from graphserver.core import Graph, State, WalkOptions
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
wo.transfer_penalty = 60 * 10
wo.walking_reluctance = 2

gdb = GraphDatabase('../../data/trimet-linked-20100117.gsdb')
g = gdb.incarnate()
spt = g.shortest_path_tree( "sta-9677", "sta-13070", State(1, t0), wo )
spt = g.shortest_path_tree( "sta-10120", "osm-40508580", State(1, t0), wo )
spt = g.shortest_path_tree( "sta-3277", "osm-40508580", State(1, t0), wo )
spt = g.shortest_path_tree( "sta-7777", "sta-8340", State(1, t0), wo )
spt = g.shortest_path_tree( "sta-10120", "sta-408", State(1, t0), wo )
spt = g.shortest_path_tree( "sta-9677", "sta-7777", State(1, t0), wo )
spt = g.shortest_path_tree( "sta-7777", None, State(1, t0), wo )
spt = g.shortest_path_tree( "sta-7777", "sta-13197", State(1, t0), wo )
7625 8636 - gives bad results cf trimet

spt.get_vertex("sta-13070").best_state.narrate()
time.ctime(t)


import random, time
s = [v.label for v in g.vertices if v.label[0:4] == 'sta-']
random.shuffle(s)
t = time.time()
iterations = 15 
for i in range(iterations) :
	spt = g.shortest_path_tree( s[i], None, State(1, t0), wo )		

print (time.time() - t) / iterations
