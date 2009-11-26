from graphserver.core import Graph, Link, Street, State
from graphserver.ext.osm.osmdb import OSMDB
import time, sys

# Acte I.
# Fuse nodes that are ridiculously close to one another.
def fuse_nodes(db, epsilon = 0.0005) :
        nodes = {}
        for row in db.execute("SELECT DISTINCT start_nd from edges"):
            nodes[str(row[0])] = 0
        for row in db.execute("SELECT DISTINCT end_nd from edges"):
            nodes[str(row[0])] = 0        
        c = db.cursor()
        while True:
            try:
                node0, dummy = nodes.popitem()
            except:
                break
            lat0, lon0 = c.execute("SELECT lat, lon from nodes where id=?", (node0, )).next()
            node, lat, lon, dist = db.nearest_node(lat0, lon0)
            # should really use geoid distance, especially since all the elements are here.
            if dist < epsilon and node != node0 :
                print "Fusing %s -> %s" % (node0, node)
                c.execute("UPDATE edges SET start_nd = ? WHERE start_nd = ?", (node, node0))
                c.execute("UPDATE edges SET end_nd = ? WHERE end_nd = ?", (node, node0))
                db.index.delete(int(node0), (lon0,lat0))
        db.conn.commit
        c.close()
                        
def FindDisjunctGraphs (dbname):
        db = OSMDB(dbname)
        
        #should really be done before simplifying and splitting
        #fuse_nodes(db)
        
        c = db.cursor()
        c.execute("DROP table if exists graph_nodes")
        c.execute("DROP table if exists graph_edges")
        c.execute("CREATE table graph_nodes (graph_num INTEGER, node_id TEXT, WKT_GEOMETRY TEXT)")
        c.execute("CREATE table graph_edges (graph_num INTEGER, edge_id TEXT, WKT_GEOMETRY TEXT)")
        c.execute("CREATE index graph_nodes_id_indx ON graph_nodes(node_id)")
        c.execute("CREATE index graph_edges_id_indx ON graph_edges(edge_id)")
        c.close()
      
        g = Graph()
        t0 = time.time()
        
        vertices = {}
        print "load vertices into memory"
        for row in db.execute("SELECT DISTINCT start_nd from edges"):
            g.add_vertex(str(row[0]))
            vertices[str(row[0])] = 0
            #print str(row[0])

        for row in db.execute("SELECT DISTINCT end_nd from edges"):
            g.add_vertex(str(row[0]))
            vertices[str(row[0])] = 0

        #k = vertices.keys()
        #k.sort()
        #print k, len(k)
        
        print "load edges into memory"
        for start_nd, end_nd in db.execute("SELECT start_nd, end_nd from edges"):
            g.add_edge(start_nd, end_nd, Link())
            g.add_edge(end_nd, start_nd, Link())
            #print start_nd, end_nd
            
        db.conn.commit()
        
        t1 = time.time()
        print "populating graph took: %f"%(t1-t0)
        t0 = t1
        
        print len(vertices)
        iteration = 1
        c = db.cursor()
        while True:
            #c.execute("SELECT id from nodes where id not in (SELECT node_id from graph_nodes) LIMIT 1")
            try:
                vertex, dummy = vertices.popitem()
                #print vertex
            except:
                break
            spt = g.shortest_path_tree(vertex, None, State(1,0))
            print spt.size
            for v in spt.vertices:
                lat, lon = c.execute("SELECT lat, lon from nodes where id=?", (v.label, )).next()
                c.execute("INSERT into graph_nodes VALUES (?, ?, ?)", (iteration, v.label, "POINT(%f %f)" % (lon, lat)))
                for e in v.outgoing: # this gives a wierd maze graph, should do for all edges outside loop.
                    lat1, lon1 = c.execute("SELECT lat, lon from nodes where id=?", (e.from_v.label, )).next()
                    lat2, lon2 = c.execute("SELECT lat, lon from nodes where id=?", (e.to_v.label, )).next()
                    c.execute("INSERT into graph_edges VALUES (?, ?, ?)", 
                        (iteration, e.from_v.label + '->' + e.to_v.label, "LINESTRING(%f %f, %f %f)" % (lon1, lat1, lon2, lat2)))
                #print v.label
                vertices.pop(v.label, None)
                g.remove_vertex(v.label, True, True)
                #print v.label
            spt.destroy()
            
            t1 = time.time()
            print "pass %s took: %f nvertices %d"%(iteration, t1-t0, len(vertices))
            t0 = t1
            iteration += 1
        c.close()
        
        db.conn.commit()
        g.destroy()
        # audit
        for gnum, count in db.execute("SELECT graph_num, count(*) FROM graph_nodes GROUP BY graph_num"):
            print "FOUND: %s=%s" % (gnum, count)
        
if __name__ == '__main__' :
    FindDisjunctGraphs(sys.argv[1])
