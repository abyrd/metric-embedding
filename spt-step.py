#!/usr/bin/env python

import random
import httplib
import time, os, sys

from graphserver.ext.gtfs.gtfsdb import GTFSDatabase
from graphserver.graphdb         import GraphDatabase
from graphserver.core            import Graph, Street, State, WalkOptions

os.environ['TZ'] = 'US/Pacific'
time.tzset()
t0s = "Tue Mar 09 08:50:00 2010"
t0t = time.strptime(t0s)
d0s = time.strftime('%a %b %d %Y', t0t)
t0  = time.mktime(t0t)
print d0s
print time.ctime(t0), t0

gtfsdb = GTFSDatabase  ('/home/andrew/data/pdx/trimet-2010-02-28.gtfsdb')
gdb    = GraphDatabase ('/home/andrew/data/pdx/trimet.gsdb'  )
g      = gdb.incarnate ()

wo = WalkOptions() 
wo.max_walk = 1600 
wo.walking_overage = 0.0
wo.walking_speed = 1.0 # trimet uses 0.03 miles / 1 minute - but it uses straight line distance as well
wo.transfer_penalty = 60 * 10
wo.walking_reluctance = 2
wo.max_transfers = 5
wo.transfer_slack = 60 * 4

while(True) :
    input = raw_input('o d t / wo > ').split(' ')
    if len(input) == 0 : sys.exit(0)
    elif input[0] == 'wo' :
        try :
            wo.max_walk = int(raw_input('wo.max_walk = '))
            wo.walking_overage = float(raw_input('wo.walking_overage = '))
            wo.walking_speed = float(raw_input('wo.walking_speed = '))
            wo.transfer_penalty = int(raw_input('wo.transfer_penalty = '))
            wo.walking_reluctance = float(raw_input('wo.walking_reluctance = '))
        except :
            print 'invalid input.'
    else:
        try:
            o,d,t = input
        except :
            print 'invalid input.'
            continue
            
    vo  = 'sta-' + o
    vd  = 'sta-' + d
    t0t = time.strptime('%s %s:00' % (d0s, t), '%a %b %d %Y %H:%M:%S')
    t0  = time.mktime(t0t)
    print "origin:      ", o, vo
    print "destination: ", d, vd
    print "time:        ", time.ctime(t0), t0
    
    spt = g.shortest_path_tree( vo, vd, State(1, t0), wo )
    try:
        vp = spt.get_vertex( 'sta-' + d ).best_state
    except:
        print 'Graphserver search failed.'
        continue
        
    vp.narrate()
    print "\nwalked %i meters, %i vehicles." % (vp.dist_walked, vp.num_transfers)
    tm = (vp.time - t0 - 0) / 60.0 + 5
    
    URL_FORMAT = '/ws/V1/trips/tripplanner/maxIntineraries/1/fromPlace/%s/toPlace/%s/date/%s/time/%s/walk/1/appId/6AC697CF5EB8719DB6F3AEF0B'
    url_time = time.strftime('%I:%M%p',  t0t) # '08:53AM'
    url_date = time.strftime('%m-%d-%Y', t0t) # '11-30-2009'
    url = URL_FORMAT % (o, d, url_date, url_time)
    print url
    conn = httplib.HTTPConnection('developer.trimet.org')
    conn.request("GET", url)
    r1 = conn.getresponse()
    print r1.status, r1.reason
    data = r1.read()
    #print data.replace('>', '>\n')
    idx  = data.find('response success') + 18
    if data[idx] == 't' :
        idx0 = data.find('<startTime>')
        idx1 = data.find('<duration>')
        print data[idx0:idx1]
    else :
        print 'Web API search failed.', data

