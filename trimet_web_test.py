#!/usr/bin/env python
#
# Test which connects to Tri Met web services to check results
# version for working directly with Graphserver SPTs
#
# Please be nice and do not hit Tri-met's web server too fast!

import numpy as np
import random
import time
import httplib
import os

from BeautifulSoup import BeautifulStoneSoup

from graphserver.ext.gtfs.gtfsdb import GTFSDatabase
from graphserver.graphdb         import GraphDatabase
from graphserver.core            import Graph, Street, State, WalkOptions

SAMPLE_SIZE = 200
SHOW_GS_ROUTE = True
os.environ['TZ'] = 'US/Pacific'
time.tzset()
t0s = "Mon May 17 08:50:00 2010"
t0t = time.strptime(t0s)
d0s = time.strftime('%a %b %d %Y', t0t)
t0  = time.mktime(t0t)
print 'search date: ', d0s
print 'search time: ', time.ctime(t0), t0

gtfsdb = GTFSDatabase  ('/Users/andrew/devel/data/trimet.gtfsdb')
gdb    = GraphDatabase ('/Users/andrew/devel/data/trimet.gdb'  )
g      = gdb.incarnate ()

station_labels = [s[0] for s in gtfsdb.stops()]

origins      = station_labels[:]
destinations = station_labels[:]
random.shuffle(origins)
random.shuffle(destinations)
pairs = zip(origins, destinations)[:SAMPLE_SIZE]

wo = WalkOptions() 
wo.max_walk = 2000 
wo.walking_overage = 0.0
wo.walking_speed = 1.0 # trimet uses 0.03 miles / 1 minute - but it uses straight line distance as well
wo.transfer_penalty = 60 * 10
wo.walking_reluctance = 1.5
wo.max_transfers = 5
wo.transfer_slack = 60 * 4

residuals  = []
magnitudes = []
normalize = 0
for o, d in pairs : 
    og = gtfsdb.stop(o)
    dg = gtfsdb.stop(d)
    #print 'Origin (mds, gtfs)'
    #print o, og
    #print 'Destination (mds, gtfs)'
    #print d, dg
    print 'from %s to %s (%f, %f) -> (%f, %f)' % (og[0], dg[0], og[2], og[3], dg[2], dg[3])

    #replace
    spt = g.shortest_path_tree( 'sta-' + o, 'sta-' + d, State(1, t0), wo )
    try:
        vp = spt.get_vertex( 'sta-' + d ).best_state
    except:
        print 'Graphserver search failed.'
        continue
            
    print "gs:   walked %i meters, %i vehicles." % (vp.dist_walked, vp.num_transfers)

    tg = vp.time
    
    URL_FORMAT = '/ws/V1/trips/tripplanner/maxIntineraries/1/fromPlace/%s/toPlace/%s/date/%s/time/%s/walk/1/appId/6AC697CF5EB8719DB6F3AEF0B'
    url_time = time.strftime('%I:%M%p',  t0t) # '08:53AM'
    url_date = time.strftime('%m-%d-%Y', t0t) # '11-30-2009'
    url = URL_FORMAT % (o, d, url_date, url_time)
    conn = httplib.HTTPConnection('developer.trimet.org')
    conn.request("GET", url)
    r1 = conn.getresponse()
    #print r1.status, r1.reason
    data = r1.read()
    conn.close()
    soup = BeautifulStoneSoup(data)        
    if soup.response['success'] == 'true' :
        endtime = soup.itinerary.endtime.string
        tw = time.mktime(time.strptime(d0s + ' ' + endtime, '%a %b %d %Y %I:%M %p'))
        lastleg = soup.itinerary.find(order='end')
        if lastleg is None :
            print "No last leg wot."
            continue
        if lastleg.endtime == None :
            tw += int(lastleg.duration.string) * 60

        print 'gs:   ', time.ctime(tg) 
        print 'web:  ', time.ctime(tw) 
        diff = (tw - tg) / 60.0
        print 'diff: %04.1f min' % diff
        residuals.append(diff)
        magnitudes.append((vp.time - t0) / 60.0)
        if SHOW_GS_ROUTE and abs(diff) > 8 :
            vp.narrate()
    else :
        print 'Web API search failed.' # , data
    # show route if difference is notable
    spt.destroy()
    # maybe use rawinput to wait for enter
    time.sleep(2)

print 'Residuals, magnitudes: ', zip( residuals, magnitudes )
print 'Normalized errors (percent): ', [ e[0] / e[1] * 100 for e in zip( residuals, magnitudes ) ]
print 
print 'RMS error:', np.sqrt( np.sum(np.array(residuals) ** 2.0) / SAMPLE_SIZE )
