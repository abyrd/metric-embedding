#!/usr//bin/env python
#
# Test which connects to Tri Met web services to check results
#

TRIP_TIME  = '2:00PM'
TRIP_DATE  = '12-5-2009'
URL_FORMAT = '/ws/V1/trips/tripplanner/maxIntineraries/1/fromcoord/%s/tocoord/%s/date/%s/time/%s/appId/6AC697CF5EB8719DB6F3AEF0B'

import httplib

for i in range(1) :
    from_latlon = (45.587647, -122.593173) # PDX
    to_latlon   = (45.509700, -122.716290) # ZOO
    conn = httplib.HTTPConnection('developer.trimet.org')
    from_str = '%f,%f' % from_latlon
    to_str   = '%f,%f' % to_latlon
    conn.request("GET", URL_FORMAT % (from_str, to_str, TRIP_DATE, TRIP_TIME) )
    r1 = conn.getresponse()
    print r1.status, r1.reason
    data = r1.read()
    idx0 = data.find('<duration>') + 10
    idx1 = data.find('</duration>')
    duration = int(data[idx0:idx1])
    print duration
    conn.close()
