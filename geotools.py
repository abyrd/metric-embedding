#
#  geotools.py
#  

from math import sin, cos, tan, atan, degrees, radians, pi, sqrt, atan2, asin, ceil
from graphserver.core import Graph, Street
from graphserver.ext.gtfs.gtfsdb import GTFSDatabase
from numpy import zeros

# based on OpenLayers.Util.distVincenty=function(p1, p2)
def angular_dist(dist_meters, at_lat) :
    """Returns degrees latitude and longitude for a given distance in meters. Result depends on latitude."""
    # first-cut bounding box (in degrees) using spherical law of cosines
    # based on Chris Veness at http://www.movable-type.co.uk/scripts/latlong-db.html (LGPL)
    # warning: will not work in Chukotka Autonomous Okrug and some parts of Fiji!
    EARTH_RADIUS = 6371000.0 # meters. assume spherical geoid.
        
    # degrees of latitude give relatively constant distances
    angular_distance_rad = dist_meters / EARTH_RADIUS
    angular_distance_deg_lat = degrees(angular_distance_rad)

    # compensate for degrees longitude getting smaller with increasing latitude
    angular_distance_deg_lon = abs( degrees( angular_distance_rad / cos( radians(at_lat) ) ) )

    return( (angular_distance_deg_lat, angular_distance_deg_lon) )
    
    
def geoid_dist(lat1, lon1, lat2, lon2) :
    """Returns distance in meters between any two points on earth."""
    
    a = 6378137
    b = 6356752.3142
    f = 1/298.257223563
    
    L = radians( lon2-lon1 )
    U1 = atan( (1-f) * tan( radians(lat1) ) )
    U2 = atan( (1-f) * tan( radians(lat2) ) )
    sinU1 = sin(U1); cosU1 = cos(U1)
    sinU2 = sin(U2); cosU2 = cos(U2)
    lmbda = L; lmbdaP = 2*pi
    
    iterLimit = 20
    
    while( iterLimit > 0 ):
        if abs(lmbda-lmbdaP) < 1E-12:
            break
        
        sinLambda = sin(lmbda); cosLambda = cos(lmbda)
        sinSigma = sqrt((cosU2*sinLambda) * (cosU2*sinLambda) + \
            (cosU1*sinU2-sinU1*cosU2*cosLambda) * (cosU1*sinU2-sinU1*cosU2*cosLambda))
        if sinSigma==0:
            return 0  # co-incident points

        cosSigma = sinU1*sinU2 + cosU1*cosU2*cosLambda
        sigma = atan2(sinSigma, cosSigma)
        alpha = asin(cosU1 * cosU2 * sinLambda / sinSigma)
        cosSqAlpha = cos(alpha) * cos(alpha)
        cos2SigmaM = cosSigma - 2*sinU1*sinU2/cosSqAlpha
        C = f/16*cosSqAlpha*(4+f*(4-3*cosSqAlpha))
        lmbdaP = lmbda;
        lmbda = L + (1-C) * f * sin(alpha) * \
            (sigma + C*sinSigma*(cos2SigmaM+C*cosSigma*(-1+2*cos2SigmaM*cos2SigmaM)))
            
        iterLimit -= 1
            
    if iterLimit==0:
        return None  # formula failed to converge

    uSq = cosSqAlpha * (a*a - b*b) / (b*b);
    A = 1 + uSq/16384*(4096+uSq*(-768+uSq*(320-175*uSq)))
    B = uSq/1024 * (256+uSq*(-128+uSq*(74-47*uSq)))
    deltaSigma = B*sinSigma*(cos2SigmaM+B/4*(cosSigma*(-1+2*cos2SigmaM*cos2SigmaM)-
            B/6*cos2SigmaM*(-3+4*sinSigma*sinSigma)*(-3+4*cos2SigmaM*cos2SigmaM)))
    s = b*A*(sigma-deltaSigma)
    
    return s
    
    
def create_grid ( this, gtfsdb, grid_spacing=100, overhang=2000, link_radius=500, obstruction=1.4, link_grid = True ) :
    """Makes a regular grid of points over a geographic space. Adds them as nodes to this Graph, and optionally links them to stations and to one another. 
<grid_spacing> is in meters. 
<overhang> is how far to go beyond the GTFS stations' extent in meters.
Returns a 2D array (row, column) of tuples (lat, lon, Vertex*) for the grid points. 
Should give constant-area grid cells of 1 ha with default grid spacing of 100m. 
        
However, over the San Francisco BART service area, using this method, a 100 meter grid is 1.4 meters wider at the top than at the bottom.
If longitude distances were not recalculated at each row as a function of latitude, it would be about 336.246 meters too narrow at the top, or off by over 3 grid cells.
Grid cells are on average 4.5 cm too wide and 3.7 cm too short, giving an area of 9967.69 square meters. This is probably because we are using a spherical geoid in the calculations. """
    
    # get extent of stations in GTFS - lon and lat are reversed in result compared to other functions, this should be changed
    min_lon, min_lat, max_lon, max_lat = gtfsdb.extent()
    # print min_lat, min_lon, max_lat, max_lon
    # move bounding box out by <overhang> meters
    delta_lat, delta_lon = angular_dist(overhang, max_lat)
    max_lat += delta_lat
    max_lon += delta_lon
    delta_lat, delta_lon = angular_dist(overhang, min_lat)
    min_lat -= delta_lat
    min_lon -= delta_lon
    # print min_lat, min_lon, max_lat, max_lon

    # get grid dimensions at latitude where it is widest
    # works only in northern hemisphere for the moment
    delta_lat, delta_lon = angular_dist(grid_spacing, min_lat)
    cols = int( ceil( (max_lon - min_lon) / delta_lon ) )
    rows = int( ceil( (max_lat - min_lat) / delta_lat ) )
       
    # iterate up latitudes creating grid points
    grid = zeros( (rows, cols, 2) )
    point_list = []
    curr_lat = min_lat
    print "Creating grid points and linking to stations within %d meters..." % link_radius
    for row in range(rows) :
        if row % 50 == 0 : print "row %i / %i" % (row, rows)
        delta_lat, delta_lon = angular_dist(grid_spacing, curr_lat)
        curr_lon = min_lon
        # print delta_lat, delta_lon
        for col in range(cols) :
            grid_label = 'grid-%s-%s' % (row, col)
            v = this.add_vertex( grid_label )
            for stop_id, stop_name, stop_lat, stop_lon in gtfsdb.nearby_stops( curr_lat, curr_lon, link_radius / obstruction ) :
                dd = obstruction * geoid_dist( curr_lat, curr_lon, stop_lat, stop_lon )
                if dd < link_radius :
                    stop_label = "sta-%s" % stop_id
                    this.add_edge( grid_label, stop_label, Street("walk", dd) )
                    this.add_edge( stop_label, grid_label, Street("walk", dd) )
                    # print "%s to %s onstructed dist %f" % (grid_label, stop_label, dd)
                    point_list.append( (row, col, v) )
            grid[row][col] = (curr_lat, curr_lon)
            curr_lon += delta_lon
        curr_lat += delta_lat
    if link_grid :
        print "Linking grid internally..."
        # try linking only to neighbourhood?
        for row, col, v in point_list :
            v_label = 'grid-%s-%s' % (row, col)
            for row2, col2 in [(row-1, col-1), (row-1, col+1), (row+1, col-1), (row+1, col+1)] :
                v_label2 = 'grid-%s-%s' % (row2, col2)
                try :
                    this.add_edge( v_label, v_label2, Street("walk", grid_spacing * 2) )
                except :
                    # I shall fail on mesh edges
                    pass                
    return grid, point_list
    

        
        
        