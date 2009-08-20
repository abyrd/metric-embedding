#!/usr/bin/env python2.3
#
#  gs_vis_client.py
#  
#
#  Created by Andrew BYRD on 17/08/09.
#  Copyright (c) 2009 __MyCompanyName__. All rights reserved.
#
from socket import *

s = socket( AF_INET, SOCK_STREAM )
s.connect( ('localhost', 10002) )
data = raw_input('>>' )
s.send( data )
s.close()