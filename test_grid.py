#!/usr/bin/env python
from graphserver.ext.osm.osmdb import OSMDB
o = OSMDB('test.osmdb')
o.make_grid()
