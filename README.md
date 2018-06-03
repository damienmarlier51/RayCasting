# RayCasting

Fast Ray casting using tensorflow. It can be run on GPU for large set of points.
The purpose of this code is to find the points within a polygon.

## Example 1

Given a list of points [[x1,y1],[x2,y2]...[xn,yn]] and a polygon [[Px1,Py1],[Px2,Py2]...[Pxm,Pym]], the example below will return the points within the polygon:

```
import numpy as np
import pandas as pd
import math
from ray_casting import rayCasting
import matplotlib.pylab as plt

nb_points = 10000
points = np.random.rand(nb_points,2)*2-1
df = pd.DataFrame(data=points, columns=["x","y"])

column_1 = "x"
column_2 = "y"

linepoints = [[0.5,0.5],[0.5,0],[0,0],[0,0.5]]
idxs = rayCasting.get_points_within_polygons(df[[column_1,column_2]].values,linepoints)
within_polygon_df = df.iloc[idxs]
```

## Example 2

The following example lets you build your own polygon:

```
import numpy as np
import pandas as pd
import math
from ray_casting import rayCasting
import matplotlib.pylab as plt

nb_points = 10000
points = np.random.rand(nb_points,2)
df = pd.DataFrame(data=points, columns=["x","y"])

column_1 = "x"
column_2 = "y"

#Draw polygon on matplotlib
within_polygon_df = rayCasting.run_selector(df, column_1, column_2)[0]
```

When running the previous example, on the matplotlib figure, press right click to add a point to the polygon. When more than 3 points drawn, press left click to close the polygon. Below is the expected output.

<p float="center">
	<img src="https://github.com/damienmarlier51/RayCasting/blob/master/pictures/Figure_1.png" width="33%"/>
	<img src="https://github.com/damienmarlier51/RayCasting/blob/master/pictures/Figure_2.png" width="33%"/>
	<img src="https://github.com/damienmarlier51/RayCasting/blob/master/pictures/Figure_3.png" width="33%"/>
</p>


## Authors

* **Damien Marlier**