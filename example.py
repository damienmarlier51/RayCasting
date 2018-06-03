import numpy as np
import pandas as pd
import math
from ray_casting import rayCasting
import matplotlib.pylab as plt

def dummy_example_1():
    
    nb_points = 10000
    
    points = np.random.rand(nb_points,2)
    df = pd.DataFrame(data=points, columns=["x","y"])

    column_1 = "x"
    column_2 = "y"

    #Draw polygon on matplotlib
    within_polygon_df = rayCasting.run_selector(df, column_1, column_2)[0]

def dummy_example_2():
    
	nb_points = 10000

	points = np.random.rand(nb_points,2)*2-1
	df = pd.DataFrame(data=points, columns=["x","y"])

	column_1 = "x"
	column_2 = "y"

	linepoints = [[0.5,0.5],[0.5,0],[0,0],[0,0.5]]
	idxs = rayCasting.get_points_within_polygons(df[[column_1,column_2]].values,linepoints)
	within_polygon_df = df.iloc[idxs]

	plt.scatter(df[column_1], df[column_2], s=1)
	plt.scatter(within_polygon_df[column_1], within_polygon_df[column_2], s=1)
	plt.show()

if __name__ == "__main__":
	#dummy_example_2()
	dummy_example_1()