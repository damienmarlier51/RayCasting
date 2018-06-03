import numpy as np
import pandas as pd
import matplotlib
import matplotlib.cm as cm
import matplotlib.colors as cl
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import itertools
import time
import sys
import tensorflow as tf

class rayCasting:

    df = None
    column_1 = None
    column_2 = None
    linepoints = []
    polygon_idx = 0
    cropped_dfs = []

    @staticmethod
    def set_dataframe(df):
        rayCasting.df = df

    @staticmethod
    def set_column_1(column_1):
        rayCasting.column_1 = column_1

    @staticmethod
    def set_column_2(column_2):
        rayCasting.column_2 = column_2

    @staticmethod
    def onclick(event):

        cropped_dfs = []
        df = rayCasting.df
        column_1 = rayCasting.column_1
        column_2 = rayCasting.column_2
        linepoints = rayCasting.linepoints

        # right click
        if event.button == 3:
            x = event.xdata
            y = event.ydata
            linepoints.append([x,y])
            plt.plot([x[0] for x in linepoints], [x[1] for x in linepoints],color='red')

        # left click
        if event.button == 1 and len(linepoints)>=3:

            linepoints.append(linepoints[0])

            plt.plot([x[0] for x in linepoints], [x[1] for x in linepoints],color='red')
            
            min_x = min([x[0] for x in linepoints])
            max_x = max([x[0] for x in linepoints])
            
            min_y = min([x[1] for x in linepoints])
            max_y = max([x[1] for x in linepoints])

            bounded_df = df[(df[column_1] > min_x) & (df[column_1] < max_x) & (df[column_2] > min_y) & (df[column_2] < max_y)]
            bounded_df.reset_index(inplace=True, drop=True)
            bounded_df_idxs = bounded_df.index.values

            idxs = rayCasting.get_points_within_polygons(bounded_df[[column_1,column_2]].values,linepoints)

            in_polygon_df = bounded_df.ix[bounded_df_idxs[idxs],:]
            rayCasting.cropped_dfs.append(in_polygon_df)

            plt.scatter(in_polygon_df[column_1].values,in_polygon_df[column_2].values,color='green',s=1)

            linepoints = []
            rayCasting.polygon_idx += 1

        rayCasting.linepoints = linepoints
        
        plt.draw()

    @staticmethod
    def build_tensor_graph(x1_,a_,ab_,eps_,device='/cpu:0'):

        with tf.device(device):

            x, y = tf.split(x1_,2,1)

            out = tf.subtract(y,a_[1])
            out = tf.subtract(ab_[1],out)
            out = tf.divide(out,tf.add(ab_[1],eps_))
            condition_1 = tf.floor(out)
            
            out_1 = tf.multiply(ab_[1],tf.subtract(x,a_[0]))
            out_2 = tf.multiply(ab_[0],tf.subtract(y,a_[1]))
            out = tf.subtract(out_1,out_2)
            condition_2 = tf.sign(out)

            zero_condition_1 = tf.equal(condition_1,0)
            zero_condition_2 = tf.equal(condition_2,-1)

            is_on_left = tf.logical_and(zero_condition_1,zero_condition_2)

        return is_on_left

    @staticmethod
    def get_points_within_polygons(points,linepoints,device='/cpu:0'):

        type_='float32'
        x1_ = tf.placeholder(type_, shape=points.shape)

        #Build graph for each segment
        combined_left_points = []
        for i,vertex in enumerate(linepoints):
            if i == len(linepoints)-1:
                left_points = rayCasting.get_graph_on_left_of_segment(x1_,[linepoints[-1],linepoints[0]],type_,device)
            else:
                left_points = rayCasting.get_graph_on_left_of_segment(x1_,[linepoints[i],linepoints[i+1]],type_,device)
            combined_left_points.append(left_points)
        
        out = tf.concat(combined_left_points,1)
        out = tf.cast(out, tf.int32)
        out = tf.reduce_sum(out, 1)
        out = tf.floormod(out,2)
        out = tf.where(tf.equal(out,1))

        sess = tf.Session()
        output = sess.run(out, feed_dict={x1_:points})

        return [x[0] for x in output]

    @staticmethod
    def get_graph_on_left_of_segment(x1_,segment,type_='float32',device='/cpu:0'):

        #reorder segement
        a = segment[0]
        b = segment[1]

        if a[1] > b[1]:
            a,b = b,a
     
        eps = 0.000001   

        x_ab = b[0]-a[0]
        y_ab = b[1]-a[1]
            
        x_a_ = tf.constant(a[0],dtype=type_)
        x_ab_ = tf.constant(x_ab,dtype=type_)

        y_a_ = tf.constant(a[1],dtype=type_)
        y_ab_ = tf.constant(y_ab,dtype=type_)

        eps_ = tf.constant(eps,dtype=type_)

        graph_ = rayCasting.build_tensor_graph(x1_,[x_a_,y_a_],[x_ab_,y_ab_],eps_,device)

        return graph_

    @staticmethod
    def run_selector(df,column_1,column_2):

        rayCasting.set_dataframe(df)
        rayCasting.set_column_1(column_1)
        rayCasting.set_column_2(column_2)

        ax  = plt.gca()
        fig = plt.gcf()

        plt.scatter(df[column_1], df[column_2], s=1)
        cid = fig.canvas.mpl_connect('button_press_event', rayCasting.onclick)

        plt.show()
        plt.draw()

        return rayCasting.cropped_dfs

if __name__ == "__main__":
    rayCasting.dummy_example()



    
