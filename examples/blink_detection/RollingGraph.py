import numpy as np
import cv2
from time import time, sleep

from torch import threshold

class RollingGraph:
    """
        Class designed to draw in an OpenCv window, graph of variables evolving with time.
        The time is not absolute here, but each new call to method 'new_iter' corresponds to a time step.
        'new_iter' takes as argument an array of the current variable values  
    """
    def __init__(self, window_name="Graph", width=640, height=250, step_width=5, y_min=0, y_max=255, colors=[(0,0,255)], thickness=[2], threshold=None, waitKey=True):
        """
            width, height: width and height in pixels of the OpenCv window in which the graph is draw
            step_width: width in pixels on the x-axis between each 2 consecutive points
            y_min, y_max : min and max of the variables
            colors : array of the colors used to draw the variables
            thickness: array of the thickness of the variable curves
            waitKey : boolean. In OpenCv, to display a window, we must call cv2.waitKey(). This call can be done by RollingGraph (if True) or by the program who calls RollingGraph (if False)

        """
        self.window_name = window_name
        self.width = width
        self.height = height
        self.step_width = step_width
        self.y_min = y_min
        self.y_max = y_max
        self.waitKey = waitKey
        assert len(colors) == len(thickness)
        self.colors = colors
        self.thickness = thickness
        self.iter = 0
        self.canvas = np.zeros((height,width,3),dtype=np.uint8)
        self.nb_values = len(colors)
        self.threshold = threshold
        
    def new_iter(self, values):
        # Values = array of values, same length as colors
        assert len(values) == self.nb_values
        self.iter += 1
        if self.iter > 1:
            if self.iter * self.step_width >= self.width:
                self.canvas[:,0:self.step_width,:] = 0
                self.canvas = np.roll(self.canvas, -self.step_width, axis=1)
                self.iter -= 1
            for i in range(self.nb_values):
                cv2.line(self.canvas,((self.iter-1)*self.step_width,int(self.height-(self.prev_values[i]-self.y_min)*self.height/(self.y_max-self.y_min))), (self.iter*self.step_width,int(self.height-(values[i]-self.y_min)*self.height/(self.y_max-self.y_min))), self.colors[i], self.thickness[i]) 
            if self.threshold is not None:
                cv2.line(self.canvas,(0,int(self.height-(self.threshold-self.y_min)*self.height/(self.y_max-self.y_min))), (self.width,int(self.height-(self.threshold-self.y_min)*self.height/(self.y_max-self.y_min))), (255,255,255), 1) 
            cv2.imshow(self.window_name, self.canvas)    
            if self.waitKey: cv2.waitKey(1)
        self.prev_values = values


if __name__ == "__main__":
    from time import sleep
    from math import sin

    rg =RollingGraph("Example", y_min=-1, y_max=1, threshold=0, colors=[(0,255,0), (0,0,255)], thickness=[2, 2])
    
    i=0
    while True:
        rg.new_iter([sin(i/20), sin(i/11)])
        i += 1
        k = cv2.waitKey(10)
        if k == 27:
            break





