# Adapted from https://github.com/luxonis/depthai-experiments/blob/master/gen2-calc-spatials-on-host/calc.py
import math
import numpy as np
import depthai as dai

# Values
DELTA = 5
THRESH_LOW = 200 # 20cm
THRESH_HIGH = 30000 # 30m

class HostSpatialCalc:
    # We need device object to get calibration data
    def __init__(self, device, delta=DELTA, thresh_low=THRESH_LOW, thresh_high=THRESH_HIGH):
        self.THRESH_LOW = thresh_low
        self.THRESH_HIGH = thresh_high
        self.DELTA = delta
        calibData = device.readCalibration()
        # Required information for calculating spatial coordinates on the host
        # self.monoHFOV = np.deg2rad(calibData.getFov(dai.CameraBoardSocket.LEFT))
        # the FOV of color camera is measured for the full resolution, 4056x3040, 
        # so would need to scale back accordingly if 4k/1080p is selected.
        # 4k is 3840x2160 central crop of 4056x3040, so there is a HFOV loss,  
        # so HFOV will be scaled back by 3840/4056. 1080p has the same FOV as 4k (it's 4k downscaled/binned).
        self.HFOV = np.deg2rad(calibData.getFov(dai.CameraBoardSocket.RGB)) * 3840 / 4056


    def setLowerThreshold(self, threshold_low):
        self.THRESH_LOW = threshold_low
    def setUpperThreshold(self, threshold_low):
        self.THRESH_HIGH = threshold_low
    def setDeltaRoi(self, delta):
        self.DELTA = delta

    def _check_input(self, roi, frame): # Check if input is ROI or point. If point, convert to ROI
        if len(roi) == 4: return roi
        if len(roi) != 2: raise ValueError("You have to pass either ROI (4 values) or point (2 values)!")
        # Limit the point so ROI won't be outside the frame
        x = min(max(roi[0], self.DELTA), frame.shape[1] - self.DELTA)
        y = min(max(roi[1], self.DELTA), frame.shape[0] - self.DELTA)
        return (x-self.DELTA,y-self.DELTA,x+self.DELTA,y+self.DELTA)

    def _calc_angle(self, frame, offset):
        return math.atan(math.tan(self.HFOV / 2.0) * offset / (frame.shape[1] / 2.0))

    # roi has to be list of ints
    def get_xyz(self, depthFrame, roi, averaging_method=np.mean):
        roi = self._check_input(roi, depthFrame) # If point was passed, convert it to ROI
        xmin, ymin, xmax, ymax = roi
        # Calculate the average depth in the ROI.
        depthROI = depthFrame[ymin:ymax, xmin:xmax]
        inRange = (self.THRESH_LOW <= depthROI) & (depthROI <= self.THRESH_HIGH)

        averageDepth = averaging_method(depthROI[inRange])

        centroid = { # Get centroid of the ROI
            'x': int((xmax + xmin) / 2),
            'y': int((ymax + ymin) / 2)
        }

        midW = int(depthFrame.shape[1] / 2) # middle of the depth img width
        midH = int(depthFrame.shape[0] / 2) # middle of the depth img height
        bb_x_pos = centroid['x'] - midW
        bb_y_pos = centroid['y'] - midH

        angle_x = self._calc_angle(depthFrame, bb_x_pos)
        angle_y = self._calc_angle(depthFrame, bb_y_pos)

        spatials = [averageDepth * math.tan(angle_x), -averageDepth * math.tan(angle_y), averageDepth]
        return spatials, roi