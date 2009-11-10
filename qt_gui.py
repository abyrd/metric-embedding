#!/usr/bin/env python2.6
#
#

import pycuda
import pycuda.driver as cuda

import numpy as np
import time

from PyQt4 import QtCore, QtGui

from PIL import Image

import prep
import mds

class LaunchWindow(QtGui.QWidget):
    def __init__(self):
        super(LaunchWindow, self).__init__()
        
        self.mainLayout = QtGui.QGridLayout()
        self.setLayout(self.mainLayout)
        self.lytRow = 0
        
        self.marginSlider = self.createSlider('Map margin %im', 500, 10000, 500, 5000)
        self.dimensionSlider = self.createSlider('Target space dimensionality %i', 1, 8, 1, 4)
        self.listSlider = self.createSlider('%i stations cached locally', 5, 50, 5, 20)
        self.chunkSlider = self.createSlider('Kernel chunk size %i', 50, 1000, 50, 500)
        self.imageSlider = self.createSlider('Images every %i iterations', 1, 20, 1, 1)
        self.iterationSlider = self.createSlider('Stop after %i iterations', 1, 1000, 50, 300)
        self.convergeSlider = self.createSlider('Converge on maximum error %i sec', 1, 1000, 50, 300)
        
        self.launchButton = QtGui.QPushButton('&launch kernel')
        QtCore.QObject.connect(self.launchButton, QtCore.SIGNAL("clicked()"), self.simKernel )
        self.mainLayout.addWidget(self.launchButton, self.lytRow, 0)
        self.lytRow += 1
        
        self.progressBar = QtGui.QProgressBar(self)
        self.mainLayout.addWidget(self.progressBar, self.lytRow, 0)
    
    def simKernel(self) :
        dialog = QtGui.QProgressDialog('test', 'cancel', 0, 100, self)
        self.progressBar.setRange(0, 100)
        for i in range(100) :
            self.progressBar.setValue(i)
            dialog.setValue(i)
            t = time.time()
            while time.time() < t + 0.1:
                if (dialog.wasCanceled()) :
                    return
                app.processEvents()
            
    def createSlider(self, text, min, max, step, default):
        label = QtGui.QLabel()
        slider = QtGui.QSlider(QtCore.Qt.Horizontal)
        QtCore.QObject.connect( slider, QtCore.SIGNAL("valueChanged(int)"), lambda value: label.setText(text % value) )
        slider.setRange(min, max)
        slider.setSingleStep(step)
        slider.setTickInterval(step)
        slider.setValue(default) # emits a signal to change the label
        slider.setTickPosition(QtGui.QSlider.TicksBelow)
        self.mainLayout.addWidget(label, self.lytRow, 0)
        self.lytRow += 1
        self.mainLayout.addWidget(slider, self.lytRow, 0)
        self.lytRow += 1
        return slider

class GraphWidget(QtGui.QLabel):
    def __init__(self):
        super(GraphWidget, self).__init__()        
        self.setWindowTitle('GPU MDS Algorithm')
        self.pixmap = QtGui.QPixmap()
               
    def showImageFile(self, filename) :
        self.pixmap.load(filename)
        self.setPixmap(self.pixmap)
        
if __name__ == '__main__':
    import sys
    app = QtGui.QApplication(sys.argv)
    graphWidget = GraphWidget()
    graphWidget.show()
    launchWindow = LaunchWindow()
    launchWindow.show()
    #graphWidget.showImageFile(QtGui.QFileDialog.getOpenFileName(None, 'Open Image', './data', 'Image Files (*.png *.jpg *.bmp)'))
    sys.exit(app.exec_())
    