#!/usr/bin/env python2.6

import numpy as np
import time

import pycuda.driver as cuda
from PyQt4 import QtCore, QtGui

#import prep
import qt_mds

class LaunchWindow(QtGui.QWidget):
    def __init__(self):
        super(LaunchWindow, self).__init__()

        self.setWindowTitle('GPU MDS Algorithm')
        self.setGeometry(100, 75, 300, 600)
        self.mainLayout = QtGui.QGridLayout()
        self.setLayout(self.mainLayout)

        self.errGraphWidget = GraphWidget('AVERAGE ABSOLUTE ERROR (MINUTES) / WHITE=30')

        self.velGraphWidget = GraphWidget('POINT VELOCITY (SEC/TIMESTEP) / WHITE=60')

        # thread will send signals to its parent, so tell it self instead of app
        self.mdsThread = qt_mds.MDSThread(self)
        self.connect(self.mdsThread, QtCore.SIGNAL("finished()"), self.resetUI)
        self.connect(self.mdsThread, QtCore.SIGNAL("terminated()"), self.resetUI)
        self.connect(self.mdsThread, QtCore.SIGNAL("outputProgress(int, int, int, float, float)"), self.updateProgress)
        #self.connect(self.mdsThread, QtCore.SIGNAL("outputImage(QString)"), self.displayGraph)
        self.connect(self.mdsThread, QtCore.SIGNAL("outputImage(QImage, QImage)"), self.showImages)
        
        self.lytRow = 0   
        self.filenameButton = QtGui.QPushButton('Choose File')
        QtCore.QObject.connect(self.filenameButton, QtCore.SIGNAL("clicked()"), self.chooseFile)
        self.mainLayout.addWidget(self.filenameButton, self.lytRow, 0)
        
        self.filenameLabel = QtGui.QLabel()
        self.filenameLabel.setText('data/od_matrix_BART.npz')
        self.mainLayout.addWidget(self.filenameLabel, self.lytRow, 1, 1, 2)
          
        self.lytRow += 1   
        self.marginSlider = self.createSlider('Map margin %im', 500, 10000, 500, 5000)
        self.dimensionSlider = self.createSlider('Target space dimensionality %i', 1, 8, 1, 4)
        self.listSlider = self.createSlider('%i stations cached locally', 5, 50, 5, 20)
        self.chunkSlider = self.createSlider('Kernel chunk size %i', 50, 1000, 50, 500)
        self.imageSlider = self.createSlider('Images every %i iterations', 1, 20, 1, 1)
        self.iterationSlider = self.createSlider('Stop after %i iterations', 1, 1000, 50, 300)
        self.convergeSlider = self.createSlider('Converge on maximum error %i sec', 1, 1000, 50, 300)
        
        self.stopButton = QtGui.QPushButton('&Stop')
        QtCore.QObject.connect(self.stopButton, QtCore.SIGNAL("clicked()"), self.stopKernel)
        self.mainLayout.addWidget(self.stopButton, self.lytRow, 0)

        self.launchButton = QtGui.QPushButton('&Launch MDS Kernel')
        QtCore.QObject.connect(self.launchButton, QtCore.SIGNAL("clicked()"), self.launchKernel)
        self.mainLayout.addWidget(self.launchButton, self.lytRow, 1, 1, 2)
            
        self.lytRow += 1        
        self.progressLabel = QtGui.QLabel(self)
        self.mainLayout.addWidget(self.progressLabel, self.lytRow, 0, 1, 3)

        self.lytRow += 1        
        self.progressBar = QtGui.QProgressBar(self)
        self.mainLayout.addWidget(self.progressBar, self.lytRow, 0, 1, 3)

    def createSlider(self, text, min, max, step, default):
        label = QtGui.QLabel()
        slider = QtGui.QSlider(QtCore.Qt.Horizontal)
        QtCore.QObject.connect( slider, QtCore.SIGNAL("valueChanged(int)"), lambda value: label.setText(text % value))
        slider.setRange(min, max)
        slider.setSingleStep(step)
        slider.setTickInterval(step)
        slider.setValue(default) # emits a signal to change the label
        slider.setTickPosition(QtGui.QSlider.TicksBelow)
        self.mainLayout.addWidget(label, self.lytRow, 0, 1, 3)
        self.lytRow += 1
        self.mainLayout.addWidget(slider, self.lytRow, 0, 1, 3)
        self.lytRow += 1
        return slider
    
    def chooseFile(self) :
        self.filenameLabel.setText(QtGui.QFileDialog.getOpenFileName(self, 'Open OD Matrix', './data', 'NumPy Matrices (*.npz)'))
        # activate self.launchButton 
        
#    def launchKernel(self) :
#        graphWidget = GraphWidget()
#        graphWidget.show()
#        imgName = QtGui.QFileDialog.getOpenFileName(None, 'Open Image', './data', 'Image Files (*.png *.jpg *.bmp)')
#        graphWidget.showImageFile(imgName)
#        dialog = QtGui.QProgressDialog('test', 'cancel', 0, 100, self)
#        self.progressBar.setRange(0, 100)
#        for i in range(100) :
#            self.progressBar.setValue(i)
#            self.progressLabel.setNum(i)
#            dialog.setValue(i)
#            t = time.time()
#            while time.time() < t + 0.1:
#                if (dialog.wasCanceled()) :
#                    return
#                app.processEvents()
#        dialog.hide()
#        graphWidget.hide()
    
    def launchKernel(self) :
        self.launchButton.setEnabled(False)
        self.mdsThread.calculate(
                str(self.filenameLabel.text()), 
                self.dimensionSlider.value(), 
                self.iterationSlider.value(), 
                self.imageSlider.value(), 
                self.chunkSlider.value(), 
                self.listSlider.value(), 
                0 )
    
    def stopKernel(self) :
        self.updateProgress(1, 2, 3, 4.4, 5.5) 
        
    def resetUI (self) :
        self.launchButton.setEnabled(True)
        
    def updateProgress(self, iter, station, station_max, runtime, iter_avg) :
        self.progressLabel.setText('iteration %03i / station %04i of %04i / total runtime %03.1f min / avg pass time %02.1f sec' % (iter, station, station_max, runtime, iter_avg) )
        self.progressBar.setRange(0, station_max)
        self.progressBar.setValue(station)

    def displayGraph(self, filename) :
        self.graphWidget.showImageFile(filename)

    def showImages(self, errImage, velImage) :
        self.errGraphWidget.showImage(errImage)
        self.velGraphWidget.showImage(velImage)

class GraphWidget(QtGui.QLabel):
    def __init__(self, title = 'Give me a name!'):
        super(GraphWidget, self).__init__()        
        self.setWindowTitle(title)
        self.setGeometry(600, 200, 400, 400)
        self.show()
               
    def showImageFile(self, filename) :
        self.setPixmap(QtGui.QPixmap(filename))
        self.resize(self.pixmap().size())

    def showImage(self, image) :
        self.setPixmap(QtGui.QPixmap().fromImage(image))
        self.resize(self.pixmap().size())
        
if __name__ == '__main__':
    import sys
    # initialize cuda here before doing anything, but make contexts in threads
    # this is necessary to do threads without an invalid context crash
    cuda.init()
        
    app = QtGui.QApplication(sys.argv)
    launchWindow = LaunchWindow()
    launchWindow.show()
    sys.exit(app.exec_())
    