#!/usr/bin/env python2.6

import numpy as np
import time

import pycuda.driver as cuda
from PyQt4 import QtCore, QtGui

import qt_prep
import qt_mds

class LaunchWindow(QtGui.QWidget):
    def __init__(self):
        super(LaunchWindow, self).__init__()

        self.setWindowTitle('GPU MDS Algorithm')
        self.setGeometry(100, 75, 300, 600)
        self.mainLayout = QtGui.QGridLayout()
        self.setLayout(self.mainLayout)

        self.makeMatrixWidget = MakeMatrixWidget()
        self.errGraphWidget = GraphWidget('RMS ERROR (RED = 30 MINUTES)')
        self.velGraphWidget = GraphWidget('POINT VELOCITY (RED = 60 SEC/TIMESTEP)')

        self.lytRow = 0   
        self.filenameButton = QtGui.QPushButton('Load OD matrix')
        QtCore.QObject.connect(self.filenameButton, QtCore.SIGNAL("clicked()"), self.chooseFile)
        self.mainLayout.addWidget(self.filenameButton, self.lytRow, 0)
        self.makeMatrixButton = QtGui.QPushButton('Make OD matrix')
        QtCore.QObject.connect(self.makeMatrixButton, QtCore.SIGNAL("clicked()"), self.makeMatrix)
        self.mainLayout.addWidget(self.makeMatrixButton, self.lytRow, 1)
        self.processRawButton = QtGui.QPushButton('Process raw GTFS')
        QtCore.QObject.connect(self.processRawButton, QtCore.SIGNAL("clicked()"), self.processRaw)
        self.mainLayout.addWidget(self.processRawButton, self.lytRow, 2)

        self.lytRow += 1   
        self.filenameLabel = QtGui.QLabel()
        self.filenameLabel.setText('No OD matrix loaded.')
        self.mainLayout.addWidget(self.filenameLabel, self.lytRow, 0, 1, 3)
        
        self.lytRow += 1   
        self.stationDisplayLabel = QtGui.QLabel()
        self.mainLayout.addWidget(self.stationDisplayLabel, self.lytRow, 0, 1, 3)
        self.stationDisplayLabel.setPixmap(QtGui.QPixmap('fuller.jpg'))
            
        self.lytRow += 1   
        self.marginSlider = self.createSlider('Map margin %im', 500, 10000, 500, 5000)
        self.dimensionSlider = self.createSlider('Target space dimensionality %i', 1, 8, 1, 4)
        self.listSlider = self.createSlider('%i stations cached locally', 5, 50, 5, 15)
        self.chunkSlider = self.createSlider('Kernel chunk size %i', 50, 1000, 50, 500)
        self.imageSlider = self.createSlider('Images every %i iterations', 1, 20, 1, 1)
        self.iterationSlider = self.createSlider('Stop after %i iterations', 1, 1000, 50, 300)
        self.convergeSlider = self.createSlider('Converge on maximum error %i sec', 1, 1000, 50, 300)
        
        self.stopButton = QtGui.QPushButton('&Stop')
        QtCore.QObject.connect(self.stopButton, QtCore.SIGNAL("clicked()"), self.stopKernel)
        self.mainLayout.addWidget(self.stopButton, self.lytRow, 0)

        self.launchButton = QtGui.QPushButton('&Launch MDS Kernel')
        QtCore.QObject.connect(self.launchButton, QtCore.SIGNAL("clicked()"), self.launchKernel)
        self.launchButton.setEnabled(False)
        self.mainLayout.addWidget(self.launchButton, self.lytRow, 1, 1, 2)
            
        self.lytRow += 1        
        self.progressBar = QtGui.QProgressBar(self)
        self.mainLayout.addWidget(self.progressBar, self.lytRow, 0, 1, 3)

        self.lytRow += 1        
        self.progressLabel = QtGui.QLabel(self)
        self.mainLayout.addWidget(self.progressLabel, self.lytRow, 0, 1, 3)

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
            
    def chooseFile(self, filename = None) :
        self.setEnabled(False)
        if filename == None :
            self.filenameLabel.setText(QtGui.QFileDialog.getOpenFileName(self, 'Open OD Matrix', './data', 'NumPy Matrices (*.npz)'))
        else :
            self.filenameLabel.setText(filename)
        progressDialog = QtGui.QProgressDialog('Loading OD matrix and station coordinates...', 'Cancel', 0, 2)
        progressDialog.show()
        try:
            npz = np.load(str(self.filenameLabel.text()))
            self.station_coords = npz['station_coords']
            self.grid_dim       = npz['grid_dim']
            progressDialog.setValue(1)
            app.processEvents()
            self.matrix         = npz['matrix'] #.astype(np.int32)
            matrixImage = QtGui.QImage(self.grid_dim[0] / 2, self.grid_dim[1] / 2, QtGui.QImage.Format_Indexed8)
            matrixImage.fill(20)
            matrixImage.setColorTable([QtGui.QColor(i, i, i).rgb() for i in range(256)])
#            for coord in self.station_coords :
#                x = coord[0]/2
#                y = coord[1]/2
#                matrixImage.setPixel(x,   y,   255)    
#                matrixImage.setPixel(x,   y+1, 255)    
#                matrixImage.setPixel(x+1, y,   255)    
#                matrixImage.setPixel(x+1, y+1, 255)    
            self.stationDisplayLabel.setPixmap(QtGui.QPixmap().fromImage(matrixImage))
            self.launchButton.setEnabled(True)
        except Exception as ex : QtGui.QMessageBox.critical(self, 'Laod Failed', str(ex))
        progressDialog.hide()
        self.setEnabled(True)

    def makeMatrix(self) :
        self.setEnabled(False)    
        gtfsdb = QtGui.QFileDialog.getOpenFileName(self, 'Open SQLite GTFS', './data', 'SQLite GTFS (*.gtfsdb)')
        gsdb   = QtGui.QFileDialog.getOpenFileName(self, 'Open SQLite Graphserver Graph', './data', 'SQLite Graphserver (*.gsdb)')
        try:
            self.makeMatrixThread = qt_prep.MakeMatrixThread(self)
            self.makeMatrixWidget.show()
            self.connect(self.makeMatrixThread, QtCore.SIGNAL("finished()"), self.resetUI)
            self.connect(self.makeMatrixThread, QtCore.SIGNAL("terminated()"), self.resetUI)
            self.connect(self.makeMatrixThread, QtCore.SIGNAL("say(QString)"), self.makeMatrixWidget.say)
            self.connect(self.makeMatrixThread, QtCore.SIGNAL("display(QImage)"), self.makeMatrixWidget.display)
            self.connect(self.makeMatrixThread, QtCore.SIGNAL("progress(int, int)"), self.makeMatrixWidget.progress)
            self.makeMatrixThread.makeMatrix(gtfsdb, gsdb)
            
        except Exception as ex :
            QtGui.QMessageBox.critical(self, 'Load Failed', str(ex))
            self.setEnabled(True)


    def processRaw(self) :
        pass
            
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
        # thread will send signals to its parent, so set it to self instead of app
        self.mdsThread = qt_mds.MDSThread(self)
        self.connect(self.mdsThread, QtCore.SIGNAL("finished()"), self.resetUI)
        self.connect(self.mdsThread, QtCore.SIGNAL("terminated()"), self.resetUI)
        self.connect(self.mdsThread, QtCore.SIGNAL("outputProgress(int, int, int, float, float)"), self.updateProgress)
        #self.connect(self.mdsThread, QtCore.SIGNAL("outputImage(QString)"), self.displayGraph)
        self.connect(self.mdsThread, QtCore.SIGNAL("outputImage(QImage, QImage)"), self.showImages)
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
        self.makeMatrixWidget.hide()
        self.setEnabled(True)
        self.launchButton.setEnabled(True)
        
    def updateProgress(self, iter, station, station_max, runtime, iter_avg) :
        self.progressLabel.setText('iteration %03i / station %04i of %04i / total runtime %03.1f min / avg pass time %02.1f sec' % (iter, station, station_max, runtime, iter_avg) )
        self.progressBar.setRange(0, station_max)
        self.progressBar.setValue(station)

    def showImages(self, errImage, velImage) :
        self.errGraphWidget.showImage(errImage)
        self.velGraphWidget.showImage(velImage)

class GraphWidget(QtGui.QLabel):
    def __init__(self, title = 'Give me a name!'):
        super(GraphWidget, self).__init__()        
        self.setWindowTitle(title)
        self.setGeometry(600, 200, 400, 400)
               
    def showImage(self, image) :
        self.setPixmap(QtGui.QPixmap().fromImage(image))
        self.resize(self.pixmap().size())
        self.show()

class MakeMatrixWidget(QtGui.QWidget):
    def __init__(self, title = 'Give me a name!'):
        super(MakeMatrixWidget, self).__init__()        
        self.setWindowTitle('Making OD Matrix')
        self.setGeometry(300, 200, 40, 40)
        self.mainLayout = QtGui.QGridLayout()
        self.setLayout(self.mainLayout)
        self.mapLabel = QtGui.QLabel()
        self.mapLabel.setPixmap(QtGui.QPixmap('fuller.jpg'))        
        self.mainLayout.addWidget(self.mapLabel, 0, 0)
        self.progressBar = QtGui.QProgressBar()
        self.mainLayout.addWidget(self.progressBar, 1, 0)
        self.textLabel = QtGui.QLabel() 
        self.mainLayout.addWidget(self.textLabel, 2, 0)
        self.cancelButton = QtGui.QPushButton() 
        self.mainLayout.addWidget(self.textLabel, 3, 0)
        
    def say(self, text) :
        self.textLabel.setText(text)
        
    def display(self, image) :
        self.mapLabel.setPixmap(QtGui.QPixmap().fromImage(image))

    def progress(self, val, max) :
        self.progressBar.setRange(0, max)
        self.progressBar.setValue(val)

if __name__ == '__main__':
    import sys
    # initialize cuda here before doing anything, but make contexts in threads
    # this is necessary to do threads without an invalid context crash (not really...)

    app = QtGui.QApplication(sys.argv)
    launchWindow = LaunchWindow()
    launchWindow.show()
    sys.exit(app.exec_())
    
