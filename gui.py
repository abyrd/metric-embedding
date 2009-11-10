#!/usr/bin/env python2.6
#
#

import pycuda
import pycuda.driver as cuda

from Tkinter import *
from tkFileDialog   import askopenfilename        
from tkFileDialog   import askdirectory        
from tkColorChooser import askcolor               
from tkMessageBox   import askquestion, showerror
from tkSimpleDialog import askfloat

import threading
from PIL import Image, ImageTk

import prep
import mds

class Interface(Tk):
    def __init__(self):
        Tk.__init__(self)
        self.title('GPU MDS Algorithm')
        self.geometry('700x600+100+30')
        
        self.gtfs_mode    = IntVar()
        self.gtfs_filename  = StringVar()
        self.gtfs_filename.set(' ')
        self.map_margin   = IntVar()
        self.images       = IntVar()
        self.images_every = IntVar()
        self.converge     = IntVar()
        self.converge_on  = IntVar()
        self.debug = IntVar()
        
        Radiobutton(self, text="Precomputed OD matrix", variable=self.gtfs_mode, value=0).grid(row=0, column=0, sticky=W)
        Button(self, text="choose", command=self.choose_od).grid(row=0, column=1, sticky=W)
        self.od_directory = StringVar()
        self.od_directory.set('data/od_matrix_BART.npz')
        Label(self, textvariable=self.od_directory).grid(row=0, column=2, sticky=W)

        Radiobutton(self, text="New OD matrix from GTFS", variable=self.gtfs_mode, value=1).grid(row=1, column=0, sticky=W)
        Button(self, text="choose", command=self.choose_gtfs).grid(row=1, column=1, sticky=W)
        Label(self, textvariable=self.gtfs_filename).grid(row=1, column=2, sticky=W)
        
        Label(self, text="Map margin (meters)").grid(row=2, column=0, sticky=W)
        Scale(self, variable=self.map_margin,   
                      from_=0, to=10000,
                      state = 'disabled',
                      tickinterval=2000,
                      resolution=500,
                      showvalue=YES, orient='horizontal').grid(row=2, column=1, columnspan=2, sticky=EW)
        
        
        Label(self, text="Target space dimensionality").grid(row=3, column=0, sticky=W)
        self.dimensions = Scale(self, 
                            from_=2, to=6,
                            tickinterval=1,
                            length=300,
                            showvalue=NO, orient='horizontal')
        self.dimensions.set(4)
        self.dimensions.grid(row=3, column=1, columnspan=2, sticky=W)

        Label(self, text="Per texel station list size").grid(row=4, column=0, sticky=W)
        self.list_size = Scale(self,        
                      from_=1, to=50,
                      tickinterval=10,
                      resolution=5,
                      showvalue=NO, orient='horizontal')
        self.list_size.grid(row=4, column=1, columnspan=2, sticky=EW)
        self.list_size.set(20)

        Label(self, text="Kernel chunk size").grid(row=5, column=0, sticky=W)
        self.kernel_chunk = Scale(self,       
                      from_=100, to=1000,
                      tickinterval=100,
                      resolution=100,
                      showvalue=NO, orient='horizontal')
        self.kernel_chunk.grid(row=5, column=1, columnspan=2, sticky=EW)
        self.kernel_chunk.set(500)
        
        Checkbutton(self, text="Write image every n iterations", variable=self.images).grid(row=6, sticky=W)
        Scale(self, variable=self.images_every,       
                      from_=1, to=20,
                      tickinterval=5,
                      showvalue=NO, orient='horizontal').grid(row=6, column=1, columnspan=2, sticky=EW)

        Radiobutton(self, text="Terminate after N interations", variable=self.converge, value=0).grid(row=7, column=0, sticky=W)
        self.n_iterations = Scale(self,       
                      from_=5, to=1000,
                      tickinterval=200,
                      resolution=50,
                      showvalue=NO, orient='horizontal')
        self.n_iterations.grid(row=7, column=1, columnspan=2, sticky=EW)
        self.n_iterations.set(5)
        
        Radiobutton(self, text="Converge to average error", variable=self.converge, value=1).grid(row=8, sticky=W)
        Scale(self,   variable=self.converge_on,       
                      from_=0, to=1000,
                      tickinterval=200,
                      resolution=50,
                      showvalue=NO, orient='horizontal').grid(row=8, column=1, columnspan=2, sticky=EW)
                      
        Checkbutton(self, text="Produce debug output", variable=self.debug).grid(row=9, sticky=W)

        self.progress = StringVar()
        self.progress.set('BLAH')
        Label(self, textvariable=self.progress).grid(row=98, columnspan=3, sticky=W)

        Button(self, text="[ LAUNCH ]", command=self.onRun).grid(row=99, columnspan=3, sticky=EW)

        # configure grid on container (main window)
        self.columnconfigure(2,weight=1)
        self.rowconfigure(98,weight=1)

        # make another top level window
        image_window = Toplevel()
        image_window.title('Graphical Progress')
        image_window.geometry('600x500+800+40')
        self.canvas = Canvas(image_window, width=600, height=500)
        self.canvas.pack()
        # top.lower()

    def choose_gtfs(self):
        self.gtfs_filename.set(askopenfilename(filetypes=[('GTFS', '.gtfsdb'), ('text files', '.txt')]))
    def choose_od(self):
        self.od_directory.set( askopenfilename(filetypes=[('NumPy Zipped', '.npz')]))
    def onRun(self):
        if self.gtfs_mode.get() :
            prep.prep()
        if not (self.images.get() == 1) : self.images_every.set(0)
        mds.MDS(self.od_directory.get(), self.dimensions.get(), self.n_iterations.get(), self.images_every.get(), self.kernel_chunk.get(), self.list_size.get(), self.debug.get(), self).start()
    def showText(self, text) :
        self.progress.set(text)
    def showImage(self, filename) :
        #self.canvas.delete(ALL)
        self.img = ImageTk.BitmapImage(Image.open(filename))
        self.canvas.create_rectangle(50, 25, 150, 75, fill="blue")
        self.canvas.create_image(0, 0, image = self.img)

        
if __name__ == '__main__': 
    cuda.init()
    Interface().mainloop()
    