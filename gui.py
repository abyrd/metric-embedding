#!/usr/bin/env python2.6
#

from Tkinter import *
from tkFileDialog   import askopenfilename        
from tkFileDialog   import askdirectory        
from tkColorChooser import askcolor               
from tkMessageBox   import askquestion, showerror
from tkSimpleDialog import askfloat

def callback():
    status.set('%s', 'blah!')

class Application(Tk):
    def __init__(self):
        Tk.__init__(self)
        self.title('GPU MDS Algorithm')
        self.geometry('700x600+100+30')
        
        self.gtfs_mode    = IntVar()
        self.od_directory  = StringVar()
        self.od_directory.set(' ')
        self.gtfs_filename  = StringVar()
        self.gtfs_filename.set(' ')

        self.map_margin   = IntVar()
        self.dimensions   = IntVar()
        self.list_size    = IntVar()
        self.kernel_chunk = IntVar()
        self.images       = IntVar()
        self.images_every = IntVar()
        self.converge     = IntVar()
        self.n_iterations = IntVar()
        self.converge_on  = IntVar()
        self.debug = IntVar()
        
        Radiobutton(self, text="Precomputed OD matrix", variable=self.gtfs_mode, value=0).grid(row=0, column=0, sticky=W)
        Button(self, text="choose", command=self.choose_od).grid(row=0, column=1, sticky=W)
        Label(self, textvariable=self.od_directory).grid(row=0, column=2, sticky=W)

        Radiobutton(self, text="New OD matrix from GTFS", variable=self.gtfs_mode, value=1).grid(row=1, column=0, sticky=W)
        Button(self, text="choose", command=self.choose_gtfs).grid(row=1, column=1, sticky=W)
        Label(self, textvariable=self.gtfs_filename).grid(row=1, column=2, sticky=W)
        
        Label(self, text="Map margin (meters)").grid(row=2, column=0, sticky=W)
        Scale(self, variable=self.map_margin,   
                      bg='black',
                      from_=0, to=10000,
                      state = 'disabled',
                      tickinterval=2000,
                      resolution=500,
                      showvalue=YES, orient='horizontal').grid(row=2, column=1, columnspan=2, sticky=EW)
        
        
        Label(self, text="Target space dimensionality").grid(row=3, column=0, sticky=W)
        Scale(self, variable=self.dimensions,       
                      from_=2, to=6,
                      tickinterval=1,
                      length=300,
                      showvalue=NO, orient='horizontal').grid(row=3, column=1, columnspan=2, sticky=W)

        Label(self, text="Per texel station list size").grid(row=4, column=0, sticky=W)
        Scale(self, variable=self.list_size,       
                      from_=1, to=50,
                      tickinterval=10,
                      resolution=5,
                      showvalue=NO, orient='horizontal').grid(row=4, column=1, columnspan=2, sticky=EW)

        Label(self, text="Kernel chunk size").grid(row=5, column=0, sticky=W)
        Scale(self, variable=self.kernel_chunk,       
                      from_=100, to=1000,
                      tickinterval=100,
                      resolution=100,
                      showvalue=NO, orient='horizontal').grid(row=5, column=1, columnspan=2, sticky=EW)

        Checkbutton(self, text="Write image every n iterations", variable=self.images).grid(row=6, sticky=W)
        Scale(self, variable=self.images_every,       
                      from_=1, to=20,
                      tickinterval=5,
                      showvalue=NO, orient='horizontal').grid(row=6, column=1, columnspan=2, sticky=EW)

        Radiobutton(self, text="Terminate after N interations", variable=self.converge, value=0).grid(row=7, column=0, sticky=W)
        Scale(self, variable=self.n_iterations,       
                      from_=0, to=1000,
                      tickinterval=200,
                      resolution=50,
                      showvalue=NO, orient='horizontal').grid(row=7, column=1, columnspan=2, sticky=EW)

        Radiobutton(self, text="Converge to average error", variable=self.converge, value=1).grid(row=8, sticky=W)
        Scale(self, command=self.onMove,     
                      variable=self.converge_on,       
                      from_=0, to=1000,
                      tickinterval=200,
                      resolution=50,
                      showvalue=NO, orient='horizontal').grid(row=8, column=1, columnspan=2, sticky=EW)
                      
        Checkbutton(self, text="Produce debug output", variable=self.debug).grid(row=9, sticky=W)

        Button(self, text="[ LAUNCH ]", command=self.onRun).grid(row=99, columnspan=3, sticky=NSEW)

        # configure grid on container (main window)
        self.columnconfigure(2,weight=1)
        self.rowconfigure(99,weight=1)

        # make another top level window
        top = Toplevel()
        top.lower()

    def onMove(self, value):
        print 'in onMove', value
    def onRun(self):
        print 'You picked %d dimensions.' % self.dimensions.get()
    def choose_gtfs(self):
        self.gtfs_filename.set(askopenfilename(filetypes=[('GTFS', '.gtfsdb'), ('text files', '.txt')]))
    def choose_od(self):
        self.od_directory.set(askdirectory())
    def report(self):
        print self.dimensions.get()

if __name__ == '__main__': 
    Application().mainloop()
    