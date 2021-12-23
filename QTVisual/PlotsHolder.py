from PyQt5.QtCore import Qt
from PyQt5.QtWidgets import (
    QVBoxLayout,
    QSlider,
)
from superqt import QRangeSlider
from typing import Type, List
from .Plot import PlotPanel
import pyqtgraph as pg
class PGFigureLayoutWrap(QVBoxLayout):
    def __init__(self,plotPanels:List[PlotPanel],totalDP:int,slider2DP:int = 0,displayDP:int = 5000):
        super().__init__()
        
        self.totalDP = totalDP
        self.slider2DP = int(totalDP/10) if slider2DP <= 0 else slider2DP
        self.displayDP = displayDP
        self.viewStartIdx = 0
        self.viewEndIdx   = self.displayDP

        self.plotpanels = plotPanels
        self.vertlines = []
        for i,p in enumerate(self.plotpanels):
            self.addWidget(p)
            self.vertlines.append(pg.InfiniteLine(angle = 90,movable = False))
            p.addItem(self.vertlines[i])
            p.createPlot(self.viewStartIdx,self.viewEndIdx)
        self.slidersmall = QRangeSlider(Qt.Horizontal)
        self.sliderbig = QSlider(Qt.Horizontal)
        self.vertlineslide = QSlider(Qt.Horizontal)


        self.slidersmall.sliderMoved.connect(self.sliderSmove)
        self.slidersmall.setTracking(True)
        self.sliderbig.sliderMoved.connect(self.sliderBmove)
        self.vertlineslide.sliderMoved.connect(self.vertmove)

        self.slidersmall.setMaximum(self.slider2DP)
        self.slidersmall.setValue((0,self.displayDP))
        self.sliderbig.setMaximum(self.totalDP-self.slider2DP)
        self.vertlineslide.setMaximum(displayDP)
        
        self.addWidget(self.slidersmall)
        self.addWidget(self.sliderbig)
        self.addWidget(self.vertlineslide)
    def updatePanels(self,startidx,endidx):
        for plotpanel in self.plotpanels:
            plotpanel.updateData(startidx,endidx)
    def vertmove(self,e):
        print(e)
        for line in self.vertlines:
            line.setValue(e)
    def sliderSmove(self,e):
        print('Small:',e)
        self.viewStartIdx = self.sliderbig.value() + e[0]
        self.viewEndIdx   = self.viewStartIdx + (e[1] - e[0])#self.displayDP
        print(int(self.viewStartIdx),int(self.viewEndIdx))
        self.updatePanels(int(self.viewStartIdx),int(self.viewEndIdx))
    def sliderBmove(self,e):
        #print('Big:',e)
        self.viewStartIdx = self.slidersmall.value()[0] + e
        self.viewEndIdx   = self.viewStartIdx + (self.slidersmall.value()[1] - self.slidersmall.value()[0])#self.displayDP
        self.updatePanels(int(self.viewStartIdx),int(self.viewEndIdx))