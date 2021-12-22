from PyQt5.QtCore import Qt
from PyQt5.QtWidgets import (
    QVBoxLayout,
    QSlider,
)
from typing import Type, List
from .Plot import PlotPanel
import pyqtgraph as pg
class PGFigureLayoutWrap(QVBoxLayout):
    def __init__(self,plotPanels:List[PlotPanel],totalDP:int,slider2DP:int = 0,displayDP:int = 5000):
        super().__init__()
        
        self.totalDP = totalDP
        self.slider2DP = int(totalDP/10) if slider2DP <= 0 else slider2DP
        self.displayDP = displayDP

        self.plotpanels = plotPanels
        self.vertlines = []
        for i,p in enumerate(self.plotpanels):
            self.addWidget(p)
            self.vertlines.append(pg.InfiniteLine(angle = 90,movable = False))
            p.addItem(self.vertlines[i])
        self.slidersmall = QSlider(Qt.Horizontal)
        self.sliderbig = QSlider(Qt.Horizontal)
        self.vertlineslide = QSlider(Qt.Horizontal)


        self.slidersmall.sliderMoved.connect(self.sliderSmove)
        self.sliderbig.sliderMoved.connect(self.sliderBmove)
        self.vertlineslide.sliderMoved.connect(self.vertmove)

        self.slidersmall.setMaximum(self.slider2DP-self.displayDP)
        self.sliderbig.setMaximum(self.totalDP-self.slider2DP)
        self.vertlineslide.setMaximum(displayDP)
        
        self.addWidget(self.slidersmall)
        self.addWidget(self.sliderbig)
        self.addWidget(self.vertlineslide)
    def vertmove(self,e):
        print(e)
        for line in self.plotpanels:
            line.setValue(e)
    def sliderSmove(self,e):
        print('Small:',e)
    def sliderBmove(self,e):
        print('Big  :',e)