import sys

from PyQt5.QtCore import Qt, QEvent
from PyQt5.QtGui import QPalette, QColor
from PyQt5.QtWidgets import QAction, QApplication, QLabel, QMainWindow, QMenu
from PyQt5.QtWidgets import (
    QHBoxLayout,
    QVBoxLayout,
    QWidget,
    QSlider,
    QLabel,
)

from PyQt5.QtGui import QPalette, QColor
import pyqtgraph as pg
import pandas as pd
from typing import Type
class PlotPanel(pg.PlotWidget):
    # fix axes decimals https://stackoverflow.com/questions/59768880/how-to-format-y-axis-displayed-numbers-in-pyqtgraph
    def __init__(self,df:Type[pd.DataFrame],colors = None, types = None):
        super().__init__()
        self.palette = ('b', 'g', 'r', 'c', 'm', 'y')#, 'k', 'w')
        self.colors = {series_name:self.palette[i%len(self.palette)] for i,series_name in enumerate(df)} if colors is None else colors
        self.df = df
        self.indicators = {} # Dict{plot}
        print(self.indicators)
        #self.installEventFilter(self)
        #if isinstance(series,list) and isinstance(series[0],pd.Series):
    def createPlot(self,startidx,endidx):
        for series_name in self.df:
            pen = pg.mkPen(self.colors[series_name])
            self.indicators[series_name] = self.plot(self.df[series_name].iloc[startidx:endidx],pen=pen,name=series_name)
            
    def updateData(self,startidx,endidx):
        for series_name in self.df:
            self.indicators[series_name].setData(self.df[series_name].iloc[startidx:endidx],connect="finite")
    #def eventFilter(self, watched, event):
    #    if event.type() == QEvent.GraphicsSceneWheel:
    #        return True
    #    return super().eventFilter(watched, event)