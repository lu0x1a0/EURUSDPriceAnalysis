import sys

from PyQt5.QtCore import Qt, QSize
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
from typing import Union, List,Type
class PlotPanel(pg.PlotWidget):
    # fix axes decimals https://stackoverflow.com/questions/59768880/how-to-format-y-axis-displayed-numbers-in-pyqtgraph
    def __init__(self,series:Union[List[pd.Series],Type[pd.DataFrame]],colors = None, types = None):
        self.palette = ('b', 'g', 'r', 'c', 'm', 'y')#, 'k', 'w')
        self.colors = [self.palette[i%len(self.palette)] for i,_ in enumerate(series)] if colors is None else colors
        self.series = series
        if isinstance(series,list) and isinstance(series[0],pd.Series):
            for l in self.series:
                pass            
        elif isinstance(series,pd.DataFrame):
            pass
    def plot(self):
        pass
    def updateData(startidx,endidx):
        pass
    #https://stackoverflow.com/questions/59768880/how-to-format-y-axis-displayed-numbers-in-pyqtgraph