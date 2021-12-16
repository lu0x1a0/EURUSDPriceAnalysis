import sys

from PyQt5.QtCore import Qt, QSize
from PyQt5.QtGui import QPalette, QColor
from PyQt5.QtWidgets import QAction, QApplication, QLabel, QMainWindow, QMenu
from PyQt5.QtWidgets import (
    QHBoxLayout,
    QVBoxLayout,
    QStackedLayout,
    QWidget,
    QCheckBox,
    QSlider
)
import pyqtgraph as pg

import numpy as np
import pandas as pd
class LinePlotWrap():
   pass
class PGFigureLayoutWrap(QVBoxLayout):
    def __init__(self,SeriesList):
        super().__init__()
        self.lineDict = {}
        self.panels = 1

        if   isinstance(SeriesList,pd.DataFrame):
            self.colors = ('b', 'g', 'r', 'c', 'm', 'y', 'k', 'w')
            for i in range(len(SeriesList.columns)):
                if SeriesList[SeriesList.columns[i]].to_numpy().dtype != 'int8':
                    self.lineDict[SeriesList.columns[i]] = (
                        SeriesList[SeriesList.columns[i]],
                        self.colors[i%len(self.colors)],
                        0
                    )
                else :
                    self.lineDict[SeriesList.columns[i]] = (
                        SeriesList[SeriesList.columns[i]],
                        self.colors[i%len(self.colors)],
                        self.panels
                    )
                    self.panels += 1

        elif isinstance(SeriesList,list) and len(SeriesList)>0:
            if isinstance(SeriesList[0],pd.Series):
                ...
            elif isinstance(SeriesList[0],np.ndarray):
                ...
            elif isinstance(SeriesList[0],dict):
                ...
            else:
                pass
        else:
            ...
        
        self.plotpanels = []
        for x in range(self.panels):
            self.plotpanels.append(pg.PlotWidget())
            self.addWidget(self.plotpanels[x])
        
        self.DATAPOINTS = 500
        self.minorDP = 2000
        self.totalDP = len(self.lineDict[list(self.lineDict.keys())[0]][0]) # assumes all in SeriesList are equal length
        self.viewStartIdx = 0
        self.viewEndIdx = self.DATAPOINTS
        
        self.slidersmall = QSlider(Qt.Horizontal)
        self.slidersmall.sliderMoved.connect(self.sliderSmove)
        self.slidersmall.setMaximum(self.minorDP-self.DATAPOINTS)
        self.sliderbig = QSlider(Qt.Horizontal)
        self.sliderbig.sliderMoved.connect(self.sliderBmove)
        #print(self.totalDP-self.minorDP)
        self.sliderbig.setMaximum(self.totalDP-self.minorDP)
        self.addWidget(self.slidersmall)
        self.addWidget(self.sliderbig)

        self.plot() 
    def plot(self):
        self.plotlines = {}
        for key in self.lineDict:
            pen = pg.mkPen(self.lineDict[key][1]) # sets the color
            #print(self.lineDict[key])
            self.plotlines[key] = self.plotpanels[self.lineDict[key][2]].plot(self.lineDict[key][0][0:self.DATAPOINTS],pen = pen,name=key)
            
    def updatePlotData(self):   
        for key in self.lineDict:
            #print("-------------------------------------------")
            #print(type(self.plots[self.lineDict[key][2]]))
            pen = pg.mkPen(self.lineDict[key][1]) # sets the color
            self.plotlines[key].setData(
                self.lineDict[key][0][self.viewStartIdx:self.viewEndIdx],
                name = key
            )
            #print(type(self.plots[self.lineDict[key][2]]))
    def sliderSmove(self,e):
        self.viewStartIdx = self.sliderbig.value() + e
        self.viewEndIdx   = self.viewStartIdx + self.DATAPOINTS
        self.updatePlotData()
    def sliderBmove(self,e):
        print("Called")
        self.viewStartIdx = self.slidersmall.value() + e
        self.viewEndIdx   = self.viewStartIdx + self.DATAPOINTS
        self.updatePlotData()
class MainWindow(QMainWindow):

    def __init__(self):
        super(MainWindow, self).__init__()

        self.setWindowTitle("My App")
        self.layout = QHBoxLayout()
        
        self.leftbar = QVBoxLayout()
        self.leftbar.addWidget(QCheckBox())
        
        # right Content
        self.rightmain = QVBoxLayout()
        
        self.DATAPOINTS = 500

        #self.data = self.getOHLC_raw(to_pickle=True)
        minutes = self.getOHLC_pickle("EURUSD_M_2010_2021.pkl")
        
        #self.data = self.data[['Open','Close']]
        
        hourly = data.resample('1H').agg({'Open': 'first', 
                        'High': 'max', 
                        'Low': 'min', 
                        'Close': 'last'}).dropna()
        print("COPY TIME:", time()-t)
        emaperiods = [100,200,300,24*100,24*200]
        dataHset = DemaMinDayMultinom(hourly,emaperiods = emaperiods)
        from DataManipulation.indicators import NStepForwardPredictByD12
        self.data = [
            {'name':'Close'     ,'data':hourly['Close'],   'indtype':'series', 'panel':0           },
            {'name':'ema100'    ,'data':hourly['ema100'],  'indtype':'series', 'panel':0           },
            {'name':'ema200'    ,'data':hourly['ema200'],  'indtype':'series', 'panel':0           },
            {'name':'ema300'    ,'data':hourly['ema300'],  'indtype':'series', 'panel':0           },
            {'name':'ema2400'   ,'data':hourly['ema2400'], 'indtype':'series', 'panel':0           },
            {'name':'ema4800'   ,'data':hourly['ema4800'], 'indtype':'series', 'panel':0           },
            {'name':'5StepEMA2400','data':NStepForwardPredictByD12(5).look(hourly['ema2400'].to_numpy()), 'indtype':'predictforward','panel':0   },
            {'name':'5StepEMA4800','data':NStepForwardPredictByD12(5).look(hourly['ema4800'].to_numpy()), 'indtype':'predictforward','panel':0   },

        ]

        self.minorDP = int(len(self.data)/100)
        self.totalDP = len(self.data)


        self.rightmain = PGFigureLayoutWrap(self.data)

        self.layout.addLayout(self.leftbar)
        self.layout.addLayout(self.rightmain)

        widget = QWidget()
        widget.setLayout(self.layout)
        self.setCentralWidget(widget)

    def plotdata(self):
        self.singleline = self.plotwindow.plot(self.data['Close'].iloc[:self.DATAPOINTS])
    def getOHLC_pickle(self,pklpath):
        import pickle
        with open(pklpath,'rb') as f:
            data = pickle.load(f)
            return data
    def getOHLC_raw(self,to_pickle=False):
        from DataManipulation.DataHandler import  mergeMonths, DemaMinDayMultinom
        import pandas as pd
        from torch.utils.data import ConcatDataset
        #data = mergePeriod(1,4,beg="DAT_ASCII_EURUSD_M1_2021",dir="./eurusd2021/",dump=False)
        data2021 = mergeMonths(start=1, end=9,beg="DAT_ASCII_EURUSD_M1_2021",dir = './Data/eurusd2021/', dump=False)

        dir = "./Data/"
        beg = "DAT_ASCII_EURUSD_M1_"

        years = range(2010,2021)
        dataY = []
        for y in years:
            dataY.append(pd.read_csv(dir+beg+str(y)+'.csv',sep=';',names = ['Open','High','Low','Close','Volume']))
        dataY.append(data2021)
        data = pd.concat(dataY)
        data.index.name = "Date"
        data.index = pd.to_datetime(data.index)
        if to_pickle:
            import pickle
            with open('OHLC.pkl','wb') as f:
                pickle.dump(data,f,protocol=pickle.HIGHEST_PROTOCOL)
        return data
app = QApplication(sys.argv)

window = MainWindow()
window.show()

app.exec()