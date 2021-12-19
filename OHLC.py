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
    QSlider,
    QLabel,
)
import pyqtgraph as pg

import numpy as np
import pandas as pd

from abc import ABC, abstractmethod

from DataManipulation.indicators import NStepForwardPredictByD1,D1Crossing, D1, D2, MYEMA, LTSupport

class IndicatorPlot(ABC):
    def __init__(self,name,data,hostplotwidget,color = None):
        self.name = name
        self.data = data
        self.hostplotwidget = hostplotwidget
        self.plot = None
        if color is not None:
            self.color = color
    def getSerialColor(self):
        colors = ('b', 'g', 'r', 'c', 'm', 'y')#, 'k', 'w')
        NOPlotSharingWith = len(self.hostplotwidget.getPlotItem().dataItems)
        return colors[NOPlotSharingWith%len(colors)]
    def datalen(self):
        return len(self.data)
    @abstractmethod
    def createLine(self,startidx,endidx,color = None):
        pass
    @abstractmethod
    def updateData(self,startidx,endidx):
        pass
class SeriesPlot(IndicatorPlot):
    def __init__(self,name, data,hostplotwidget,color =None):
        super().__init__(name,data,hostplotwidget,color)
    def createLine(self,startidx,endidx,color = None):
        if color is None:
            if hasattr(self,'color') == False:
                self.color = self.getSerialColor()
        else:
            self.color = color
        pen = pg.mkPen(self.color) # sets the color
        self.plot = self.hostplotwidget.plot(self.data[startidx:endidx],pen = pen,name=self.name)
        self.hostplotwidget.plotItem.vb.disableAutoRange()
        self.hostplotwidget.plotItem.setMouseEnabled(x=False, y=True)
    def updateData(self,startidx,endidx):
        self.plot.setData(self.data[startidx:endidx],connect="finite")
        
class PredictForwardPlot(IndicatorPlot):
    def __init__(self,name,data,hostplotwidget,color =None):
        super().__init__(name,data,hostplotwidget,color)
        self.startpointratio = 4/5
        self.plot = None
    def createLine(self,startidx,endidx,color = None):
        if color is None:
            if hasattr(self,'color') == False:
                self.color = self.getSerialColor()
        else:
            self.color = color
        pen = pg.mkPen(self.color) # sets the color
        self.startpoint = int(self.startpointratio*(endidx-startidx)) + startidx
        self.plot = self.hostplotwidget.plot(
            list(range(self.startpoint+1-startidx,self.startpoint+1-startidx+len(self.data[self.startpoint,:]))),
            self.data[self.startpoint,:],
            pen = pen,
            name=self.name
        )
    def updateData(self,startidx,endidx):
        self.startpoint = int(self.startpointratio*(endidx-startidx)) + startidx
        self.plot.setData(
            list(range(self.startpoint+1-startidx,self.startpoint+1-startidx+len(self.data[self.startpoint,:]))),
            self.data[self.startpoint,:],
            connect="finite"
        )

class PGFigureLayoutWrap(QVBoxLayout):
    def __init__(self,SeriesList,movingStats = None,datapoints = 10000):
        super().__init__()
        
        self.indicators = []
        #    self.colors = ('b', 'g', 'r', 'c', 'm', 'y', 'k', 'w')
        
        self.plotpanels = []
        self.panelsverts = []
        if isinstance(SeriesList,pd.DataFrame):
            ... 
        elif isinstance(SeriesList,list) and len(SeriesList)>0:
            if isinstance(SeriesList[0],pd.Series):
                ...
            elif isinstance(SeriesList[0],np.ndarray):
                ...
            elif isinstance(SeriesList[0],dict):
                for i in range(SeriesList[-1]['panelidx']+1):
                    self.plotpanels.append(pg.PlotWidget())
                    self.addWidget(self.plotpanels[i])
                    panelsvert = pg.InfiniteLine(angle = 90,movable = False)
                    self.panelsverts.append(panelsvert)
                    self.plotpanels[i].addItem(panelsvert)
                for s in SeriesList:
                    if s['indtype'] == 'series':
                        self.indicators.append(
                            SeriesPlot(
                                s['name'],
                                s['data'],
                                self.plotpanels[s['panelidx']],
                                color = s['color'] if 'color' in s else None
                            )
                        )
                    elif s['indtype'] == 'predictforward':
                        self.indicators.append(
                            PredictForwardPlot(
                                s['name'],
                                s['data'],
                                self.plotpanels[s['panelidx']],
                                color = s['color'] if 'color' in s else None
                            )
                        )
            else:
                pass
        else:
            ...
        
        self.DATAPOINTS = datapoints#self.indicators[0].datalen()
        self.minorDP = int(self.DATAPOINTS*2)
        self.totalDP = self.indicators[0].datalen()
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
        
        self.vertlineslide = QSlider(Qt.Horizontal)
        self.vertlineslide.setMaximum(self.minorDP-self.DATAPOINTS)
        self.vertlineslide.sliderMoved.connect(self.vertmove)
        self.addWidget(self.vertlineslide)

        #if movingStats is not None:
        #    self.statwidget = QWidget()
        #    self.statwidget.setLayout(QHBoxLayout)
        #    for stat in movingStats:
        #        self.statwidget.addWidget()
        #
        #
        #    self.addWidget(statwidget)
            
        
        self.plot() 
    def plot(self):
        self.plotlines = {}
        for indicator in self.indicators:
            indicator.createLine(self.viewStartIdx,self.viewEndIdx)
    def updatePlotData(self): 
        for indicator in self.indicators:
            indicator.updateData(self.viewStartIdx,self.viewEndIdx)
    def sliderSmove(self,e):
        self.viewStartIdx = self.sliderbig.value() + e
        self.viewEndIdx   = self.viewStartIdx + self.DATAPOINTS
        self.updatePlotData()
    def sliderBmove(self,e):
        self.viewStartIdx = self.slidersmall.value() + e
        self.viewEndIdx   = self.viewStartIdx + self.DATAPOINTS
        self.updatePlotData()
    def vertmove(self,e):
        for line in self.panelsverts:
            line.setValue(e)
class MainWindow(QMainWindow):

    def __init__(self,period):
        super(MainWindow, self).__init__()

        self.setWindowTitle("My App")
        self.layout = QHBoxLayout()
        
        self.leftbar = QVBoxLayout()
        self.leftbar.addWidget(QCheckBox())
        
        # right Content
        if period == 'm':
            self.data = self.prepareMinutes()        
            self.rightmain = PGFigureLayoutWrap(self.data,datapoints = 10000*60)
        elif period == 'h':
            self.data = self.prepareHourly()        
            self.rightmain = PGFigureLayoutWrap(self.data,datapoints = 10000)


        self.layout.addLayout(self.leftbar)
        self.layout.addLayout(self.rightmain)

        widget = QWidget()
        widget.setLayout(self.layout)
        self.setCentralWidget(widget)
    def prepareHourly(self,):
        minutes = self.getOHLC_pickle("EURUSD_M_2010_2021.pkl")
        
        hourly = minutes.resample('1H').agg({'Open': 'first', 
                        'High': 'max', 
                        'Low': 'min', 
                        'Close': 'last'}).dropna()
        emaperiods = [100,200,300,24*100,24*200]
        from DataManipulation.DataHandler import DemaMinDayMultinom
        dataHset = DemaMinDayMultinom(hourly,emaperiods = emaperiods)
        data = [
            {'name':'Close'     ,'data':hourly['Close'],   'indtype':'series', 'panelidx':0,'color':'b'           },
            {'name':'ema100'    ,'data':hourly['ema100'],  'indtype':'series', 'panelidx':0,'color':'r'           },
            {'name':'ema200'    ,'data':hourly['ema200'],  'indtype':'series', 'panelidx':0,'color':'g'           },
            {'name':'ema300'    ,'data':hourly['ema300'],  'indtype':'series', 'panelidx':0,'color':'y'           },
            {'name':'ema2400'   ,'data':hourly['ema2400'], 'indtype':'series', 'panelidx':0,'color':'c'           },
            {'name':'ema4800'   ,'data':hourly['ema4800'], 'indtype':'series', 'panelidx':0,'color':'m'           },
            #{'name':'5StepEMA2400','data':NStepForwardPredictByD1(24*5).look(hourly['ema2400'].to_numpy()), 'indtype':'predictforward','panelidx':0,'color':'b'   },
            #{'name':'5StepEMA4800','data':NStepForwardPredictByD1(24*5).look(hourly['ema4800'].to_numpy()), 'indtype':'predictforward','panelidx':0,'color':'r'   },
            #{'name':'24_48crossing','data':D1Crossing(hourly['ema2400'].to_numpy(), hourly['ema4800'].to_numpy(),n=24*5).astype('int8'), 'indtype':'series','panelidx':1   },
                        
            {'name':'4800Var4800','data':hourly['ema4800'].rolling(4800).std(),'indtype':'series','panelidx':1,'color':'m'},
            {'name':'2400Var2400','data':hourly['ema2400'].rolling(2400).std(),'indtype':'series','panelidx':1,'color':'c'},
            {'name':'300Var300','data':hourly['ema300'].rolling(300).std()    ,'indtype':'series','panelidx':1,'color':'y'},
            {'name':'100Var100','data':hourly['ema100'].rolling(100).std()    ,'indtype':'series','panelidx':1,'color':'r'},
            
            {'name':'100Var100D1','data':D1(hourly['ema100'].rolling(100).std())    ,'indtype':'series','panelidx':2,'color':'r'},
            {'name':'300Var300D1','data':D1(hourly['ema300'].rolling(300).std())    ,'indtype':'series','panelidx':2,'color':'y'},
            {'name':'2400Var2400D1','data':D1(hourly['ema2400'].rolling(2400).std())    ,'indtype':'series','panelidx':2,'color':'c'},
            {'name':'2400Var2400D1','data':D1(hourly['ema2400'].rolling(4800).std())    ,'indtype':'series','panelidx':2,'color':'m'},
            
            {'name':'_DIVIDER','data':np.zeros(len(hourly['ema300']))         ,'indtype':'series','panelidx':2,'color':'b'},
            
            {'name':'300Var300D2','data':D2(hourly['ema300'].rolling(300).std())    ,'indtype':'series','panelidx':3,'color':'y'},
            {'name':'2400Var2400D2','data':D2(hourly['ema2400'].rolling(2400).std())    ,'indtype':'series','panelidx':3,'color':'c'},
            {'name':'4800Var4800D2','data':D2(hourly['ema4800'].rolling(4800).std())    ,'indtype':'series','panelidx':3,'color':'m'},

            #{'name':'ema2400D1'   ,'data':D1(hourly['ema2400']), 'indtype':'series', 'panelidx':4,'color':'c'           },
            {'name':'ema2400support'   ,'data':LTSupport(hourly['ema2400'].to_numpy(),emaperiod=2400), 'indtype':'series', 'panelidx':4,'color':'c'           },

            
            #{'name':'300D1DEV','data':hourly['ema300']-hourly['ema300'].rolling(300).mean(),'indtype':'series','panelidx':3}, Unstable
        ]
        return data
    def prepareMinutes(self,):
        minutes = self.getOHLC_pickle("EURUSD_M_2010_2021.pkl")
        emaperiods = [100,200,300,24*100,24*200]
        from DataManipulation.DataHandler import DemaMinDayMultinom
        dataMset = DemaMinDayMultinom(minutes,emaperiods = emaperiods)

        ema100h = MYEMA(minutes['Close'],100*60)
        ema300h = MYEMA(minutes['Close'],300*60)
        ema2400h = MYEMA(minutes['Close'],2400*60)
        ema4800h = MYEMA(minutes['Close'],4800*60)
        
        data = [
            {'name':'Close'     ,'data':minutes['Close'],   'indtype':'series', 'panelidx':0,'color':'b'           },
            {'name':'ema100'    ,'data':minutes['ema100'],  'indtype':'series', 'panelidx':0,'color':'r'           },
            {'name':'ema200'    ,'data':minutes['ema200'],  'indtype':'series', 'panelidx':0,'color':'g'           },
            {'name':'ema300'    ,'data':minutes['ema300'],  'indtype':'series', 'panelidx':0,'color':'y'           },
            {'name':'ema2400'   ,'data':minutes['ema2400'], 'indtype':'series', 'panelidx':0,'color':'c'           },
            {'name':'ema4800'   ,'data':minutes['ema4800'], 'indtype':'series', 'panelidx':0,'color':'m'           },
            {'name':'ema100h'    ,'data':ema100h, 'indtype':'series', 'panelidx':0,'color':'m'           },
            {'name':'ema300h'    ,'data':ema300h, 'indtype':'series', 'panelidx':0,'color':'m'           },
            {'name':'ema2400h'   ,'data':ema2400h, 'indtype':'series', 'panelidx':0,'color':'m'           },
            {'name':'ema4800h'   ,'data':ema4800h, 'indtype':'series', 'panelidx':0,'color':'m'           },
            
            {'name':'4800hVar4800','data':pd.Series(ema4800h).rolling(4800*60).std(),'indtype':'series','panelidx':1,'color':'m'},
            {'name':'2400hVar2400','data':pd.Series(ema2400h).rolling(2400*60).std(),'indtype':'series','panelidx':1,'color':'c'},
            {'name':'300hVar300','data':pd.Series(ema300h).rolling(300*60).std()    ,'indtype':'series','panelidx':1,'color':'y'},
            {'name':'100hVar100','data':pd.Series(ema100h).rolling(100*60).std()    ,'indtype':'series','panelidx':1,'color':'r'},
            {'name':'300hVar300D1','data':D1(pd.Series(ema300h).rolling(300*60).std())    ,'indtype':'series','panelidx':2,'color':'y'},
            #{'name':'_DIVIDER','data':np.zeros(len(minutes['ema300']))         ,'indtype':'series','panelidx':2,'color':'b'},
            
            {'name':'300Var300D2','data':D2(pd.Series(ema300h).rolling(300*60).std())    ,'indtype':'series','panelidx':3,'color':'y'},

            #{'name':'4800Var4800','data':minutes['ema4800'].rolling(4800).std(),'indtype':'series','panelidx':1,'color':'m'},
            #{'name':'2400Var2400','data':minutes['ema2400'].rolling(2400).std(),'indtype':'series','panelidx':1,'color':'c'},
            #{'name':'300Var300','data':minutes['ema300'].rolling(300).std()    ,'indtype':'series','panelidx':1,'color':'y'},
            #{'name':'100Var100','data':minutes['ema100'].rolling(100).std()    ,'indtype':'series','panelidx':1,'color':'r'},
            #{'name':'300Var300D1','data':D1(minutes['ema300'].rolling(300).std())    ,'indtype':'series','panelidx':2,'color':'y'},
            #{'name':'_DIVIDER','data':np.zeros(len(minutes['ema300']))         ,'indtype':'series','panelidx':3,'color':'b'},
            #{'name':'300Var300D2','data':D2(minutes['ema300'].rolling(300).std())    ,'indtype':'series','panelidx':3,'color':'y'},
            
            #{'name':'300D1DEV','data':minutes['ema300']-minutes['ema300'].rolling(300).mean(),'indtype':'series','panelidx':3}, Unstable
        ]
        return data
    def prepareLongSupportSplit(self,):
        minutes = self.getOHLC_pickle("EURUSD_M_2010_2021.pkl")
        hourly = minutes.resample('1H').agg({'Open': 'first', 
                        'High': 'max', 
                        'Low': 'min', 
                        'Close': 'last'}).dropna()
        emaperiods = [100,200,300,24*100,24*200]
        from DataManipulation.DataHandler import DemaMinDayMultinom
        dataHset = DemaMinDayMultinom(hourly,emaperiods = emaperiods)
        data = [
            {'name':'Close'     ,'data':hourly['Close'],   'indtype':'series', 'panelidx':0,'color':'b'           },
            {'name':'ema100'    ,'data':hourly['ema100'],  'indtype':'series', 'panelidx':0,'color':'r'           },
            {'name':'ema200'    ,'data':hourly['ema200'],  'indtype':'series', 'panelidx':0,'color':'g'           },
            {'name':'ema300'    ,'data':hourly['ema300'],  'indtype':'series', 'panelidx':0,'color':'y'           },
            {'name':'ema2400'   ,'data':hourly['ema2400'], 'indtype':'series', 'panelidx':0,'color':'c'           },
            {'name':'ema4800'   ,'data':hourly['ema4800'], 'indtype':'series', 'panelidx':0,'color':'m'           },            
            
            {'name':'4800Var4800','data':hourly['ema4800'].rolling(4800).std(),'indtype':'series','panelidx':2,'color':'m'},
            {'name':'2400Var2400','data':hourly['ema2400'].rolling(2400).std(),'indtype':'series','panelidx':2,'color':'c'},
            {'name':'300Var300','data':hourly['ema300'].rolling(300).std()    ,'indtype':'series','panelidx':2,'color':'y'},
            {'name':'100Var100','data':hourly['ema100'].rolling(100).std()    ,'indtype':'series','panelidx':2,'color':'r'},
            {'name':'300Var300D1','data':D1(hourly['ema300'].rolling(300).std())    ,'indtype':'series','panelidx':3,'color':'y'},
            {'name':'2400Var2400D1','data':D1(hourly['ema2400'].rolling(2400).std())    ,'indtype':'series','panelidx':3,'color':'c'},
            {'name':'2400Var2400D1','data':D1(hourly['ema2400'].rolling(4800).std())    ,'indtype':'series','panelidx':3,'color':'m'},
            
            {'name':'_DIVIDER','data':np.zeros(len(hourly['ema300']))         ,'indtype':'series','panelidx':3,'color':'b'},
            
            {'name':'300Var300D2','data':D2(hourly['ema300'].rolling(300).std())    ,'indtype':'series','panelidx':4,'color':'y'},
            {'name':'2400Var2400D2','data':D2(hourly['ema2400'].rolling(2400).std())    ,'indtype':'series','panelidx':4,'color':'c'},
            {'name':'4800Var4800D2','data':D2(hourly['ema4800'].rolling(4800).std())    ,'indtype':'series','panelidx':4,'color':'m'},
            #{'name':'300D1DEV','data':hourly['ema300']-hourly['ema300'].rolling(300).mean(),'indtype':'series','panelidx':3}, Unstable
        ]
        return data
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
if __name__ == '__main__':
    print("MAIN")
    import argparse 
    
    parser = argparse.ArgumentParser(description='Process some integers.')
    parser.add_argument('-p','--period',action='store',required=True,help = "h for hour, m for minutes")

    args = parser.parse_args()
    if args.period == 'h' or args.period == 'm':
        app = QApplication(sys.argv)

        window = MainWindow(args.period)
        window.show()

        app.exec()    
    else:
        print("not supported")

