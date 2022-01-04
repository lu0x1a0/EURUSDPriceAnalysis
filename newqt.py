import sys
from PyQt5.QtWidgets import QApplication, QMainWindow, QHBoxLayout, QVBoxLayout, QWidget

from QTVisual.PlotsHolder import PGFigureLayoutWrap
from QTVisual.Plot import PlotPanel
#from utils import plotloader

def getOHLC_pickle(pklpath):
        import pickle
        with open(pklpath,'rb') as f:
            data = pickle.load(f)
            return data
def getOHLC_raw(to_pickle=False):
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



class MainWindow(QMainWindow):

    def __init__(self, picklepath = None, dfs = None):
        super(MainWindow, self).__init__()

        self.setWindowTitle("My App")
        self.layout = QHBoxLayout()
        
        self.leftbar = QVBoxLayout()
        
        if picklepath is None:
            self.plotpanels,self.data = Test2Plots()
        elif dfs is None:
            self.dfs = getOHLC_pickle(picklepath)
            self.data = self.dfs[0]
            self.plotpanels = [PlotPanel(df) for df in self.dfs]
        else:
            self.dfs = dfs
            self.data = self.dfs[0]
            self.plotpanels = [PlotPanel(df) for df in self.dfs]
        #print(self.data.tail(100))
        self.rightmain = PGFigureLayoutWrap(self.plotpanels, len(self.data))
        self.layout.addLayout(self.leftbar)
        self.layout.addLayout(self.rightmain)

        widget = QWidget()
        widget.setLayout(self.layout)
        self.setCentralWidget(widget)
def Test1Plots():
    from DataManipulation.indicators import DEMA,MYEMA,D1
    from time import time
    t = time()
    data = getOHLC_pickle("EURUSD_M_2010_2021.pkl")
    # 9min
    demaperiod = [9]
    # 100 min, 200 min, 5 h, 40 h, 80 h, 100 h, 300 h, 100 d, 200 d
    #                        1.6d, 3.3d, 4.16d, 12.5d, 
    emaperiod  = [100,200,300,2400,4800, 100*60,300*60,2400*60,4800*60]
    for p in demaperiod:
        data['dema'+str(p)] = DEMA(data['Close'],p)
    for p in emaperiod:
        data['ema'+str(p)] = MYEMA(data['Close'],p)
    D1demaperiod = [9]
    D1emaperiod = [100,200,300,2400,4800]
    for p in D1demaperiod:
        data['D1dema'+str(p)] = D1(data['dema'+str(p)])
    for p in D1emaperiod:
        data['D1ema'+str(p)] = D1(data['ema'+str(p)])

    stdPeriod = [9,100,300,100*60]
    stdInd    = ['D1dema9']#,'D1ema100','D1ema300']
    for p in stdPeriod:
        for i in stdInd:
            data[i+"_std"+str(p)] = data[i].rolling(p).std()
    print(data.columns)
    print(time()-t)
    t = time()
    plotpanels = [
        PlotPanel(
            data[['Close']+['dema'+str(p) for p in demaperiod]+['ema'+str(p) for p in emaperiod]],
        ),
        PlotPanel(
            data[['D1dema'+str(p) for p in D1demaperiod]+['D1ema'+str(p) for p in D1emaperiod]]
        ),
        PlotPanel(
            data[['D1dema9_std'+str(p) for p in stdPeriod]]
        )
    ]
    return plotpanels,data

def Test2Plots():
    from DataManipulation.indicators import DEMA,MYEMA,D1
    from time import time
    t = time()
    mdata = getOHLC_pickle("EURUSD_M_2010_2021.pkl")
    data = mdata.resample('1H').agg({'Open': 'first', 
                        'High': 'max', 
                        'Low': 'min', 
                        'Close': 'last'}).dropna()
    # 9min
    demaperiod = [9]
    # 100 min, 200 min, 5 h, 40 h, 80 h, 100 h, 300 h, 100 d, 200 d
    #                        1.6d, 3.3d, 4.16d, 12.5d, 
    emaperiod  = [100,200,300,2400,4800, 100*60,300*60,2400*60,4800*60]
    for p in demaperiod:
        data['dema'+str(p)] = DEMA(data['Close'],p)
    for p in emaperiod:
        data['ema'+str(p)] = MYEMA(data['Close'],p)
    D1demaperiod = [9]
    D1emaperiod = [100,200,300,2400,4800]
    for p in D1demaperiod:
        data['D1dema'+str(p)] = D1(data['dema'+str(p)])
    for p in D1emaperiod:
        data['D1ema'+str(p)] = D1(data['ema'+str(p)])

    stdPeriod = [9,100,300,100*60]
    stdInd    = ['D1dema9','dema9']#,'D1ema100','D1ema300']
    for p in stdPeriod:
        for i in stdInd:
            data[i+"_std"+str(p)] = data[i].rolling(p).std()
    
    print(data.columns)
    print(time()-t)
    t = time()
    plotpanels = [
        PlotPanel(
            data[['Close']+['dema'+str(p) for p in demaperiod]+['ema'+str(p) for p in emaperiod]],
        ),
        PlotPanel(
            data[['D1dema'+str(p) for p in D1demaperiod]+['D1ema'+str(p) for p in D1emaperiod]]
        ),
        PlotPanel(
            data[['D1dema9_std'+str(p) for p in stdPeriod]]
        ),
        PlotPanel(
            data[['dema9_std'+str(p) for p in stdPeriod]]
        )
    ]
    return plotpanels,data


app = QApplication(sys.argv)

window = MainWindow()
window.show()

app.exec()    