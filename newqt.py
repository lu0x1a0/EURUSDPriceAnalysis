import sys
from PyQt5.QtWidgets import QApplication, QMainWindow, QHBoxLayout, QVBoxLayout, QWidget

from QTVisual.PlotsHolder import PGFigureLayoutWrap
from QTVisual.Plot import PlotPanel

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

    def __init__(self):
        super(MainWindow, self).__init__()

        self.setWindowTitle("My App")
        self.layout = QHBoxLayout()
        
        self.leftbar = QVBoxLayout()
        
        self.data = getOHLC_pickle("EURUSD_M_2010_2021.pkl")
        self.plotpanels = [
            PlotPanel(
                self.data[['Close']],
            ),
            PlotPanel(
                self.data[['Close']]
            )
        ]
        self.rightmain = PGFigureLayoutWrap(self.plotpanels, len(self.data))

        self.layout.addLayout(self.leftbar)
        self.layout.addLayout(self.rightmain)

        widget = QWidget()
        widget.setLayout(self.layout)
        self.setCentralWidget(widget)

data = getOHLC_pickle("EURUSD_M_2010_2021.pkl")



app = QApplication(sys.argv)

window = MainWindow()
window.show()

app.exec()    