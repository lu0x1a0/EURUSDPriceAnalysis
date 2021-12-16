"""
PLOT FUNCTIONS
"""
import mplfinance as mpf
def plotsubset(df,indnames, filename,extraserieses = []):
    colors = ['b','g','r','c','m','y','k','w']
    #print(len(dataset.raw))
    #print(len(df))
    apds = [
            mpf.make_addplot(df[name],panel=0,color=colors[i])
            for i,name in enumerate(indnames)
        ] + [
            mpf.make_addplot(df[indnames[1]]-df[indnames[2]],panel=1,color='b'),
            mpf.make_addplot(df[indnames[2]]-df[indnames[3]],panel=1,color='g'),
        ] + [
            mpf.make_addplot(df[indnames[-2]]-df[indnames[-1]]>0,panel=2,color='b'),
        ] + [
            mpf.make_addplot(s,panel=3+i) for i,s in enumerate(extraserieses)
        ]

    # fig, axes = mpf.plot(df,addplot=apds,figscale=14,volume=False, returnfig=True)
    fig, axes = mpf.plot(
        df,
        type = 'line', 
        figratio=(250,50),
        datetime_format=' %A, %d-%m-%Y',
        volume=False,
        returnfig=True,
        addplot=apds,
        #savefig = "./"+str(filename)+".png"
        )#addplot=apds,figratio=(10,5),volume=False, returnfig=True)