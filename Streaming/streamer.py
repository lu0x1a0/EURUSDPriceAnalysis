from CONFIG import config
print(config)

import logging

from trading_ig import IGService, IGStreamService
#from trading_ig.config import config
from trading_ig.lightstreamer import Subscription

import pandas as pd
import datetime
minute_storage = pd.DataFrame(columns=['time','Open','High','Low','Close'])
temporary_storage = pd.DataFrame(columns=['time','bid','ask','mid'])
def on_prices_update_OLD(item_update):
    # print("price: %s " % item_update)
    #print(
    #    "{stock_name:<19}: Time {UPDATE_TIME:<8} - "
    #    "Bid {BID:>5} - Ask {OFFER:>5} - CHANGE {CHANGE:>5} - STATE {MARKET_STATE} -"
    #    "MARKET_DELAY {MARKET_DELAY}".format(
    #        stock_name=item_update["name"], **item_update["values"]
    #    )
    #)
    #print(item_update)
    ...
# A simple function acting as a Subscription listener
def on_prices_update(item_update):
    global temporary_storage
    global minute_storage
    # print("price: %s " % item_update)
    #print(
    #    "{stock_name:<19}: Time {UPDATE_TIME:<8} - "
    #    "Bid {BID:>5} - Ask {OFFER:>5} - CHANGE {CHANGE:>5} - STATE {MARKET_STATE} -"
    #    "MARKET_DELAY {MARKET_DELAY}".format(
    #        stock_name=item_update["name"], **item_update["values"]
    #    )
    #)
    #print(item_update)
    vals = item_update['values']
    updatetime = datetime.datetime.strptime(vals['UPDATE_TIME'],"%H:%M:%S")
    
    timewithoutSeconds = updatetime.replace(second=0, microsecond=0)
    if len(temporary_storage)>1:
        lasttimewithoutSeconds = temporary_storage['time'].iloc[-2].replace(second=0,microsecond=0)
        #print(timewithoutSeconds,lasttimewithoutSeconds)
        if timewithoutSeconds != lasttimewithoutSeconds:
            print("Minute has passed, Now:",timewithoutSeconds," Last:",lasttimewithoutSeconds)
            minute_storage = minute_storage.append({
                'time':lasttimewithoutSeconds,
                'Open':temporary_storage.iloc[0]['mid'],
                'High':temporary_storage['mid'].max(),
                'Low':temporary_storage['mid'].min(),
                'Close':temporary_storage.iloc[-1]['mid']
            },ignore_index = True)
            print(minute_storage)
            temporary_storage = pd.DataFrame(columns=['time','bid','ask','mid'])
    temporary_storage = temporary_storage.append(
        {'time':updatetime,
        'bid':float(vals['BID']),
        'ask':float(vals['OFFER']),
        'mid':(float(vals['BID'])+float(vals['OFFER']))/2
    }, ignore_index =  True)
def on_account_update(balance_update):
    print("balance: %s " % balance_update)


def main():
    logging.basicConfig(level=logging.INFO)
    # logging.basicConfig(level=logging.DEBUG)

    ig_service = IGService(
        config.username, config.password, config.api_key, config.acc_type, acc_number=config.acc_number
    )

    ig_stream_service = IGStreamService(ig_service)
    ig_stream_service.create_session()
    #ig_stream_service.create_session(version='3')

    # Making a new Subscription in MERGE mode
    subscription_prices = Subscription(
        mode = "MERGE",
        #mode="DISTINCT",
        #items = ["CHART:CS.D.EURUSD.CFD.IP:TICK"],
        #items = ["MARKET:CS.D.EURUSD.CFD.IP"],
        items = ["L1:CS.D.EURUSD.CFD.IP"],
        
        #items=["L1:CS.D.GBPUSD.CFD.IP", "L1:CS.D.USDJPY.CFD.IP"], # sample CFD epics
        #items=["L1:CS.D.GBPUSD.TODAY.IP", "L1:IX.D.FTSE.DAILY.IP"], # sample spreadbet epics
        fields=["UPDATE_TIME", "BID", "OFFER", "CHANGE", "MARKET_STATE","MARKET_DELAY"],
        #fields = ['LTV']
    )

    # Adding the "on_price_update" function to Subscription
    subscription_prices.addlistener(on_prices_update)

    # Registering the Subscription
    sub_key_prices = ig_stream_service.ls_client.subscribe(subscription_prices)

    # Making an other Subscription in MERGE mode
    subscription_account = Subscription(
        mode="MERGE", items=["ACCOUNT:" + config.acc_number], fields=["AVAILABLE_CASH"],
    )

    # Adding the "on_balance_update" function to Subscription
    subscription_account.addlistener(on_account_update)

    # Registering the Subscription
    sub_key_account = ig_stream_service.ls_client.subscribe(subscription_account)

    input(
        "{0:-^80}\n".format(
            "HIT CR TO UNSUBSCRIBE AND DISCONNECT FROM \
    LIGHTSTREAMER"
        )
    )

    # Disconnecting
    ig_stream_service.disconnect()


if __name__ == "__main__":
    main()