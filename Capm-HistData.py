import pandas as pd
import yfinance as yf
import numpy as np
import matplotlib.pyplot as plt

###################################################
############# Historical Datas ####################
###################################################

ticker = "^GSPC"
MK = yf.download(ticker)
MK_mensile = MK.resample('ME').last()
MK_mensile['Monthly_Return'] = MK_mensile['Adj Close'].pct_change()
MK_mensile['Monthly_TotalLogReturn'] = np.log(MK_mensile['Monthly_Return']+1)

data_inizio = pd.Timestamp("2014-09-30")
data_fine = pd.Timestamp("2024-11-30")
MK_filtrato = MK_mensile.loc[data_inizio:data_fine]

###################################################
BR = pd.read_csv("BR.N.csv")
BR = BR.drop("Unnamed: 0",axis=1)
BR = BR.drop("Instrument",axis=1)
BR['Date'] = pd.to_datetime(BR['Date'], errors='coerce')
BR['Date'] = BR['Date'].dt.date
BR = BR.set_index("Date")
BR.index = pd.to_datetime(BR.index)
BR["TotalLogReturn"] = (np.log(BR["1 Month Total Return"]/100 + 1))
numBR = BR["1 Month Total Return"].isna().sum().sum()
#print("Number of NaN: " + str(numBR))
BR = BR.iloc[numBR:]
BR["Company Market Cap"].fillna(BR["Company Market Cap"][1], inplace=True)
###################################################
BRKb = pd.read_csv("BRKb.N.csv")
BRKb = BRKb.drop("Unnamed: 0",axis=1)
BRKb = BRKb.drop("Instrument",axis=1)
BRKb['Date'] = pd.to_datetime(BRKb['Date'], errors='coerce')
BRKb['Date'] = BRKb['Date'].dt.date
BRKb = BRKb.set_index("Date")
BRKb.index = pd.to_datetime(BRKb.index)
BRKb["TotalLogReturn"] = (np.log(BRKb["1 Month Total Return"]/100 + 1))
numBRKb = BRKb["Company Market Cap"].isna().sum().sum()
#print("Number of NaN: " + str(numBRKb))
BRKb = BRKb.iloc[numBRKb:]
###################################################
CZR = pd.read_csv("CZR.OQ.csv")
CZR = CZR.drop("Unnamed: 0",axis=1)
CZR = CZR.drop("Instrument",axis=1)
CZR['Date'] = pd.to_datetime(CZR['Date'], errors='coerce')
CZR['Date'] = CZR['Date'].dt.date
CZR = CZR.set_index("Date")
CZR.index = pd.to_datetime(CZR.index)
CZR["TotalLogReturn"] = (np.log(CZR["1 Month Total Return"]/100+1))
numCZR = CZR["Company Market Cap"].isna().sum().sum()
#print("Number of NaN: " + str(numCZR))
CZR = CZR.iloc[numCZR:]
CZR.fillna(CZR["TotalLogReturn"].mean(), inplace = True)
###################################################
PPG = pd.read_csv("PPG.N.csv")
PPG = PPG.drop("Unnamed: 0",axis=1)
PPG = PPG.drop("Instrument",axis=1)
PPG['Date'] = pd.to_datetime(PPG['Date'], errors='coerce')
PPG['Date'] = PPG['Date'].dt.date
PPG = PPG.set_index("Date")
PPG.index = pd.to_datetime(PPG.index)
PPG["TotalLogReturn"] = (np.log(PPG["1 Month Total Return"]/100+1))
numPPG = PPG["Company Market Cap"].isna().sum().sum()
#print("Number of NaN: " + str(numPPG))
###################################################
PRU = pd.read_csv("PRU.N.csv", delimiter=",")
PRU = PRU.drop("Unnamed: 0",axis=1)
PRU = PRU.drop("Instrument",axis=1)
PRU['Date'] = pd.to_datetime(PRU['Date'], errors='coerce')
PRU['Date'] = PRU['Date'].dt.date
PRU = PRU.set_index("Date")
PRU.index = pd.to_datetime(PRU.index)
PRU["TotalLogReturn"] = (np.log(PRU["1 Month Total Return"]/100+1))
numPRUG = PRU["Company Market Cap"].isna().sum().sum()
#print("Number of NaN: " + str(numPRUG))
###################################################
data_inizio = pd.Timestamp("2014-09-30")
data_fine = pd.Timestamp("2024-11-30")

BR_filtrato = BR.loc[data_inizio:data_fine]
BRKb_filtrato = BRKb.loc[data_inizio:data_fine]
CZR_filtrato = CZR.loc[data_inizio:data_fine]
PPG_filtrato = PPG.loc[data_inizio:data_fine]
PRU_filtrato = PRU.loc[data_inizio:data_fine]
###################################################
MKmean = MK_filtrato['Monthly_TotalLogReturn'].mean()
BRmean = BR_filtrato['TotalLogReturn'].mean()
BRKbmean = BRKb_filtrato['TotalLogReturn'].mean()
CZRmean = CZR_filtrato['TotalLogReturn'].mean()
PPGmean = PPG_filtrato ['TotalLogReturn'].mean()
PRUmean = PRU_filtrato['TotalLogReturn'].mean()

###################################################
############### Datas for CAPM ####################
###################################################

MK_filtrato['Monthly_TotalLogReturn']

log_return_titoli = [BR_filtrato['TotalLogReturn'], BRKb_filtrato['TotalLogReturn'], CZR_filtrato['TotalLogReturn'], PPG_filtrato['TotalLogReturn'], PRU_filtrato['TotalLogReturn']]

medie_titoli = [BRmean, BRKbmean, CZRmean, PPGmean, PRUmean]

ER_mk = MKmean

Rf = 0.02
Rf_mensile = Rf/12
Rf_log = np.log(1+Rf_mensile)

nomi_titoli = ['MK','BR', 'BRKb', 'CZR', 'PPG', 'PRU']
nomi_titoli_grafico = ['BR', 'BRKb', 'CZR', 'PPG', 'PRU']