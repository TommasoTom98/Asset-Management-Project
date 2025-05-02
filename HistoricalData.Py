import pandas as pd
import numpy as np

########################################################################
############### Broadridge Financial Solutions Inc - BR.N ##############
########################################################################

BR = pd.read_csv("BR.N.csv")
#BR = BR.drop("Unnamed: 0",axis=1)
BR = BR.drop("Instrument",axis=1)
BR['Date'] = pd.to_datetime(BR['Date'], errors='coerce')
# Date - Time extraction
BR['Date'] = BR['Date'].dt.date
BR = BR.set_index("Date")
BR.index = pd.to_datetime(BR.index)  
BR["TotalLogReturn"] = (np.log(BR["1 Month Total Return"]/100+1))
numBR = BR["1 Month Total Return"].isna().sum().sum()
#print("Number of NaN: " + str(numBR))
BR = BR.iloc[numBR:]
# fill NaN date 30-03-2007
BR["Company Market Cap"].fillna(BR["Company Market Cap"][1], inplace=True)

########################################################################
################ Berkshire Hathaway Inc Class - BRKb.NB ################
########################################################################

BRKb = pd.read_csv("BRKb.N.csv")
BRKb = BRKb.drop("Unnamed: 0",axis=1)
BRKb = BRKb.drop("Instrument",axis=1)
BRKb['Date'] = pd.to_datetime(BRKb['Date'], errors='coerce')
# Date - Time extraction
BRKb['Date'] = BRKb['Date'].dt.date
BRKb = BRKb.set_index("Date")
BRKb.index = pd.to_datetime(BRKb.index)
BRKb["TotalLogReturn"] = (np.log(BRKb["1 Month Total Return"]/100+1))
numBRKb = BRKb["Company Market Cap"].isna().sum().sum()
#print("Number of NaN: " + str(numBRKb))
BRKb = BRKb.iloc[numBRKb:]
########################################################################
################# Caesars Entertainment In - CZR.OQc ###################
########################################################################

CZR = pd.read_csv("CZR.OQ.csv")
CZR = CZR.drop("Unnamed: 0",axis=1)
CZR = CZR.drop("Instrument",axis=1)
CZR['Date'] = pd.to_datetime(CZR['Date'], errors='coerce')
# Estrarre data e ora
CZR['Date'] = CZR['Date'].dt.date
CZR = CZR.set_index("Date")
CZR.index = pd.to_datetime(CZR.index)
CZR["TotalLogReturn"] = (np.log(CZR["1 Month Total Return"]/100+1))
numCZR = CZR["Company Market Cap"].isna().sum().sum()
print("Number of NaN: " + str(numCZR))
CZR = CZR.iloc[numCZR:]
# fill NaN date 30-09-2014 LogReturn
CZR.fillna(CZR["TotalLogReturn"].mean(), inplace = True)

########################################################################
####################### PPG Industries - PPG.N #########################
########################################################################

PPG = pd.read_csv("PPG.N.csv")
PPG = PPG.drop("Unnamed: 0",axis=1)
PPG = PPG.drop("Instrument",axis=1)
PPG['Date'] = pd.to_datetime(PPG['Date'], errors='coerce')
# Estrarre data e ora
PPG['Date'] = PPG['Date'].dt.date
PPG = PPG.set_index("Date")
PPG.index = pd.to_datetime(PPG.index)
PPG["TotalLogReturn"] = (np.log(PPG["1 Month Total Return"]/100+1))
numPPG = PPG["Company Market Cap"].isna().sum().sum()
#print("Number of NaN: " + str(numPPG))

########################################################################
##################### Prudential Financial - PRU.N #####################
########################################################################

PRU = pd.read_csv("PRU.N.csv", delimiter=",")
PRU = PRU.drop("Unnamed: 0",axis=1)
PRU = PRU.drop("Instrument",axis=1)
PRU['Date'] = pd.to_datetime(PRU['Date'], errors='coerce')
# Estrarre data e ora
PRU['Date'] = PRU['Date'].dt.date
PRU = PRU.set_index("Date")
PRU.index = pd.to_datetime(PRU.index)
PRU["TotalLogReturn"] = (np.log(PRU["1 Month Total Return"]/100+1))
numPRUG = PRU["Company Market Cap"].isna().sum().sum()
#print("Number of NaN: " + str(numPRUG))