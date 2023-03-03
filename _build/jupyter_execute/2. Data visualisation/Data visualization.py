#!/usr/bin/env python
# coding: utf-8

# # Data Visualization

# In[1]:


import pandas as pd 
import re

codes = """Image par défaut points	LA COURNEUVECLUB	01-03-23	+ 200
Image par défaut points	CHELLESCLUB	01-03-23	+ 200
Image par défaut points	HAUTSDEFRANCECLUB	01-03-23	+ 200
Image par défaut points	LA COURNEUVECLUB	01-03-23	+ 200
Image par défaut points	CHELLESCLUB	01-03-23	+ 200
Image par défaut points	HAUTSDEFRANCECLUB	01-03-23	+ 200
Image par défaut points	LA COURNEUVECLUB	01-03-23	+ 200
Image par défaut points	CHELLESCLUB	01-03-23	+ 200
Image par défaut points	HAUTSDEFRANCECLUB	01-03-23	+ 200
Image par défaut points	LA COURNEUVECLUB	01-03-23	+ 200
Image par défaut points	CHELLESCLUB	01-03-23	+ 200
Image par défaut points	HAUTSDEFRANCECLUB	01-03-23	+ 200
Image par défaut points	LA COURNEUVECLUB	01-03-23	+ 200
Image par défaut points	CHELLESCLUB	01-03-23	+ 200
Image par défaut points	HAUTSDEFRANCECLUB	01-03-23	+ 200
Image par défaut points	LA COURNEUVECLUB	01-03-23	+ 200
Image par défaut points	CHELLESCLUB	01-03-23	+ 200
Image par défaut points	HAUTSDEFRANCECLUB	01-03-23	+ 200
Image par défaut points	LA COURNEUVECLUB	01-03-23	+ 200
Image par défaut points	CHELLESCLUB	01-03-23	+ 200
Image par défaut points	HAUTSDEFRANCECLUB	01-03-23	+ 200
Image par défaut points	PLAINECOMMUNECLUB	01-03-23	+ 200
Image par défaut points	Bonus Bridgestone	01-03-23	+ 500
Image par défaut points	Bonus Sodexo	01-03-23	+ 500
Image par défaut points	PLAINECOMMUNECLUB	01-03-23	+ 200
Image par défaut points	Bonus Bridgestone	01-03-23	+ 500
Image par défaut points	Bonus Sodexo	01-03-23	+ 500
Image par défaut points	PLAINECOMMUNECLUB	01-03-23	+ 200
Image par défaut points	Bonus Bridgestone	01-03-23	+ 500
Image par défaut points	Bonus Sodexo	01-03-23	+ 500
Image par défaut points	PLAINECOMMUNECLUB	01-03-23	+ 200
Image par défaut points	Bonus Bridgestone	01-03-23	+ 500
Image par défaut points	Bonus Sodexo	01-03-23	+ 500
Image par défaut points	LYONCLUB	01-03-23	+ 200
Image par défaut points	ALPESMARITIMESCLUB	01-03-23	+ 200
Image par défaut points	LE BOURGETCLUB	01-03-23	+ 200
Image par défaut points	BonusAliexpress	01-03-23	+ 500
Image par défaut points	MONTIGNYLEBRETONNEUXCLUB	01-03-23	+ 200
Image par défaut points	Bonus Optic 2000	01-03-23	+ 500
Image par défaut points	BonusAliexpress	01-03-23	+ 500
Image par défaut points	MONTIGNYLEBRETONNEUXCLUB	01-03-23	+ 200
Image par défaut points	Bonus Optic 2000	01-03-23	+ 500
Image par défaut points	BonusAliexpress	01-03-23	+ 500
Image par défaut points	MONTIGNYLEBRETONNEUXCLUB	01-03-23	+ 200
Image par défaut points	Bonus Optic 2000	01-03-23	+ 500
Image par défaut points	BonusAliexpress	01-03-23	+ 500
Image par défaut points	MONTIGNYLEBRETONNEUXCLUB	01-03-23	+ 200
Image par défaut points	Bonus Optic 2000	01-03-23	+ 500
Image par défaut points	BonusAliexpress	01-03-23	+ 500
Image par défaut points	MONTIGNYLEBRETONNEUXCLUB	01-03-23	+ 200
Image par défaut points	Bonus Optic 2000	01-03-23	+ 500
Image par défaut points	BonusAliexpress	01-03-23	+ 500
Image par défaut points	MONTIGNYLEBRETONNEUXCLUB	01-03-23	+ 200
Image par défaut points	Bonus Optic 2000	01-03-23	+ 500
Image par défaut points	BonusAliexpress	01-03-23	+ 500
Image par défaut points	MONTIGNYLEBRETONNEUXCLUB	01-03-23	+ 200
Image par défaut points	Bonus Optic 2000	01-03-23	+ 500
Image par défaut points	SAINTOUENCLUB	01-03-23	+ 200
Image par défaut points	TAIARAPUOUESTCLUB	01-03-23	+ 200
Image par défaut points	POLYNESIEFRANÇAISECLUB	01-03-23	+ 200
Image par défaut points	GRANDPARISGRANDESTCLUB	01-03-23	+ 200
Image par défaut points	QY2024CLUB	01-03-23	+ 200
Image par défaut points	BOUCLENORDDESEINECLUB	01-03-23	+ 200
Image par défaut points	METROPOLEEUROPEENNEDELILLECLUB	01-03-23	+ 200
Image par défaut points	METROPOLEAIXMARSEILLEPROVENCECLUB	01-03-23	+ 200
Image par défaut points	GUYANCOURTCLUB	01-03-23	+ 200
Image par défaut points	METROPOLEEUROPEENNEDELILLECLUB	01-03-23	+ 200
Image par défaut points	METROPOLEAIXMARSEILLEPROVENCECLUB	01-03-23	+ 200
Image par défaut points	GUYANCOURTCLUB	01-03-23	+ 200
Image par défaut points	METROPOLEEUROPEENNEDELILLECLUB	01-03-23	+ 200
Image par défaut points	METROPOLEAIXMARSEILLEPROVENCECLUB	01-03-23	+ 200
Image par défaut points	GUYANCOURTCLUB	01-03-23	+ 200
Image par défaut points	METROPOLEEUROPEENNEDELILLECLUB	01-03-23	+ 200
Image par défaut points	METROPOLEAIXMARSEILLEPROVENCECLUB	01-03-23	+ 200
Image par défaut points	GUYANCOURTCLUB	01-03-23	+ 200
Image par défaut points	METROPOLEEUROPEENNEDELILLECLUB	01-03-23	+ 200
Image par défaut points	METROPOLEAIXMARSEILLEPROVENCECLUB	01-03-23	+ 200
Image par défaut points	GUYANCOURTCLUB	01-03-23	+ 200
Image par défaut points	METROPOLEEUROPEENNEDELILLECLUB	01-03-23	+ 200
Image par défaut points	METROPOLEAIXMARSEILLEPROVENCECLUB	01-03-23	+ 200
Image par défaut points	GUYANCOURTCLUB	01-03-23	+ 200
Image par défaut points	METROPOLEEUROPEENNEDELILLECLUB	01-03-23	+ 200
Image par défaut points	METROPOLEAIXMARSEILLEPROVENCECLUB	01-03-23	+ 200
Image par défaut points	GUYANCOURTCLUB	01-03-23	+ 200
Image par défaut points	METROPOLEEUROPEENNEDELILLECLUB	01-03-23	+ 200
Image par défaut points	METROPOLEAIXMARSEILLEPROVENCECLUB	01-03-23	+ 200
Image par défaut points	GUYANCOURTCLUB	01-03-23	+ 200
Image par défaut points	BonusAllianz	01-03-23	+ 500
Image par défaut points	Code bonus Enedis	01-03-23	+ 500
Image par défaut points	BonusAtos	01-03-23	+ 500
Image par défaut points	BonusAllianz	01-03-23	+ 500
Image par défaut points	Code bonus Enedis	01-03-23	+ 500
Image par défaut points	BonusAtos	01-03-23	+ 500
Image par défaut points	BonusAllianz	01-03-23	+ 500
Image par défaut points	Code bonus Enedis	01-03-23	+ 500
Image par défaut points	BonusAtos	01-03-23	+ 500
Image par défaut points	BonusAllianz	01-03-23	+ 500
Image par défaut points	Code bonus Enedis	01-03-23	+ 500
Image par défaut points	BonusAtos	01-03-23	+ 500
Image par défaut points	BonusAllianz	01-03-23	+ 500
Image par défaut points	Code bonus Enedis	01-03-23	+ 500
Image par défaut points	BonusAtos	01-03-23	+ 500
Image par défaut points	Bonus RANDSTAD	01-03-23	+ 500
Image par défaut points	BORDEAUXMETROPOLECLUB	01-03-23	+ 200
Image par défaut points	BonusFDJ	01-03-23	+ 500
Image par défaut points	Bonus RANDSTAD	01-03-23	+ 500
Image par défaut points	BORDEAUXMETROPOLECLUB	01-03-23	+ 200
Image par défaut points	BonusFDJ	01-03-23	+ 500
"""

df= pd.DataFrame(codes.split("\n"),columns=["text"])

def reduce_str(x):
    x = x.replace("Image par défaut points\t", "")
    x = x.replace("\t01-03-23\t+", "")
    return x

df['text'] = df['text'].apply(lambda x : reduce_str(x))

df1 = df['text'].str.extractall(r"(?P<Code>\w* ?\w+ )(?P<Points>\d\d\d)").groupby(level=0).last()

