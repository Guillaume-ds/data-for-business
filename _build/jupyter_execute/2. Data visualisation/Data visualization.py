#!/usr/bin/env python
# coding: utf-8

# # Data Visualization

# In[1]:


import pandas as pd 
import re

codes_louis = """ 
AIRBNB 500pts /u/hempy94
AliExpressParis2024 500pts /u/PiBrickShop
AllianzParis2024 500pts /u/PiBrickShop
ALPESMARITIMESCLUB 500pts /u/hempy94
Atos-Partenaire-IT-Mondial 500?pts
AUVERGNERHONEALPESCLUB 500pts /u/hempy94
BORDEAUXCLUB 500pts /u/hempy94
BORDEAUXMETROPOLECLUB 500pts /u/hempy94
BOUCHESDURHONECLUB 500pts /u/hempy94
BOUCLENORDDESEINECLUB 500pts /u/hempy94
BridgestoneParis2024 500pts /u/PiBrickShop
CEPROCxCLUBPARIS2024 500pts /u/hempy94
CHATEAUFORTCLUB 500pts /u/hempy94
CHELLESCLUB 500pts /u/hempy94
CLUBFRANCEAPARIS 500pts /u/hempy94
CLUBxBREAKING 500pts /u/hempy94
COLOMBESCLUB 500pts /u/hempy94
DUGNYCLUB 500pts /u/hempy94
ELANCOURTCLUB 500pts /u/hempy94
ESTENSEMBLECLUB 500pts /u/hempy94
FDJParis2024 500pts /u/PiBrickShop
FFCXPARIS2024 ?pts Email
GIRONDECLUB 500pts /u/hempy94
GRANDPARISGRANDESTCLUB 500pts /u/hempy94
GUYANCOURTCLUB 500pts /u/hempy94
HAUTSDEFRANCECLUB 500pts /u/hempy94
HAUTSDESEINECLUB 500pts /u/hempy94
ILESAINTDENISCLUB 500pts /u/hempy94
IntelParis2024 1000pts pts
JESUISLEBOSS 1000pts pts
LA COURNEUVECLUB 500pts /u/hempy94
LE BOURGETCLUB 500pts /u/hempy94
lecoqsportifxparis2024 500pts /u/DUMDUM0173
LILLECLUB 500pts /u/hempy94
LOIREATLANTIQUECLUB 500pts /u/hempy94
LOIRECLUB 500pts /u/hempy94
LYONCLUB 500pts /u/hempy94
MAGNYLESHAMEAUXCLUB 500pts /u/hempy94
MARSEILLECLUB 500pts /u/hempy94
"""

codes_red = """ 
AIRBNB 500pts [/u/hempy94](https://www.reddit.com/u/hempy94/)

AliExpressParis2024 500pts [/u/PiBrickShop](https://www.reddit.com/u/PiBrickShop/)

AllianzParis2024 500pts [/u/PiBrickShop](https://www.reddit.com/u/PiBrickShop/)

ALPESMARITIMESCLUB 500pts [/u/hempy94](https://www.reddit.com/u/hempy94/)

Atos-Partenaire-IT-Mondial 500?pts

AUVERGNERHONEALPESCLUB 500pts [/u/hempy94](https://www.reddit.com/u/hempy94/)

BORDEAUXCLUB 500pts [/u/hempy94](https://www.reddit.com/u/hempy94/)

BORDEAUXMETROPOLECLUB 500pts [/u/hempy94](https://www.reddit.com/u/hempy94/)

BOUCHESDURHONECLUB 500pts [/u/hempy94](https://www.reddit.com/u/hempy94/)

BOUCLENORDDESEINECLUB 500pts [/u/hempy94](https://www.reddit.com/u/hempy94/)

BridgestoneParis2024 500pts [/u/PiBrickShop](https://www.reddit.com/u/PiBrickShop/)

CEPROCxCLUBPARIS2024 500pts [/u/hempy94](https://www.reddit.com/u/hempy94/)

CHATEAUFORTCLUB 500pts [/u/hempy94](https://www.reddit.com/u/hempy94/)

CHELLESCLUB 500pts [/u/hempy94](https://www.reddit.com/u/hempy94/)

CLUBFRANCEAPARIS 500pts [/u/hempy94](https://www.reddit.com/u/hempy94/)

CLUBxBREAKING 500pts [/u/hempy94](https://www.reddit.com/u/hempy94/)

COLOMBESCLUB 500pts [/u/hempy94](https://www.reddit.com/u/hempy94/)

DUGNYCLUB 500pts [/u/hempy94](https://www.reddit.com/u/hempy94/)

ELANCOURTCLUB 500pts [/u/hempy94](https://www.reddit.com/u/hempy94/)

ESTENSEMBLECLUB 500pts [/u/hempy94](https://www.reddit.com/u/hempy94/)

FDJParis2024 500pts [/u/PiBrickShop](https://www.reddit.com/u/PiBrickShop/)

FFCXPARIS2024 ?pts Email

GIRONDECLUB 500pts [/u/hempy94](https://www.reddit.com/u/hempy94/)

GRANDPARISGRANDESTCLUB 500pts [/u/hempy94](https://www.reddit.com/u/hempy94/)

GUYANCOURTCLUB 500pts [/u/hempy94](https://www.reddit.com/u/hempy94/)

HAUTSDEFRANCECLUB 500pts [/u/hempy94](https://www.reddit.com/u/hempy94/)

HAUTSDESEINECLUB 500pts [/u/hempy94](https://www.reddit.com/u/hempy94/)

ILESAINTDENISCLUB 500pts [/u/hempy94](https://www.reddit.com/u/hempy94/)

IntelParis2024 1000pts pts

JESUISLEBOSS 1000pts pts

LA COURNEUVECLUB 500pts [/u/hempy94](https://www.reddit.com/u/hempy94/)

LE BOURGETCLUB 500pts [/u/hempy94](https://www.reddit.com/u/hempy94/)

lecoqsportifxparis2024 500pts [/u/DUMDUM0173](https://www.reddit.com/u/DUMDUM0173/)

LILLECLUB 500pts [/u/hempy94](https://www.reddit.com/u/hempy94/)

LOIREATLANTIQUECLUB 500pts [/u/hempy94](https://www.reddit.com/u/hempy94/)

LOIRECLUB 500pts [/u/hempy94](https://www.reddit.com/u/hempy94/)

LYONCLUB 500pts [/u/hempy94](https://www.reddit.com/u/hempy94/)

MAGNYLESHAMEAUXCLUB 500pts [/u/hempy94](https://www.reddit.com/u/hempy94/)

MARSEILLECLUB 500pts [/u/hempy94](https://www.reddit.com/u/hempy94/)

MERCI 100pts [/u/josrun1](https://www.reddit.com/u/josrun1/)

METROPOLEAIXMARSEILLEPROVENCECLUB 500pts [/u/hempy94](https://www.reddit.com/u/hempy94/)

METROPOLEDELYONCLUB 500pts [/u/hempy94](https://www.reddit.com/u/hempy94/)

METROPOLEDENICECLUB 500pts [/u/hempy94](https://www.reddit.com/u/hempy94/)

METROPOLEDUGRANDPARISCLUB 500pts [/u/hempy94](https://www.reddit.com/u/hempy94/)

METROPOLEEUROPEENNEDELILLECLUB 500pts [/u/hempy94](https://www.reddit.com/u/hempy94/)

MONTIGNYLEBRETONNEUXCLUB 500pts [/u/hempy94](https://www.reddit.com/u/hempy94/)

NANTERRECLUB 500pts [/u/hempy94](https://www.reddit.com/u/hempy94/)

NANTESCLUB 500pts [/u/hempy94](https://www.reddit.com/u/hempy94/)

NANTESMETROPOLECLUB 500pts [/u/hempy94](https://www.reddit.com/u/hempy94/)

NICECLUB 500pts [/u/hempy94](https://www.reddit.com/u/hempy94/)

NORDCLUB 500pts [/u/hempy94](https://www.reddit.com/u/hempy94/)

NOUVELLEAQUITAINECLUB 500pts [/u/hempy94](https://www.reddit.com/u/hempy94/)

PAFXCLUBPARIS2024 500pts [/u/hempy94](https://www.reddit.com/u/hempy94/)

PARIS2024 1pts [/u/DUMDUM0173](https://www.reddit.com/u/DUMDUM0173/)

PARISCLUB 500pts [/u/hempy94](https://www.reddit.com/u/hempy94/)

PARISOUESTLADEFENSECLUB 500pts [/u/hempy94](https://www.reddit.com/u/hempy94/)

PARISVALLEEDELAMARNECLUB 500pts [/u/hempy94](https://www.reddit.com/u/hempy94/)

PAYSDELALOIRECLUB 500pts [/u/hempy94](https://www.reddit.com/u/hempy94/)

PLAINECOMMUNECLUB 500pts [/u/hempy94](https://www.reddit.com/u/hempy94/)

POLYNESIEFRANÇAISECLUB 500pts [/u/hempy94](https://www.reddit.com/u/hempy94/)

Promo\_agarik 1pts [/u/caxelair](https://www.reddit.com/u/caxelair/)

Pwc2024 500pts [/u/42195gg](https://www.reddit.com/u/42195gg/) & [/u/77AJ77](https://www.reddit.com/u/77AJ77/)

QUIZLCSXCLUB 500pts [/u/hempy94](https://www.reddit.com/u/hempy94/)

RANDSTADPARIS2024 500pts [/u/ASPE68](https://www.reddit.com/u/ASPE68/)

REGIONILEDEFRANCECLUB 500pts [/u/hempy94](https://www.reddit.com/u/hempy94/)

RHONECLUB 500pts [/u/hempy94](https://www.reddit.com/u/hempy94/)

SAINTCYRLECOLECLUB 500pts [/u/hempy94](https://www.reddit.com/u/hempy94/)

SAINTDENISCLUB 500pts [/u/hempy94](https://www.reddit.com/u/hempy94/)

SAINTETIENNECLUB 500pts [/u/hempy94](https://www.reddit.com/u/hempy94/)

SAINTETIENNEMETROPOLECLUB 500pts [/u/hempy94](https://www.reddit.com/u/hempy94/)

SAINTOUENCLUB 500pts [/u/hempy94](https://www.reddit.com/u/hempy94/)

SAINTQUENTINENYVELINESCLUB 500pts [/u/hempy94](https://www.reddit.com/u/hempy94/)

SAINTQUENTINENYVELINESCLUB 500pts [/u/icare21](https://www.reddit.com/u/icare21/)

SEINEETMARNECLUB 500pts [/u/hempy94](https://www.reddit.com/u/hempy94/)

SEINESAINTDENISCLUB 500pts [/u/hempy94](https://www.reddit.com/u/hempy94/)

SUDCLUB 500pts [/u/hempy94](https://www.reddit.com/u/hempy94/)

TAIARAPUOUESTCLUB 500pts [/u/hempy94](https://www.reddit.com/u/hempy94/)

TERRESDENVOLCLUB 500pts [/u/hempy94](https://www.reddit.com/u/hempy94/)

ToyotaParis2024 500pts [/u/PiBrickShop](https://www.reddit.com/u/PiBrickShop/)

TRAPPESCLUB 500pts [/u/hempy94](https://www.reddit.com/u/hempy94/)

VAIRESURMARNECLUB 500pts [/u/hempy94](https://www.reddit.com/u/hempy94/)

VERSAILLESCLUB 500pts [/u/hempy94](https://www.reddit.com/u/hempy94/)

VERSAILLESGRANDPARCCLUB 500pts [/u/hempy94](https://www.reddit.com/u/hempy94/)

VILLENEUVEDASCQCLUB 500pts [/u/hempy94](https://www.reddit.com/u/hempy94/)

VisaParis2024 500pts [/u/PiBrickShop](https://www.reddit.com/u/PiBrickShop/)

YVELINESCLUB 500pts [/u/hempy94](https://www.reddit.com/u/hempy94/)
"""

df_red = pd.DataFrame(codes_red.split("\n"),columns=["text"])
df_louis = pd.DataFrame(codes_louis.split("\n"),columns=["text"])

df1 = df_red['text'].str.extractall(r"(?P<Code>^\w+ )(?P<Points>\d\d\d)").groupby(level=0).last()
df2 = df_louis['text'].str.extractall(r"(?P<Code>^\w+ )(?P<Points>\d\d\d)").groupby(level=0).last()
df1.append(df2).drop_duplicates().reset_index(inplace=True)



