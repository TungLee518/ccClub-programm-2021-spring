#!/usr/bin/env python
# coding: utf-8

# In[1]:


import requests
from bs4 import BeautifulSoup
import pandas as pd
import yfinance as yf
import ta
from ta import add_all_ta_features
from ta.utils import dropna
import plotly.express as px
from plotly.subplots import make_subplots
import plotly.graph_objects as go
import streamlit as st
import datetime
import json
from datetime import date
from datetime import timedelta
import io

from tweepy.auth import OAuthHandler
import tweepy
from textblob import  TextBlob
import pandas as pd
from pandas import Series
import numpy as np
import re as re1
import matplotlib.pyplot as plt
from itertools import  count

import itertools
from matplotlib.pylab import rcParams
from statsmodels.tsa.statespace.sarimax import SARIMAX
from tqdm import tqdm_notebook as tqdm
import _pickle as pickle
import requests
from io import StringIO

yf.pdr_override()
st.set_option('deprecation.showfileUploaderEncoding', False)
st.write("""
# Cryptocurrency Dashboard
""")
st.sidebar.header('User Input Parameters')
time = pd.to_datetime('now')
today_datetime = datetime.date.today()
df = pd.DataFrame({'kinds':['BTC-USD','ETH-USD']})

def subtract_one_month(t): 
    one_day = datetime.timedelta(days=1) 
    one_month_earlier = t - one_day 
    while one_month_earlier.month == t.month or one_month_earlier.day > t.day: 
     one_month_earlier -= one_day 
    return one_month_earlier 
def user_input_features():
    ticker = st.sidebar.radio( 'Which type of cryptocurrency?', df['kinds'][0:2])
    start_date = st.sidebar.text_input("Start Date", '2019-01-01')
    end_date = st.sidebar.text_input("End Date", f'{today_datetime}')
    buying_price = st.sidebar.number_input("Buying Price", value=30000, step=1000)
    balance = st.sidebar.number_input("Quantity", value=0, step=1000)
    file_buffer = st.sidebar.file_uploader("Choose a .csv or .xlxs file\n 2 columns are expected 'rate' and 'price'", type=['xlsx','csv'])
    return ticker, start_date, end_date, buying_price, balance, file_buffer
def cleanTwt_BTC(twt):
    twt=re1.sub('#bitcoin','bitcoin',twt)
    twt=re1.sub('#Bitcoin','Bitcoin',twt)
    twt=re1.sub('#[A-Za-z0-9]+','',twt)
    twt=re1.sub('\\n','',twt)
    twt=re1.sub('https?:\/\/\S+','',twt)
    return twt
def cleanTwt_ETH(twt):
    twt=re1.sub('#ethereum','ethereum',twt)
    twt=re1.sub('#Ethereum','ethereum',twt)
    twt=re1.sub('#[A-Za-z0-9]+','',twt)
    twt=re1.sub('\\n','',twt)
    twt=re1.sub('https?:\/\/\S+','',twt)
    return twt
symbol, start, end, buying_price, balance, file_buffer = user_input_features()
start = pd.to_datetime(start)
end = pd.to_datetime(end)
# Read data
data = yf.download(symbol,start,end)
data.columns = map(str.lower, data.columns)
df = data.copy()
df = ta.add_all_ta_features(df, "open", "high", "low", "close", "volume", fillna=True)
df_trends = df[['close','trend_sma_fast','trend_sma_slow','trend_ema_fast','trend_ema_slow',]]
df_momentum = df[['momentum_rsi', 'momentum_roc', 'momentum_tsi', 'momentum_uo', 'momentum_stoch', 'momentum_stoch_signal', 'momentum_wr', 'momentum_ao', 'momentum_kama']]
# Price
daily_price = data.close.iloc[-1]
portfolio = daily_price * balance
st.title(f"Streamlit and {symbol} :euro:")
st.header("DF last rows")
st.dataframe(data.tail())
st.header("DF today's value")
st.markdown(f'Daily {symbol} price: {daily_price}')
st.markdown(f'{symbol} price per quantity: {portfolio}')
if file_buffer is not None:
     file = pd.read_excel(file_buffer)
     file = pd.DataFrame(file)
     st.dataframe(file)
     weighted_rate = (file['price']*file['rate']).sum() / file['price'].sum()
     st.markdown(f'{symbol} portfolio price: {weighted_rate}')
     buying_price = weighted_rate
st.dataframe(data.tail(1))

st.header(f"Candlestick for {symbol}")
fig = go.Figure()
fig.add_trace(go.Candlestick(x=df.index,
                             open=df.open,
                             high=df.high,
                             low=df.low,
                             close=df.close,
                             visible=True,
                             name='Candlestick',))

if file_buffer is not None:
    fig.add_trace(
    go.Indicator(
        mode = "number+delta",
        value = daily_price,
        delta = {"reference": weighted_rate, 'relative':True},
        #title = {"text": "<span style='font-size:0.9em'>Daily Portfolio Performance</span>"},
        domain = {'y': [0.8, 1], 'x': [0.25, 0.75]},
        visible = True))

fig.add_shape(
     # Line Horizontal
         type="line",
         x0=start,
         y0=buying_price,
         x1=end,
         y1=buying_price,
         line=dict(
             color="black",
             width=1.5,
             dash="dash",
         ),
         visible = True,
)
for column in df_trends.columns.to_list():
    fig.add_trace(
    go.Scatter(x = df_trends.index,y = df_trends[column],name = column,))
fig.update_layout(height=800,width=1000, xaxis_rangeslider_visible=False)
st.plotly_chart(fig)
st.header(f"Trends for {symbol}")
fig = go.Figure()
for column in df_trends.columns.to_list():
    fig.add_trace(
    go.Scatter(x = df_trends.index,y = df_trends[column],name = column,))

button_all = dict(label = 'All',method = 'update',args = [{'visible': df_trends.columns.isin(df_trends.columns),'title': 'All','showlegend':True,}])
def create_layout_button(column):
    return dict(label = column,
                method = 'update',
                args = [{'visible': df_trends.columns.isin([column]),
                        'title': column,
                        'showlegend': True,
                        }])
fig.update_layout(updatemenus=[go.layout.Updatemenu(active = 0, buttons = ([button_all]) + list(df_trends.columns.map(lambda column: create_layout_button(column))))],)

fig.update_layout(
    xaxis=dict(
        rangeslider=dict(
            visible=True
        ),
        type="date"
    ))
fig.update_layout(height=800,width=1000,updatemenus=[dict(direction="down",pad={"r": 10, "t": 10},showactive=True,x=0,xanchor="left",y=1.15,yanchor="top",)],)

st.plotly_chart(fig)

st.header(f"Momentum Indicators for {symbol}")
m_info =st.radio("Any instruction for momentum indicators?",("NO","YES"))
if m_info == 'YES':
    st.header('相對強弱指標(RSI)')
    st.write('RSI 指標全名是 Relative Strength Index 又稱為相對強弱指標，主要是用來評估股市中「買賣盤雙方力道的強弱」，這是一種技術分析的動量指標，衡量近期價格變化的幅度，來評估股價超買或超賣情況。RSI指標(相對強弱指標)的區間為0～100，通常數值會以50作為多空的區隔：')
    st.write('RSI值>50：代表買方力道較強勁，平均上漲幅度大於平均跌幅。')
    st.write('RSI值<50：代表賣方力道較強勁，平均下跌幅度大於平均漲幅。')
    st.header('價格變動率(ROC)')
    st.write('ROC被歸類為動量指標或速度指標，透過變化率來衡量價格動量的強弱，它會以0線為中心上下波動，用來觀察出價格走勢，並識別超買和超賣的情況。')
    st.write('ROC會以0線基準，從中來判讀出不同運用方式，0線以上的正值表示向上的買入壓力或動能，而0線以下為負值表示賣出壓力或向下的動能。')
    st.header('真實強弱指數(TSI)')
    st.write('真正強度指數(TSI)是一個動量震盪器，其範圍在-100和+100之間，基值為0。當指數為正(指向看漲的市場傾向)時，動量為正，反之亦然。它是由William Blau開發的，由兩條線組成:TSI的指數移動平均線和指數移動平均線，稱為信號線。交易者可能會尋找以下五種情況中的任何一種:超買、超賣、中線交叉、背離和信號線交叉。這個指標常與其他信號結合使用。')
    st.header('終極震盪指標(UO)')
    st.write('終極震盪指標(UO)的直接簡化是它是一種衡量買入壓力的方法。當買入壓力強時，UO上升；當買入壓力弱時，UO下降。UO的計算考慮了三個單獨的時間周期。然後對那些時間周期進行加權。這是UO的最重要特徵，因為它在最短的時間周期內具有最大的權重，但仍在最長的時間周期內起作用。其目的是避免錯誤的背離。')
    st.write('技術分析師可能會發現他們需要調整指標的參數。對於終極震盪指標產生的交易信號，超買和超賣讀數至關重要。有時，金融商品沒有足夠的價格變動或波動來產生超買和超賣信號。有時，波動性很大的商品經常會產生超買和超賣的讀數。歷史分析和研究可以幫助找到合適的指標參數。')
    st.header('隨機指標(Stoch)')
    st.write('隨機指標（Stochastic Oscillator）指標通過比較收盤價格和價格的波動範圍，預測價格趨勢逆轉的時間。「隨機」一詞是指價格在一段時間內相對於其波動範圍的位置。')
    st.write('當股價趨勢上漲時，當日收盤價會傾向接近當日價格波動的最高價；')
    st.write('當股價趨勢下跌時，當日收盤價會傾向接近當日價格波動的最低價。')
    st.header('威廉指標(WR)')
    st.write('這個指標是一個振盪指標，是依股價的擺動點來度量股票／指數是否處於超買或超賣的現象。它衡量多空雙方創出的峰值（最高價）距每天收市價的距離與一定時間內（如7天、14天、28天等）的股價波動範圍的比例，以提供出股市趨勢反轉的訊號。')
    st.write('當威廉指數的值越小，市場越處買方主導，相反越接近零，市場由賣方主導，一般來說超過-20%的水平會視為超買（Overbought）的訊號，而-80%以下則被視為超賣（Oversold）訊號')
    st.header('動量震盪指標(AO)')
    st.write('動量震盪指標(AO)是用於衡量市場動量的指標。AO計算34個週期和5個週期的簡單移動平均線之間的差。使用簡單移動平均線每個柱線的中點來計算的，而不是使用收盤價計算的。AO通常用於確認趨勢或預測可能的逆轉。')
    st.header('考夫曼自適應移動平均線(KAMA)')
    st.write('考夫曼自適應性移動平均線(KAMA)是由Perry Kaufman開發的一種智能移動平均指標。強大的趨勢追蹤指標基於指數移動平均線(EMA)計算，對趨勢和波動都有反應。當噪音很低時，它會緊跟價格；當價格波動時，它會消除噪音。跟所有移動平均線一樣，KAMA可以用來觀察趨勢。價格穿過它表示一個趨勢發生變化。價格也可以從KAMA反彈，因此可以作為動態支撐和阻力。此指標經常與其他信號和分析技術結合使用。')
    
trace=[]
Headers = df_momentum.columns.values.tolist()
for i in range(9):
    trace.append(go.Scatter(x=df_momentum.index, name=Headers[i], y=df_momentum[Headers[i]]))
fig = make_subplots(rows=9, cols=1)
for i in range(9):
     fig.append_trace(trace[i],i+1,1)
fig.update_layout(height=2200, width=1000)
st.plotly_chart(fig)

today = pd.to_datetime('today')
interval = today - start
st_date = interval.days + 1
interval = today - end
end_date = interval.days



if symbol == 'BTC-USD':
    st.header("Bitcoin Active Addresses historical chart")
    url = "https://bitinfocharts.com/comparison/bitcoin-activeaddresses.html" 
    re = requests.get(url)
    soup = BeautifulSoup(re.content, "html.parser")
    script = soup.findAll("script")[4]
    script_str = script.string
    splitted = script_str.split('getElementById("container"),')[1].split(', {labels: ["Date",')[0]
    data = splitted.replace('new Date("', '').replace('")', '').replace('[', '').replace(']', '')
    date = []
    active_addresses = []
    table = data.split(',')
    if end_date == 0:
        table = table[(-2*st_date):]
    else:
        table = table[(-2*st_date):(-2*end_date)]
        
    for i in range(int(len(table)/2)):
        date.append(table[int(i*2)])
        try:
            active_addresses.append(int(table[int(i*2+1)]))
        except:
            active_addresses.append(None)

    date = pd.to_datetime(date, format='%Y/%m/%d')
    fig = go.Figure()
    fig = go.Figure([go.Scatter(x= date, y= active_addresses)])
    fig.update_layout(width=1000)
    st.plotly_chart(fig)
    
    st.header("Bitcoin Mining Profitability historical chart")
    url = "https://bitinfocharts.com/comparison/mining_profitability-btc-eth.html#log" 
    re = requests.get(url)
    soup = BeautifulSoup(re.content, "html.parser")
    script = soup.findAll("script")[4]
    script_str = script.string
    splitted = script_str.split('getElementById("container"),')[1].split(', {labels: ["Date",')[0]
    data = splitted.replace('new Date("', '').replace('")', '').replace('[', '').replace(']', '')
    date = []
    money_b = []
    money_e = []
    table = data.split(',')
    if end_date == 0:
        table = table[(-3*st_date):]
    else:
        table = table[(-3*st_date):(-3*end_date)]
    for i in range(int(len(table)/3)):
        date.append(table[int(i*3)])
        try:
            money_b.append(float(table[int(i*3+1)]))
        except:
            money_b.append(None)
    date = pd.to_datetime(date, format='%Y/%m/%d')

    fig = go.Figure()
    fig = go.Figure([go.Scatter(x= date, y= money_b)])
    fig.update_layout(width=1000)
    st.plotly_chart(fig)
    
    st.header("MVRV ratio")
    mvrv_info =st.radio("Any instruction for MVRV ratio?",("NO","YES"), key="1")
    if mvrv_info == 'YES':
        st.header('市場價值(MVRV)')
        st.write('市場價值（market capitalization/market value，下稱市值）與實現資本價值（realized capitalization/realized value，下稱實現價值）之比，即 MVRV 指標。')
        st.write('實現價值是以每一個供應單位按其最後一次在鏈上轉移時的價格（即最後一次交易時的價格）來計算的。可以被認爲是對一個數字貨幣資產的總成本的估計。這爲理解數字貨幣投資者的行爲提供了一個非常有價值的分析視角。')
        st.write('MVRV指標可以用來幫助衡量數字貨幣資產的市場頂部和底部，也可以用來更加深入理解數字貨幣資產的投資者行爲')
        
    url = "https://charts.woobull.com/bitcoin-mvrv-ratio/" 
    re = requests.get(url)
    soup = BeautifulSoup(re.content, "html.parser")
    script = soup.findAll("script")[4]
    script_str = script.string
    splitted = script_str.split('var mvrv = {')[1].split('line:')[0]
    data_x = splitted.split('],')[0].replace('x:', '').replace('[', '').replace(']', '').strip()
    data_y = splitted.split('],')[1].replace('y:', '').replace('[', '').replace(']', '').strip()
    date = data_x.split(',')
    mvrv = [float(i) for i in data_y.split(',')]
    if end_date == 0:
        date = date[-st_date:]
        mvrv = mvrv[-st_date:]
    else:
        date = date[-st_date:-end_date]
        mvrv = mvrv[-st_date:-end_date]
        
    date = pd.to_datetime(date, format="'%Y-%m-%d %H:%M'")
    fig = go.Figure()
    fig = go.Figure([go.Scatter(x= date, y= mvrv)])
    fig.update_layout(width=1000)
    st.plotly_chart(fig)
    
    st.header("NVT ratio")
    nvt_info =st.radio("Any instruction for NVT ratio?",("NO","YES"),key="2")
    if nvt_info == 'YES':
        st.header('NVT（Network Value to Transactions Ratio）')
        st.write('NVT（Network Value to Transactions Ratio）指標是計算比特幣市值與實際鏈上交易量的比值，此指標用來評估比特幣作為轉帳網路其價格的合理性（是否被高估或低估）。當 NVT 指標呈現向上趨勢變化，代表比特幣價格可能被高估，不利於之後的價格走勢；反之當該指標向下，代表比特幣價格可能被低估，有利於之後的價格走勢。')
        st.write('比特幣市值 / 鏈上交易量越高，比特幣越高估！')
        
    url = "https://charts.woobull.com/bitcoin-nvt-ratio/"
    re = requests.get(url)
    soup = BeautifulSoup(re.content, "html.parser")
    script = soup.findAll("script")[4]
    script_str = script.string
    splitted = script_str.split('var nvt = {')[1].split('line:')[0]
    data_x = splitted.split('],')[0].replace('x:', '').replace('[', '').replace(']', '').strip()
    data_y = splitted.split('],')[1].replace('y:', '').replace('[', '').replace(']', '').strip()
    date = data_x.split(',')
    nvt = [float(i) for i in data_y.split(',')]
    if end_date == 0:
        date = date[-st_date:]
        nvt = nvt[-st_date:]
    else:
        date = date[-st_date:-end_date]
        nvt = nvt[-st_date:-end_date]

    date = pd.to_datetime(date, format="'%Y-%m-%d %H:%M'")
    fig = go.Figure()
    fig = go.Figure([go.Scatter(x= date, y= nvt)])
    fig.update_layout(width=1000)
    st.plotly_chart(fig)
    
    st.header("top-100-richest-bitcoin-addresses")
    url = requests.get('https://bitinfocharts.com/top-100-richest-bitcoin-addresses.html')
    pd.read_html(url.text)[0]
    #news
    cryptocurrencies = ["BTCUSD"]
    crypto_keywords = ["Bitcoin"]
    #twit
    twitter=cleanTwt_BTC
    search_term_option='#bitcoin-fliter:retweets'
    
    
if symbol == 'ETH-USD':
    st.header("Ethereum Active Addresses historical chart")
    url = "https://bitinfocharts.com/comparison/activeaddresses-eth.html" 
    re = requests.get(url)
    soup = BeautifulSoup(re.content, "html.parser")
    script = soup.findAll("script")[4]
    script_str = script.string
    splitted = script_str.split('getElementById("container"),')[1].split(', {labels: ["Date",')[0]
    data = splitted.replace('new Date("', '').replace('")', '').replace('[', '').replace(']', '')
    date = []
    active_addresses = []
    table = data.split(',')
    if end_date == 0:
        table = table[(-2*st_date):]
    else:
        table = table[(-2*st_date):(-2*end_date)]
    for i in range(int(len(table)/2)):
        date.append(table[int(i*2)])
        try:
            active_addresses.append(int(table[int(i*2+1)]))
        except:
            active_addresses.append(None)

    date = pd.to_datetime(date, format='%Y/%m/%d')
    fig = go.Figure()
    fig = go.Figure([go.Scatter(x= date, y= active_addresses)])
    fig.update_layout(width=1000)
    st.plotly_chart(fig)
    
    st.header("Ethereum Mining Profitability historical chart")
    url = "https://bitinfocharts.com/comparison/mining_profitability-btc-eth.html#log" 
    re = requests.get(url)
    soup = BeautifulSoup(re.content, "html.parser")
    script = soup.findAll("script")[4]
    script_str = script.string
    splitted = script_str.split('getElementById("container"),')[1].split(', {labels: ["Date",')[0]
    data = splitted.replace('new Date("', '').replace('")', '').replace('[', '').replace(']', '')
    date = []
    money_b = []
    money_e = []
    table = data.split(',')
    if end_date == 0:
        table = table[(-3*st_date):]
    else:
        table = table[(-3*st_date):(-3*end_date)]
    for i in range(int(len(table)/3)):
        date.append(table[int(i*3)])
        try:
            money_e.append(float(table[int(i*3+2)]))
        except:
            money_e.append(None)
    date = pd.to_datetime(date, format='%Y/%m/%d')

    fig = go.Figure()
    fig = go.Figure([go.Scatter(x= date, y= money_e)])
    fig.update_layout(width=1000)
    st.plotly_chart(fig)
    #news
    cryptocurrencies = ["ETHUSD"]
    crypto_keywords = ["Ethereum"]
    #twit
    twitter=cleanTwt_ETH
    search_term_option='#ethereum-fliter:retweets'
    

    
############### news ###################

st.header("news sentiment analysis pie chart")
#要換掉key
sentiment_key = '28b174a8d2mshb1458ef331833afp1be746jsn91be9d6ad169'
websearch_key = '28b174a8d2mshb1458ef331833afp1be746jsn91be9d6ad169'
date_since = datetime.date.today() - timedelta(days=1)
news_output = {}
for crypto in crypto_keywords:

        news_output["{0}".format(crypto)] = {'description': [], 'title': []}
        url = "https://contextualwebsearch-websearch-v1.p.rapidapi.com/api/search/NewsSearchAPI"
        querystring = {"q":str(crypto),"pageNumber":"1","pageSize":"30","autoCorrect":"true","fromPublishedDate":date_since,"toPublishedDate":"null"}
        headers = {
            'x-rapidapi-key': websearch_key,
            'x-rapidapi-host': "contextualwebsearch-websearch-v1.p.rapidapi.com"
            }
        response = requests.request("GET", url, headers=headers, params=querystring)
        result = json.loads(response.text)
        for news in result['value']:
            news_output[crypto]["description"].append(news['description'])
            news_output[crypto]["title"].append(news['title'])

a = []
for crypto in crypto_keywords:
        news_output[crypto]['sentiment'] = {'pos': [], 'mid': [], 'neg': []}
        if len(news_output[crypto]['description']) > 0:
            for title in news_output[crypto]['title']:
                titles = re1.sub('[^A-Za-z0-9]+', ' ', title)
                import http.client
                conn = http.client.HTTPSConnection('text-sentiment.p.rapidapi.com')
                payload = 'text='+titles
                headers = {
                    'content-type': 'application/x-www-form-urlencoded',
                    'x-rapidapi-key': sentiment_key,
                    'x-rapidapi-host': 'text-sentiment.p.rapidapi.com'
                    }
                conn.request("POST", "/analyze", payload, headers)
                res = conn.getresponse()
                data = res.read()
                title_sentiment = json.loads(data)
                if not isinstance(title_sentiment, int):
                    if title_sentiment['pos'] == 1:
                        news_output[crypto]['sentiment']['pos'].append(title_sentiment['pos'])
                        a.append('pos')
                    elif title_sentiment['mid'] == 1:
                        news_output[crypto]['sentiment']['mid'].append(title_sentiment['mid'])
                        a.append('mid')
                    elif title_sentiment['neg'] == 1:
                        news_output[crypto]['sentiment']['neg'].append(title_sentiment['neg'])
                        a.append('neg')
                    else:
                        print(f'Sentiment not found for {crypto}')

count = {"sentiment":a}
for crypto in crypto_keywords:
        if len(news_output[crypto]['title']) > 0:

            news_output[crypto]['sentiment']['pos'] = len(news_output[crypto]['sentiment']['pos'])*100/len(news_output[crypto]['title'])
            news_output[crypto]['sentiment']['mid'] = len(news_output[crypto]['sentiment']['mid'])*100/len(news_output[crypto]['title'])
            news_output[crypto]['sentiment']['neg'] = len(news_output[crypto]['sentiment']['neg'])*100/len(news_output[crypto]['title'])

#sentiment bar
x=[]
for i in news_output[crypto]['sentiment']:
    x.append(i)
y=[]
for i in news_output[crypto]['sentiment']:
    y.append(int(news_output[crypto]['sentiment'][i]))
xy = {'sentiment':x,'percentage':y}
fig_pie = px.pie(xy, values='percentage', names='sentiment', title='sentiment of news')
st.plotly_chart(fig_pie)


#table

index=[]
for i in range(1,len(news_output[crypto]['title'])+1):
    index.append(i)
data={'title':news_output[crypto]['title'],'description':news_output[crypto]['description']}
df_news = pd.DataFrame(data,index)
#st.dataframe(df_news)

news_table =st.radio("Do you need news sentiment analysis table?",("NO","YES"))
if news_table == 'YES':
    st.header("news sentiment analysis table")
    st.table(df_news)
                       
                   
 


################## twitter #######################

onlydate = pd.DataFrame({'time': [subtract_one_month(today)]})                   
onlydate['just_date'] = onlydate['time'].dt.date
a = onlydate.iloc[0]['just_date']
st.header(f"{symbol} twitter sentiment analysis pie chart")
authenticate= tweepy.OAuthHandler('aF0FRzMGrIPIvNfTJ2VhtTvT9','nJLDYNxi3C0gf07UE83cLPq0FbbNmYFF5xcvNLsxgzo7GfW7c1')
authenticate.set_access_token('1398875017715224581-bZUoKjhMrKbLQNtPfhHJfGGpraxv9N','KMKZeAWtgdlFYSiHGmdEShjmIULgrybQQfkXPnXYJxOq2')
api=tweepy.API(authenticate,wait_on_rate_limit=True)
search_term=search_term_option
tweets=tweepy.Cursor(api.search,q=search_term,lang='en',since=f'{a}',tweet_mode='extended').items(200)
all_tweets=[tweet.full_text for tweet in tweets]
df_tw=pd.DataFrame(all_tweets,columns=['Tweets'])
def getSubjectivity(twt):
    return TextBlob(twt).sentiment.subjectivity
def getPloarity(twt):
    return TextBlob(twt).sentiment.polarity
def getSentiment(score):
    if score<0:
        return 'neg'
    elif score==0:
        return 'mid'
    else:
        return 'pos'
index=[]
for i in range(1,201):
    index.append(i)
df_tw.index=Series(index)
df_tw['Cleaned_Tweets']=df_tw['Tweets'].apply(twitter)
df_tw['Subjectivity']=df_tw['Cleaned_Tweets'].apply(getSubjectivity)
df_tw['Polarity']=df_tw['Cleaned_Tweets'].apply(getPloarity)
df_tw['Sentiment']=df_tw['Polarity'].apply(getSentiment)
sentiment_count = df_tw['Sentiment'].value_counts()
sentiment_count = pd.DataFrame({"Sentiment":sentiment_count.index,"Tweets":sentiment_count.values})
fig = px.pie(sentiment_count, values = "Tweets", names = "Sentiment")
st.plotly_chart(fig)

tweet_table = st.radio("Do you need tweets sentiment analysis table?",("NO","YES"))
if tweet_table == 'YES':
    st.header("tweets sentiment analysis table")
    st.table(df_tw)
if symbol == 'BTC-USD':    
    #indicator analysis
    indicator = {i : float(df[i][-1]) for i in df_momentum.columns.values if i != 'momentum_roc' and i != 'momentum_stoch_signal' and i != 'momentum_kama'}
    indicator['mvrv'] = mvrv[-1]
    indicator['nvt'] = nvt[-1]
    long_trend_nvt = sum(nvt[i] for i in range(-1,-91,-1))/90
    def color(val):
        if val == 'BUY':
            color = 'green'
        elif val == 'HOLD':
            color = 'orange'
        else:
            color = 'red'
        return 'color: %s' % color
    def rsi(val):
        if val >= 80:
            return "SELL"
        elif val <= 20:
            return "BUY"
        else:
            return "HOLD"
    def tsi(val):
        if val >= 25:
            return "SELL"
        elif val <= -25:
            return "BUY"
        else:
            return "HOLD"
    def uo(val):
        if val >= 70:
            return "SELL"
        elif val <= 30:
            return "BUY"
        else:
            return "HOLD"
    def stoch(val):
        if val >= 95:
            return "SELL"
        elif val <= 10:
            return "BUY"
        else:
            return "HOLD"
    def wr(val):
        if val >= -10:
            return "SELL"
        elif val <= -90:
            return "BUY"
        else:
            return "HOLD"
    def ao(val):
        if (df['momentum_ao'][-2] > 0) and (val < 0):
            return "SELL"
        elif (df['momentum_ao'][-2] < 0) and (val > 0):
            return "BUY"
        else:
            return "HOLD"
    def mvrv(val):
        if val >= 3.7:
            return "SELL"
        elif val <= 1:
            return "BUY"
        else:
            return "HOLD"
    def nvt(val):
        if val-long_trend_nvt > 2.2:
            return "SELL"
        elif val-long_trend_nvt < 1.6:
            return "BUY"
        else:
            return "HOLD"
    def news(pos, mid, neg): 
        if pos >= max([pos, mid, neg]):
            return "BUY"
        elif mid >= max([pos, mid, neg]):
            return "HOLD"
        else:
            return "SELL"
    def twitter(pos, mid, neg):
        if pos >= max([pos, mid, neg]):
            return "BUY"
        elif mid >= max([pos, mid, neg]):
            return "HOLD"
        else:
            return "SELL"
    analysis = pd.DataFrame({
        'INDICATORS': ['rsi', 'tsi', 'uo', 'stoch', 'wr', 'ao', 'mvrv', 'nvt', 'news', 'twitter'],
        'VALUE': [indicator["momentum_rsi"], indicator["momentum_tsi"],indicator["momentum_uo"],
                 indicator["momentum_stoch"], indicator["momentum_wr"], indicator["momentum_ao"],
                 indicator['mvrv'], indicator['nvt'], 'null', 'null'],
        'MIN': [0, -100, 0, 0, -100, 'N/A', 0, 0, 'null', 'null'],
        'MAX': [100, 100 ,100 ,100, 0, '-N/A', "N/A", 'N/A', 'null', 'null'],
        'SIGNAL': [rsi(indicator["momentum_rsi"]), tsi(indicator["momentum_tsi"]), 
                   uo(indicator["momentum_uo"]), stoch(indicator["momentum_stoch"]), 
                   wr(indicator["momentum_wr"]), ao(indicator["momentum_ao"]), 
                   mvrv(indicator['mvrv']), nvt(indicator['nvt']),
                   news(news_output[crypto]['sentiment']['pos'], news_output[crypto]['sentiment']['mid'],news_output[crypto]['sentiment']['neg']),
                   twitter(sentiment_count["Tweets"][0], sentiment_count["Tweets"][1], sentiment_count["Tweets"][2])]
    })
    buy , sell , hold = 0 , 0, 0 
    for i in analysis['SIGNAL']:
        if i == 'BUY':
            buy += 1
        elif i == 'HOLD':
            hold += 1
        else:
            sell += 1
    labels = ['BUY','HOLD','SELL']
    values = [buy, hold, sell]
    colors = ['green', 'orange', 'red']
    fig = go.Figure(data=[go.Pie(labels=labels, values=values, marker_colors=colors, hole=.5)])
    
    data1, data2 = st.beta_columns(2)
    data1.header("indicator analysis")
    data1.dataframe(analysis.style.applymap(color, subset=['SIGNAL']))
    data2.plotly_chart(fig)
    
    st.header("Price Prediction")
    site = "https://query1.finance.yahoo.com/v7/finance/download/BTC-USD?period1=1528156800&period2=1623110400&interval=1d&events=history&includeAdjustedClose=true"
    response = requests.get(site)
    bc = pd.read_csv(StringIO(response.text))
    bc['Date'] = pd.to_datetime(bc.Date)
    bc.set_index('Date', inplace=True)
    # Converting the data to a logarithmic scale
    bc_log = pd.DataFrame(np.log(bc.Close))
    # Splitting 80/20
    index = round(len(bc)*.80)

    train = bc_log.iloc[:index]
    test = bc_log.iloc[index:]
    # Fitting the model to the training set
    model = SARIMAX(train, 
                    order=(1, 0, 0), 
                    seasonal_order=(0,0,0,0), 
                    freq='D', 
                    enforce_stationarity=False, 
                    enforce_invertibility=False)
    output = model.fit()
    etrain = np.exp(train)

    # Values to test against the train set, see how the model fits
    predictions = output.get_prediction(start=pd.to_datetime('2020'), dynamic=False)
    pred        = np.exp(predictions.predicted_mean)
    forecast = pred
    actual_val = etrain.Close

    model = SARIMAX(bc_log, 
                    order=(1, 0, 0), 
                    seasonal_order=(0,0,0,0), 
                    freq='D', 
                    enforce_stationarity=False, 
                    enforce_invertibility=False)
    output = model.fit()
    # Getting the forecast of future values
    future = output.get_forecast(steps=30)

    # Transforming values back
    pred_fut = np.exp(future.predicted_mean)

    # Confidence interval for our forecasted values
    pred_conf = future.conf_int()

    # Transforming value back
    pred_conf = np.exp(pred_conf)
    # Plotting the prices up to the most recent
    ax = np.exp(bc_log).plot(label='Actual', figsize=(16,8))
    # Plottting the forecast
    pred_fut.plot(ax=ax, label='Future Vals')

    # Shading in the confidence interval
    ax.fill_between(pred_conf.index,
                    pred_conf.iloc[:, 0],
                    pred_conf.iloc[:, 1], color='k', alpha=.25)
    ax.set_xlabel('Date')
    ax.set_ylabel('Bitcoin Price')
    ax.set_xlim(['2020-01', '2021-07'])

    plt.title('Forecasted values')
    plt.legend()
    plt.savefig('fc_val.png')
    st.pyplot(plt)




# In[ ]:





