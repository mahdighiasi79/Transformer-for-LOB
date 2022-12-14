# imports:
import pandas as pd
import numpy as np

import requests
from bs4 import BeautifulSoup
import urllib3
urllib3.disable_warnings()

import aiohttp
import asyncio
from unsync import unsync
import tracemalloc

import datetime
import jdatetime
import calendar
import time

from persiantools.persiantools import characters
from IPython.display import clear_output

headers = {'User-Agent': 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_10_1) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/39.0.2171.95 Safari/537.36'}
################################################################################################################################################################################
################################################################################################################################################################################
def __Check_JDate_Validity__(date, key_word):
    try:
        if(len(date.split('-')[0])==4):
            date = jdatetime.date(year=int(date.split('-')[0]), month=int(date.split('-')[1]), day=int(date.split('-')[2]))
            date = f'{date.year:04}-{date.month:02}-{date.day:02}'
            return date
        else:
            print(f'Please enter valid {key_word} date in YYYY-MM-DD format')
    except:
        if(len(date)==10):
            print(f'Please enter valid {key_word} date')
            return
        else:
            print(f'Please enter valid {key_word} date in YYYY-MM-DD format')
            
################################################################################################################################################################################
################################################################################################################################################################################

def __Get_TSE_WebID__(stock):
    # search TSE function ------------------------------------------------------------------------------------------------------------
    def request(name):
        page = requests.get(f'http://www.tsetmc.com/tsev2/data/search.aspx?skey={name}', headers=headers)
        data = []
        for i in page.text.split(';') :
            try :
                i = i.split(',')
                data.append([i[0],i[1],i[2],i[7]])
            except :
                pass
        data = pd.DataFrame(data, columns=['Ticker','Name','WEB-ID','Active'])
        data['Name'] = data['Name'].apply(lambda x : characters.ar_to_fa(' '.join([i.strip() for i in x.split('\u200c')]).strip()))
        data['Ticker'] = data['Ticker'].apply(lambda x : characters.ar_to_fa(''.join(x.split('\u200c')).strip()))
        data['Name-Split'] = data['Name'].apply(lambda x : ''.join(x.split()).strip())
        data['Symbol-Split'] = data['Ticker'].apply(lambda x : ''.join(x.split()).strip())
        data['Active'] = pd.to_numeric(data['Active'])
        data = data.sort_values('Ticker')
        data = pd.DataFrame(data[['Name','WEB-ID','Name-Split','Symbol-Split']].values, columns=['Name','WEB-ID',
                            'Name-Split','Symbol-Split'], index=pd.MultiIndex.from_frame(data[['Ticker','Active']]))
        return data
    #---------------------------------------------------------------------------------------------------------------------------------
    if type(stock) != str:
        print('Please Enetr a Valid Ticker or Name!')
        return False
    # cleaning input search key
    stock = characters.ar_to_fa(''.join(stock.split('\u200c')).strip())
    first_name = stock.split()[0]
    stock = ''.join(stock.split())
    # search TSE and process:
    data = request(first_name)
    df_symbol = data[data['Symbol-Split'] == stock]
    df_name = data[data['Name-Split'] == stock]
    if len(df_symbol) > 0 :
        df_symbol = df_symbol.sort_index(level=1,ascending=False).drop(['Name-Split','Symbol-Split'], axis=1)
        return df_symbol
    elif len(df_name) > 0 :
        symbol = df_name.index[0][0]
        data = request(symbol)
        symbol = characters.ar_to_fa(''.join(symbol.split('\u200c')).strip())
        df_symbol = data[data.index.get_level_values('Ticker') == symbol]
        if len(df_symbol) > 0 :
            df_symbol = df_symbol.sort_index(level=1,ascending=False).drop(['Name-Split','Symbol-Split'], axis=1)
            return df_symbol
    print('Please Enetr a Valid Ticker or Name!')
    return False
################################################################################################################################################################################
################################################################################################################################################################################

def __Get_TSE_Sector_WebID__(sector_name):
    sector_list = ['??????????','???????? ??????','???????? ????????','???????? ??????????','??????????????','?????????????? ????????','?????????????? ????????','?????????????? ??????????','???????????? ?? ??????','?????????????? ?????? ????????','????????????',\
                   '?????????? ??????????','?????????????? ????????','?????????? ????????','???????????? ?????? ????????','?????????? ??????????????','??????????','?????? ?? ??????','?????? ???????? ????','?????????? ?????? ?????? ?? ??????','??????????',\
                   '????????????','??????????????','???????? ??????????','???????? ?? ????????????','??????????','???????? ?????? ????????','???????????? ??????????','????????','???????? ????????','?????? ?? ??????',\
                   '??????????????','????????','?????????? ???????????????? ????????','?????????? ????????','????????????','?????????????? ?? ????????????????','?????? ????????????','?????????????? ??????','???????? ?? ??????????????????']
    sector_web_id = [34408080767216529,19219679288446732,13235969998952202,62691002126902464,59288237226302898,69306841376553334,58440550086834602,30106839080444358,25766336681098389,\
     12331083953323969,36469751685735891,32453344048876642,1123534346391630,11451389074113298,33878047680249697,24733701189547084,20213770409093165,21948907150049163,40355846462826897,\
     54843635503648458,15508900928481581,3615666621538524,33626672012415176,65986638607018835,57616105980228781,70077233737515808,14651627750314021,34295935482222451,72002976013856737,\
     25163959460949732,24187097921483699,41867092385281437,61247168213690670,61985386521682984,4654922806626448,8900726085939949,18780171241610744,47233872677452574,65675836323214668,\
     59105676994811497]
    df_index_lookup = pd.DataFrame({'Sector':sector_list,'Web-ID':sector_web_id}).set_index('Sector')

    """index_list_url = 'http://tsetmc.com/Loader.aspx?Partree=151315&Flow=1'
    index_list_page = requests.get(index_list_url)
    soup = BeautifulSoup(index_list_page.content, 'html.parser')
    list_of_index = (soup.find_all('tbody')[0]).find_all('a')
    index_title = []
    index_webid = []
    for i in range(len(list_of_index)):
        index_title.append(list_of_index[i].text)
        index_webid.append(list_of_index[i].get('href').split('=')[-1])
    df_index_lookup = pd.DataFrame({'Sector':index_title,'Web-ID':index_webid}) 
    # Filter the lookup table to keep just industries
    df_index_lookup = df_index_lookup.iloc[:44]
    df_index_lookup.drop([16,18,19,26], axis=0, inplace=True)
    df_index_lookup['Sector'] = df_index_lookup['Sector'].apply(lambda x: (''.join([i for i in x if not i.isdigit()]).replace('-','')))
    df_index_lookup['Sector'] = df_index_lookup['Sector'].apply(lambda x: (((str(x).replace('??','??')).replace('??','??')).replace(' ??','')).strip())
    df_index_lookup = df_index_lookup.set_index('Sector')
    df_index_lookup['Web-ID'] = df_index_lookup['Web-ID'].apply(lambda x: int(x))"""
    # try search keyy with available look-up table and find web-id:
    try:
        sector_web_id = df_index_lookup.loc[sector_name]['Web-ID']
    except:
        sector_name = characters.fa_to_ar(sector_name)
        page = requests.get(f'https://www.google.com/search?q={sector_name} tsetmc ?????????????? ????????', headers=headers)
        code = page.text.split('http://www.tsetmc.com/Loader.aspx%3FParTree%3D15131J%26i%3D')[1]
        code = code.split('&')[0]
        # check google acquired code with reference table
        if(len(df_index_lookup[df_index_lookup['Web-ID'] == int(code)]) == 1):
            sector_web_id = int(code)
        else:
            print('Invalid sector name! Please try again with correct sector name!')
            return
    return sector_web_id   

################################################################################################################################################################################
################################################################################################################################################################################

def Get_Price_History(stock = '??????????', start_date = '1400-01-01', end_date='1401-01-01', ignore_date = False, adjust_price = False, show_weekday = False, double_date = False):
    """
    ???????????? ?????????? ???????? ???? ?????? ???? ???????????? ???????????????? ?????? ?????????? ???????? ?? ??????????
    ???????????? ???????????? ?????? ?????????? ???????? ???????? ???????? ???? ?????????? ???????? ?? ??????????
    ???????????? ?????????? ???????? ???????? ?????? ???????? ?? ???????????? ???????????? ???? ???????????? ??????????
    ???????????? ?????????? ?????????? ???????????? ?????????? ???? ?????????? ?????????? ???????????? ?????????? ???????????? ????????
    """
    # a function to get price data from a given page ----------------------------------------------------------------------------------
    def get_price_data(ticker_no,ticker,name, data_part):
        r = requests.get(f'http://members.tsetmc.com/tsev2/data/InstTradeHistory.aspx?i={ticker_no}&Top=999999&A=0', headers=headers)
        df_history=pd.DataFrame(r.text.split(';'))
        columns=['Date','High','Low','Final','Close','Open','Y-Final','Value','Volume','No']
        #split data into defined columns
        df_history[columns] = df_history[0].str.split("@",expand=True)
        # drop old column 0
        df_history.drop(columns=[0],inplace=True)
        df_history.dropna(inplace=True)
        df_history['Date']=pd.to_datetime(df_history['Date'])
        df_history['Ticker'] = ticker
        df_history['Name'] = name
        df_history['Part'] = data_part
        df_history = df_history.set_index('Date')
        return df_history
    # ----------------------------------------------------------------------------------------------------------------------------------
    # check date validity
    if(not ignore_date):
        start_date = __Check_JDate_Validity__(start_date,key_word="'START'")
        if(start_date==None):
            return
        end_date = __Check_JDate_Validity__(end_date,key_word="'END'")
        if(end_date==None):
            return
        start = jdatetime.date(year=int(start_date.split('-')[0]), month=int(start_date.split('-')[1]), day=int(start_date.split('-')[2]))
        end = jdatetime.date(year=int(end_date.split('-')[0]), month=int(end_date.split('-')[1]), day=int(end_date.split('-')[2]))
        if(start>end):
            print('Start date must be a day before end date!')
            return
    #---------------------------------------------------------------------------------------------------------------------------------------
    # find web-ids 
    ticker_no_df = __Get_TSE_WebID__(stock)
    if(type(ticker_no_df)==bool):
        return
    # create an empty dataframe:
    df_history = pd.DataFrame({},columns=['Date','High','Low','Final','Close','Open','Y-Final','Value','Volume','No','Ticker','Name','Part']).set_index('Date')
    # loop to get data from different pages of a ticker:
    for index, row in (ticker_no_df.reset_index()).iterrows():
        try:
            df_temp = get_price_data(ticker_no = row['WEB-ID'],ticker = row['Ticker'],name = row['Name'],data_part = index+1)
            df_history = pd.concat([df_history,df_temp])
        except:
            pass
    # sort index and reverse the order for more processes:
    df_history = df_history.sort_index(ascending=True)
    df_history = df_history.reset_index()
    # determining week days:
    df_history['Weekday']=df_history['Date'].dt.weekday
    df_history['Weekday'] = df_history['Weekday'].apply(lambda x: calendar.day_name[x])
    df_history['J-Date']=df_history['Date'].apply(lambda x: str(jdatetime.date.fromgregorian(date=x.date())))
    df_history = df_history.set_index('J-Date')
    # rearrange columns:
    df_history=df_history[['Date','Weekday','Y-Final','Open','High','Low','Close','Final','Volume','Value','No','Ticker','Name','Part']]
    cols = ['Y-Final','Open','High','Low','Close','Final','Volume','No','Value','Part']
    df_history[cols] = df_history[cols].apply(pd.to_numeric, axis=1)
    #----------------------------------------------------------------------------------------------------------------------
    # Y-Final for new part of data could be 0 or 1000, we need to replace them with yesterday's final price:
    df_history['Final(+1)'] = df_history['Final'].shift(+1)        # final prices shifted forward by one day
    df_history['temp'] = df_history.apply(lambda x: x['Y-Final'] if((x['Y-Final']!=0)and(x['Y-Final']!=1000)) 
                                          else (x['Y-Final'] if(pd.isnull(x['Final(+1)'])) else x['Final(+1)']),axis = 1)
    df_history['Y-Final'] = df_history['temp']
    df_history.drop(columns=['Final(+1)','temp'],inplace=True)
    #-----------------------------------------------------------------------------------------------------------------------
    for col in cols:
        df_history[col] = df_history[col].apply(lambda x: int(x)) # convert to int because we do not have less than Rial
    #--------------------------------------------------------------------------------------------------------------------
    # Adjust price data:
    if(adjust_price):
        df_history['COEF'] = (df_history['Y-Final'].shift(-1)/df_history['Final']).fillna(1.0)
        df_history['ADJ-COEF']=df_history.iloc[::-1]['COEF'].cumprod().iloc[::-1]
        df_history['Adj Open'] = (df_history['Open']*df_history['ADJ-COEF']).apply(lambda x: int(x))
        df_history['Adj High'] = (df_history['High']*df_history['ADJ-COEF']).apply(lambda x: int(x))
        df_history['Adj Low'] = (df_history['Low']*df_history['ADJ-COEF']).apply(lambda x: int(x))
        df_history['Adj Close'] = (df_history['Close']*df_history['ADJ-COEF']).apply(lambda x: int(x))
        df_history['Adj Final'] = (df_history['Final']*df_history['ADJ-COEF']).apply(lambda x: int(x))
        df_history.drop(columns=['COEF','ADJ-COEF'],inplace=True)
    if(not show_weekday):
        df_history.drop(columns=['Weekday'],inplace=True)
    if(not double_date):
        df_history.drop(columns=['Date'],inplace=True)
    df_history.drop(columns=['Y-Final'],inplace=True)
    # slice requested time window:
    if(not ignore_date):
        df_history = df_history[start_date:end_date]
    return df_history


################################################################################################################################################################################
################################################################################################################################################################################

def Get_RI_History(stock = '??????????', start_date = '1400-01-01', end_date='1401-01-01', ignore_date = False, show_weekday = False, double_date = False):
    """
    ???????????? ?????????? ?????????????? ??????????-?????????? ???? ?????? ???? ???????????? ???????????????? ?????? ?????????? ???????? ?? ??????????
    ???????????? ???????????? ?????? ?????????? ??????????-?????????? ???????? ???????? ???? ?????????? ???????? ?? ??????????
    ???????????? ?????????? ?????????? ???????????? ?????????? ???? ?????????? ?????????? ???????????? ?????????? ???????????? ????????
    """
    # a function to get ri data from a given page ----------------------------------------------------------------------------------
    def get_ri_data(ticker_no,ticker,name, data_part):
        r = requests.get(f'http://www.tsetmc.com/tsev2/data/clienttype.aspx?i={ticker_no}', headers=headers)
        df_RI_tab=pd.DataFrame(r.text.split(';'))
        # define columns
        columns=['Date','No_Buy_R','No_Buy_I','No_Sell_R','No_Sell_I','Vol_Buy_R','Vol_Buy_I','Vol_Sell_R','Vol_Sell_I','Val_Buy_R','Val_Buy_I','Val_Sell_R','Val_Sell_I']
        # split data into defined columns
        df_RI_tab[columns] = df_RI_tab[0].str.split(",",expand=True)
        # drop old column 0
        df_RI_tab.drop(columns=[0],inplace=True)
        df_RI_tab['Date']=pd.to_datetime(df_RI_tab['Date'])
        df_RI_tab['Ticker'] = ticker
        df_RI_tab['Name'] = name
        df_RI_tab['Part'] = data_part
        df_RI_tab = df_RI_tab.set_index('Date')
        return df_RI_tab
    # check date validity --------------------------------------------------------------------------------------------------------------
    if(not ignore_date):
        start_date = __Check_JDate_Validity__(start_date,key_word="'START'")
        if(start_date==None):
            return
        end_date = __Check_JDate_Validity__(end_date,key_word="'END'")
        if(end_date==None):
            return
        start = jdatetime.date(year=int(start_date.split('-')[0]), month=int(start_date.split('-')[1]), day=int(start_date.split('-')[2]))
        end = jdatetime.date(year=int(end_date.split('-')[0]), month=int(end_date.split('-')[1]), day=int(end_date.split('-')[2]))
        if(start>end):
            print('Start date must be a day before end date!')
            return
    #---------------------------------------------------------------------------------------------------------------------------------------
    # find web-ids 
    ticker_no_df = __Get_TSE_WebID__(stock)
    if(type(ticker_no_df)==bool):
        return
    # create an empty dataframe:   
    df_RI_tab = pd.DataFrame({},columns=['Date','No_Buy_R','No_Buy_I','No_Sell_R','No_Sell_I','Vol_Buy_R','Vol_Buy_I','Vol_Sell_R',
                                         'Vol_Sell_I','Val_Buy_R','Val_Buy_I','Val_Sell_R','Val_Sell_I','Ticker','Name','Part']).set_index('Date')
    # loop to get data from different pages of a ticker:
    for index, row in (ticker_no_df.reset_index()).iterrows():
        try:
            df_temp = get_ri_data(ticker_no = row['WEB-ID'],ticker = row['Ticker'],name = row['Name'],data_part = index+1)
            df_RI_tab = pd.concat([df_RI_tab,df_temp])
        except:
            pass
    # sort index and reverse the order for more processes:
    df_RI_tab = df_RI_tab.sort_index(ascending=True)
    df_RI_tab = df_RI_tab.reset_index()
    # determining week days:
    df_RI_tab['Weekday']=df_RI_tab['Date'].dt.weekday
    df_RI_tab['Weekday'] = df_RI_tab['Weekday'].apply(lambda x: calendar.day_name[x])
    df_RI_tab['J-Date']=df_RI_tab['Date'].apply(lambda x: str(jdatetime.date.fromgregorian(date=x.date())))
    df_RI_tab.set_index(df_RI_tab['J-Date'],inplace = True)
    df_RI_tab = df_RI_tab.set_index('J-Date')
    # rearrange columns:
    df_RI_tab=df_RI_tab[['Date','Weekday','No_Buy_R','No_Buy_I','No_Sell_R','No_Sell_I','Vol_Buy_R','Vol_Buy_I','Vol_Sell_R','Vol_Sell_I',
                         'Val_Buy_R','Val_Buy_I','Val_Sell_R','Val_Sell_I','Ticker','Name','Part']]
    cols = ['No_Buy_R','No_Buy_I','No_Sell_R','No_Sell_I','Vol_Buy_R','Vol_Buy_I','Vol_Sell_R','Vol_Sell_I','Val_Buy_R','Val_Buy_I','Val_Sell_R','Val_Sell_I']
    df_RI_tab[cols] = df_RI_tab[cols].apply(pd.to_numeric, axis=1)
    if(not show_weekday):
        df_RI_tab.drop(columns=['Weekday'],inplace=True)
    if(not double_date):
        df_RI_tab.drop(columns=['Date'],inplace=True)
    # slice requested time window:
    if(not ignore_date):
        df_RI_tab = df_RI_tab[start_date:end_date]
    return df_RI_tab

################################################################################################################################################################################
################################################################################################################################################################################

def Get_CWI_History(start_date = '1395-01-01', end_date='1400-12-29', ignore_date = False, just_adj_close = False, show_weekday = False, double_date = False):
    """
    ???????????? ?????????? ???????? ???? ???? ???????????? ???????????????? ?????? ?????????? ???????? ?? ??????????
    ???????????? ???????????? ?????? ?????????? ???????? ???? ???????? ???????? ???? ?????????? ???????? ?? ??????????
    ???????????? ?????????? ?????????? ???????????? ?????????? ???? ?????????? ?????????? ???????????? ?????????? ???????????? ????????
    """
    # check date validity --------------------------------------------------------------------------------------------------------------
    if(not ignore_date):
        start_date = __Check_JDate_Validity__(start_date,key_word="'START'")
        if(start_date==None):
            return
        end_date = __Check_JDate_Validity__(end_date,key_word="'END'")
        if(end_date==None):
            return
        start = jdatetime.date(year=int(start_date.split('-')[0]), month=int(start_date.split('-')[1]), day=int(start_date.split('-')[2]))
        end = jdatetime.date(year=int(end_date.split('-')[0]), month=int(end_date.split('-')[1]), day=int(end_date.split('-')[2]))
        if(start>end):
            print('Start date must be a day before end date!')
            return
    #---------------------------------------------------------------------------------------------------------------------------------------
    sector_web_id = 32097828799138957
    # get only close chart data for sector index:
    r_cl = requests.get(f'http://tsetmc.com/tsev2/chart/data/Index.aspx?i={sector_web_id}&t=value', headers=headers)
    df_sector_cl = pd.DataFrame(r_cl.text.split(';'))
    columns=['J-Date','Adj Close']
    df_sector_cl[columns] = df_sector_cl[0].str.split(",",expand=True)
    df_sector_cl.drop(columns=[0],inplace=True)
    df_sector_cl['J-Date'] = df_sector_cl['J-Date'].apply(lambda x: str(jdatetime.date(int(x.split('/')[0]),int(x.split('/')[1]),int(x.split('/')[2]))))
    df_sector_cl['Date'] = df_sector_cl['J-Date'].apply(lambda x: jdatetime.date(int(x[:4]),int(x[5:7]),int(x[8:])).togregorian())  
    df_sector_cl['Date'] = pd.to_datetime(df_sector_cl['Date'])
    df_sector_cl['Weekday']=df_sector_cl['Date'].dt.weekday
    df_sector_cl['Weekday'] = df_sector_cl['Weekday'].apply(lambda x: calendar.day_name[x])
    df_sector_cl = df_sector_cl.set_index('J-Date')
    df_sector_cl = df_sector_cl[['Date','Weekday','Adj Close']]
    df_sector_cl['Adj Close'] = pd.to_numeric(df_sector_cl['Adj Close'])
    if(not just_adj_close):
        r = requests.get(f'http://www.tsetmc.com/tsev2/chart/data/IndexFinancial.aspx?i={sector_web_id}&t=ph', headers=headers)
        df_sector = pd.DataFrame(r.text.split(';'))
        columns=['Date','High','Low','Open','Close','Volume','D']
        # split data into defined columns
        df_sector[columns] = df_sector[0].str.split(",",expand=True)
        df_sector.drop(columns=[0,'D'],inplace=True)
        df_sector['Date']=pd.to_datetime(df_sector['Date'])
        df_sector['J-Date']=df_sector['Date'].apply(lambda x: str(jdatetime.date.fromgregorian(date=x.date())))
        df_sector = df_sector.set_index('J-Date')
        df_sector.drop(columns=['Date'],inplace=True)
        # now concat:
        df_sector_cl = pd.concat([df_sector,df_sector_cl],axis=1).dropna()
        df_sector_cl = df_sector_cl[['Date','Weekday','Open','High','Low','Close','Adj Close','Volume']]
        cols = ['Open','High','Low','Close','Adj Close','Volume']
        df_sector_cl[cols] = df_sector_cl[cols].apply(pd.to_numeric, axis=1)
        df_sector_cl['Volume'] = df_sector_cl['Volume'].astype('int64')
    if(not show_weekday):
        df_sector_cl.drop(columns=['Weekday'],inplace=True)
    if(not double_date):
        df_sector_cl.drop(columns=['Date'],inplace=True)
    # slice requested time window:
    if(not ignore_date):
        df_sector_cl = df_sector_cl[start_date:end_date]    
    return df_sector_cl

################################################################################################################################################################################
################################################################################################################################################################################

def Get_EWI_History(start_date = '1395-01-01', end_date='1400-12-29', ignore_date = False, just_adj_close = True, show_weekday = False, double_date = False):
    """
    ???????????? ?????????? ???????? ???? ?????? ???? ???????????? ???????????????? ?????? ?????????? ???????? ?? ??????????
    ???????????? ???????????? ?????? ?????????? ???????? ???? ?????? ???????? ???????? ???? ?????????? ???????? ?? ??????????
    ???????????? ?????????? ?????????? ???????????? ?????????? ???? ?????????? ?????????? ???????????? ?????????? ???????????? ????????
    """
    # check date validity --------------------------------------------------------------------------------------------------------------
    if(not ignore_date):
        start_date = __Check_JDate_Validity__(start_date,key_word="'START'")
        if(start_date==None):
            return
        end_date = __Check_JDate_Validity__(end_date,key_word="'END'")
        if(end_date==None):
            return
        start = jdatetime.date(year=int(start_date.split('-')[0]), month=int(start_date.split('-')[1]), day=int(start_date.split('-')[2]))
        end = jdatetime.date(year=int(end_date.split('-')[0]), month=int(end_date.split('-')[1]), day=int(end_date.split('-')[2]))
        if(start>end):
            print('Start date must be a day before end date!')
            return
    #---------------------------------------------------------------------------------------------------------------------------------------
    sector_web_id = 67130298613737946
    # get only close chart data for sector index:
    r_cl = requests.get(f'http://tsetmc.com/tsev2/chart/data/Index.aspx?i={sector_web_id}&t=value', headers=headers)
    df_sector_cl = pd.DataFrame(r_cl.text.split(';'))
    columns=['J-Date','Adj Close']
    df_sector_cl[columns] = df_sector_cl[0].str.split(",",expand=True)
    df_sector_cl.drop(columns=[0],inplace=True)
    df_sector_cl['J-Date'] = df_sector_cl['J-Date'].apply(lambda x: str(jdatetime.date(int(x.split('/')[0]),int(x.split('/')[1]),int(x.split('/')[2]))))
    df_sector_cl['Date'] = df_sector_cl['J-Date'].apply(lambda x: jdatetime.date(int(x[:4]),int(x[5:7]),int(x[8:])).togregorian())  
    df_sector_cl['Date'] = pd.to_datetime(df_sector_cl['Date'])
    df_sector_cl['Weekday']=df_sector_cl['Date'].dt.weekday
    df_sector_cl['Weekday'] = df_sector_cl['Weekday'].apply(lambda x: calendar.day_name[x])
    df_sector_cl = df_sector_cl.set_index('J-Date')
    df_sector_cl = df_sector_cl[['Date','Weekday','Adj Close']]
    df_sector_cl['Adj Close'] = pd.to_numeric(df_sector_cl['Adj Close'])
    if(not just_adj_close):
        r = requests.get(f'http://www.tsetmc.com/tsev2/chart/data/IndexFinancial.aspx?i={sector_web_id}&t=ph', headers=headers)
        df_sector = pd.DataFrame(r.text.split(';'))
        columns=['Date','High','Low','Open','Close','Volume','D']
        # split data into defined columns
        df_sector[columns] = df_sector[0].str.split(",",expand=True)
        df_sector.drop(columns=[0,'D'],inplace=True)
        df_sector['Date']=pd.to_datetime(df_sector['Date'])
        df_sector['J-Date']=df_sector['Date'].apply(lambda x: str(jdatetime.date.fromgregorian(date=x.date())))
        df_sector = df_sector.set_index('J-Date')
        df_sector.drop(columns=['Date'],inplace=True)
        # now concat:
        df_sector_cl = pd.concat([df_sector,df_sector_cl],axis=1).dropna()
        df_sector_cl = df_sector_cl[['Date','Weekday','Open','High','Low','Close','Adj Close','Volume']]
        cols = ['Open','High','Low','Close','Adj Close','Volume']
        df_sector_cl[cols] = df_sector_cl[cols].apply(pd.to_numeric, axis=1)
        df_sector_cl['Volume'] = df_sector_cl['Volume'].astype('int64')
    if(not show_weekday):
        df_sector_cl.drop(columns=['Weekday'],inplace=True)
    if(not double_date):
        df_sector_cl.drop(columns=['Date'],inplace=True)
    # slice requested time window:
    if(not ignore_date):
        df_sector_cl = df_sector_cl[start_date:end_date]    
    return df_sector_cl

################################################################################################################################################################################
################################################################################################################################################################################

def __Get_Day_IntradayTrades__(ticker_no, j_date):
    #convert to desired Cristian data format
    year, month, day = j_date.split('-')
    date = jdatetime.date(int(year), int(month), int(day)).togregorian()
    date = f'{date.year:04}{date.month:02}{date.day:02}'
    # request and process
    headers = {'User-Agent': 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_10_1) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/39.0.2171.95 Safari/537.36'}
    page = requests.get(f'http://cdn.tsetmc.com/api/Trade/GetTradeHistory/{ticker_no}/{date}/false', headers=headers)
    df_intraday = (pd.DataFrame(page.json()['tradeHistory'])).iloc[:,2:6]
    df_intraday = df_intraday.sort_values(by='nTran')
    df_intraday.drop(columns=['nTran'],inplace=True)
    df_intraday.columns = ['Time','Volume','Price']
    df_intraday['Time'] = df_intraday['Time'].astype('str').apply(lambda x: '0'+x[0]+':'+x[1:3]+':'+x[3:]  if len(x)==5 else x[:2]+':'+x[2:4]+':'+x[4:])
    df_intraday['J-Date'] = j_date
    df_intraday = df_intraday.set_index(['J-Date','Time'])
    return df_intraday

################################################################################################################################################################################
################################################################################################################################################################################

def Get_IntradayTrades_History(stock = '????????????', start_date = '1400-09-15', end_date='1400-12-29', jalali_date = True, combined_datatime = False, show_progress = True):
    """
    ???????????? ?????????? ?????? ?????????????? ???? ?????? ???? ???????????? ???????????????? ?????? ?????????? ???????? ?? ??????????
    ?????? ?????? ???? ???????? ?????? ???? ?????? ???????? ???????? ?????????? ???? ?????????? ???????? ?? ?????????? ?????????? ?????????????? ????????
    ???????? ?????????? ?????????? ???? ?????????????? ???????? ?????? ???????? ???? ?????????? ?????? ???????? ?????????? ???????? ?????? ??????
    """
    # a function to get price data from a given page ----------------------------------------------------------------------------------
    failed_jdates = []
    def get_price_data_forintraday(ticker_no):
        r = requests.get(f'http://members.tsetmc.com/tsev2/data/InstTradeHistory.aspx?i={ticker_no}&Top=999999&A=0', headers=headers)
        df_history=pd.DataFrame(r.text.split(';'))
        columns=['Date','High','Low','Final','Close','Open','Y-Final','Value','Volume','No']
        #split data into defined columns
        df_history[columns] = df_history[0].str.split("@",expand=True)
        # drop old column 0
        df_history.drop(columns=[0],inplace=True)
        df_history.dropna(inplace=True)
        df_history['Date']=pd.to_datetime(df_history['Date'])
        df_history['WEB-ID'] = ticker_no
        df_history = df_history.set_index('Date')
        return df_history
    # check date validity --------------------------------------------------------------------------------------------------------------
    start_date = __Check_JDate_Validity__(start_date,key_word="'START'")
    if(start_date==None):
        return
    end_date = __Check_JDate_Validity__(end_date,key_word="'END'")
    if(end_date==None):
        return
    start = jdatetime.date(year=int(start_date.split('-')[0]), month=int(start_date.split('-')[1]), day=int(start_date.split('-')[2]))
    end = jdatetime.date(year=int(end_date.split('-')[0]), month=int(end_date.split('-')[1]), day=int(end_date.split('-')[2]))
    if(start>end):
        print('Start date must be a day before end date!')
        return
    #-----------------------------------------------------------------------------------------------------------------------------------
    # find web-ids 
    ticker_no_df = __Get_TSE_WebID__(stock)
    if(type(ticker_no_df)==bool):
        return
    # create an empty dataframe:
    df_history = pd.DataFrame({},columns=['Date','High','Low','Final','Close','Open','Y-Final','Value','Volume','No','WEB-ID']).set_index('Date')
    # loop to get data from different pages of a ticker:
    for index, row in (ticker_no_df.reset_index()).iterrows():
        try:
            df_temp = get_price_data_forintraday(ticker_no = row['WEB-ID'])
            df_history = pd.concat([df_history,df_temp])
        except:
            pass
    # sort index and reverse the order for more processes:
    df_history = df_history.sort_index(ascending=True)
    df_history = df_history.reset_index()
    df_history['J-Date']=df_history['Date'].apply(lambda x: str(jdatetime.date.fromgregorian(date=x.date())))
    df_history = df_history.set_index('J-Date')
    df_history = df_history[start_date:end_date]
    j_date_list = df_history.index.to_list()
    ticker_no_list = df_history['WEB-ID'].to_list()
    # now we have valid Jalali date list, let's loop over and concat:
    no_days = len(j_date_list)
    if(no_days==0):
        print('There is no trading day between start and end date for this stock!')
        return
    else:
        df_intraday = pd.DataFrame(columns=['J-Date','Time','Volume','Price']).set_index(['J-Date','Time'])
        day_counter = 1
        for j_date in j_date_list:
            try:
                df_intraday = pd.concat([df_intraday,__Get_Day_IntradayTrades__(ticker_no_list[day_counter-1], j_date)], axis=0)
            except:
                failed_jdates.append(j_date)
            if(show_progress):
                clear_output(wait=True)
                print('Progress : ', f'{round((day_counter)/no_days*100,1)} %')
            day_counter+=1
    # other settings -------------------------------------------------------------------------------------------------------------
    if(jalali_date):
        if(combined_datatime):
            df_intraday = df_intraday.reset_index()
            # add date to data frame:
            df_intraday['Date'] = df_intraday['J-Date'].apply(lambda x: jdatetime.date(int(x[:4]),int(x[5:7]),int(x[8:])).togregorian())
            df_intraday['DateTime'] = pd.to_datetime(df_intraday['Date'].astype(str) +' '+ df_intraday['Time'].astype(str))
            print('Combining Jalali date and time takes a few more seconds!')
            df_intraday['J-DateTime'] = df_intraday['DateTime'].apply(lambda x: str(jdatetime.datetime.fromgregorian(datetime=x)))
            df_intraday.drop(columns=['DateTime','Date','J-Date','Time'],inplace=True)
            df_intraday = df_intraday.set_index(['J-DateTime'])
    else:
        if(combined_datatime):
            df_intraday = df_intraday.reset_index()
            df_intraday['Date'] = df_intraday['J-Date'].apply(lambda x: jdatetime.date(int(x[:4]),int(x[5:7]),int(x[8:])).togregorian())
            df_intraday['DateTime'] = pd.to_datetime(df_intraday['Date'].astype(str) +' '+ df_intraday['Time'].astype(str))
            df_intraday.drop(columns=['Date','J-Date','Time'],inplace=True)
            df_intraday = df_intraday.set_index(['DateTime'])
        else:
            df_intraday = df_intraday.reset_index()
            df_intraday['Date'] = df_intraday['J-Date'].apply(lambda x: jdatetime.date(int(x[:4]),int(x[5:7]),int(x[8:])).togregorian())
            df_intraday.drop(columns=['J-Date'],inplace=True)
            df_intraday = df_intraday.set_index(['Date','Time'])
    df_intraday[['Volume','Price']] = df_intraday[['Volume','Price']].astype('int64')
    # warning for failed dates:
    if(len(failed_jdates)!=0):
        print('WARNING: The following days data is not available on TSE website, even if those are trading days!')
        print(failed_jdates)
    return df_intraday

################################################################################################################################################################################
################################################################################################################################################################################

def Get_USD_RIAL(start_date = '1395-01-01', end_date='1400-12-29', ignore_date = False, show_weekday = False, double_date = False):
    # check date validity --------------------------------------------------------------------------------------------------------------
    if(not ignore_date):
        start_date = __Check_JDate_Validity__(start_date,key_word="'START'")
        if(start_date==None):
            return
        end_date = __Check_JDate_Validity__(end_date,key_word="'END'")
        if(end_date==None):
            return
        start = jdatetime.date(year=int(start_date.split('-')[0]), month=int(start_date.split('-')[1]), day=int(start_date.split('-')[2]))
        end = jdatetime.date(year=int(end_date.split('-')[0]), month=int(end_date.split('-')[1]), day=int(end_date.split('-')[2]))
        if(start>end):
            print('Start date must be a day before end date!')
            return
    #---------------------------------------------------------------------------------------------------------------------------------------
    r = requests.get('https://platform.tgju.org/fa/tvdata/history?symbol=PRICE_DOLLAR_RL&resolution=1D', headers=headers)
    df_data = r.json()
    df_data = pd.DataFrame({'Date':df_data['t'],'Open':df_data['o'],'High':df_data['h'],'Low':df_data['l'],'Close':df_data['c'],})
    df_data['Date'] = df_data['Date'].apply(lambda x: datetime.datetime.fromtimestamp(x))
    df_data = df_data.set_index('Date')
    df_data.index = df_data.index.to_period("D")
    df_data.index=df_data.index.to_series().astype(str)
    df_data = df_data.reset_index()
    df_data['Date'] = pd.to_datetime(df_data['Date'])
    df_data['Weekday']=df_data['Date'].dt.weekday
    df_data['Weekday'] = df_data['Weekday'].apply(lambda x: calendar.day_name[x])
    df_data['J-Date']=df_data['Date'].apply(lambda x: str(jdatetime.date.fromgregorian(date=x.date())))
    df_data = df_data.set_index('J-Date')
    df_data=df_data[['Date','Weekday','Open','High','Low','Close']]
    if(not show_weekday):
        df_data.drop(columns=['Weekday'],inplace=True)
    if(not double_date):
        df_data.drop(columns=['Date'],inplace=True)
    if(not ignore_date):
        df_data = df_data[start_date:end_date]
    return df_data

################################################################################################################################################################################
################################################################################################################################################################################

def __Get_Day_MarketClose_BQ_SQ__(ticker_no, j_date):
    #convert to desired Cristian data format
    year, month, day = j_date.split('-')
    date = jdatetime.date(int(year), int(month), int(day)).togregorian()
    date = f'{date.year:04}{date.month:02}{date.day:02}'
    # get day upper and lower band prices:
    page = requests.get(f'http://cdn.tsetmc.com/api/MarketData/GetStaticThreshold/{ticker_no}/{date}', headers=headers)
    df_ub_lb = pd.DataFrame(page.json()['staticThreshold'])
    day_ub = df_ub_lb.iloc[-1]['psGelStaMax']    # day upper band price
    day_lb = df_ub_lb.iloc[-1]['psGelStaMin']    # day lower band price
    # get LOB data:
    page = requests.get(f'http://cdn.tsetmc.com/api/BestLimits/{ticker_no}/{date}', headers=headers)
    data = pd.DataFrame(page.json()['bestLimitsHistory'])
    # find last orders before 12:30:00 (market close)
    time = 123000
    bq, sq, bq_percap, sq_percap = 0.0, 0.0, 0.0, 0.0
    while(time>122900):
        end_lob = data[data['hEven'] == time]
        end_lob = end_lob.sort_values(by='number', ascending=True).iloc[:1,5:-1]
        end_lob.columns = ['Vol_Buy','No_Buy','Price_Buy','Price_Sell','No_Sell','Vol_Sell']
        end_lob = end_lob[['No_Sell','Vol_Sell','Price_Sell','Price_Buy','No_Buy','Vol_Buy']]
        if(len(end_lob)==0): #go one second back and try again
            time-=1
            if(int(str(time)[-2:])>59):
                a = int(str(time)[:-2]+'59')
        else:
            # check buy and sell queue and do calculations
            if(end_lob.iloc[0]['Price_Sell'] == day_lb):
                sq = day_lb * end_lob.iloc[0]['Vol_Sell']
                sq_percap = sq/end_lob.iloc[0]['No_Sell']
            if(end_lob.iloc[0]['Price_Buy'] == day_ub):
                bq = day_ub * end_lob.iloc[0]['Vol_Buy']
                bq_percap = bq/end_lob.iloc[0]['No_Buy']
            break
    df_sq_bq = pd.DataFrame({'J-Date':[j_date],'Day_UL':[int(day_lb)],'Day_LL':[int(day_ub)], 'Time':[str(time)[:2]+':'+str(time)[2:4]+':'+str(time)[-2:]],\
                             'BQ_Value':[int(bq)],'SQ_Value':[int(sq)],'BQPC':[int(round(bq_percap,0))], 'SQPC':[int(round(sq_percap,0))]})
    df_sq_bq = df_sq_bq.set_index('J-Date')
    return df_sq_bq

################################################################################################################################################################################
################################################################################################################################################################################

def Get_Queue_History(stock = '????????????', start_date = '1400-09-15', end_date='1400-12-29', show_per_capita = True, show_weekday = False, double_date = False, show_progress = True):
    """
    ???????????? ???????? ???? ???????? ???? ???????? ???? ?????? ???? ???????? ???????? ?????? ???????????? ???? ???????????? ???????????????? ?????? ?????????? ???????? ?? ??????????
    ?????? ?????? ???? ???????? ?????? ???? ?????? ???????? ???????? ?????????? ???? ?????????? ???????? ?? ?????????? ?????????? ?????????????? ????????
    """
    # a function to get price data from a given page ----------------------------------------------------------------------------------
    def get_price_data_forintraday(ticker_no):
        r = requests.get(f'http://members.tsetmc.com/tsev2/data/InstTradeHistory.aspx?i={ticker_no}&Top=999999&A=0', headers=headers)
        df_history=pd.DataFrame(r.text.split(';'))
        columns=['Date','High','Low','Final','Close','Open','Y-Final','Value','Volume','No']
        #split data into defined columns
        df_history[columns] = df_history[0].str.split("@",expand=True)
        # drop old column 0
        df_history.drop(columns=[0],inplace=True)
        df_history.dropna(inplace=True)
        df_history['Date']=pd.to_datetime(df_history['Date'])
        df_history['WEB-ID'] = ticker_no
        df_history = df_history.set_index('Date')
        return df_history
    # check date validity --------------------------------------------------------------------------------------------------------------
    start_date = __Check_JDate_Validity__(start_date,key_word="'START'")
    if(start_date==None):
        return
    end_date = __Check_JDate_Validity__(end_date,key_word="'END'")
    if(end_date==None):
        return
    start = jdatetime.date(year=int(start_date.split('-')[0]), month=int(start_date.split('-')[1]), day=int(start_date.split('-')[2]))
    end = jdatetime.date(year=int(end_date.split('-')[0]), month=int(end_date.split('-')[1]), day=int(end_date.split('-')[2]))
    if(start>end):
        print('Start date must be a day before end date!')
        return
    #-----------------------------------------------------------------------------------------------------------------------------------
    # find web-ids 
    ticker_no_df = __Get_TSE_WebID__(stock)
    if(type(ticker_no_df)==bool):
        return
    # create an empty dataframe:
    df_history = pd.DataFrame({},columns=['Date','High','Low','Final','Close','Open','Y-Final','Value','Volume','No','WEB-ID']).set_index('Date')
    # loop to get data from different pages of a ticker:
    for index, row in (ticker_no_df.reset_index()).iterrows():
        try:
            df_temp = get_price_data_forintraday(ticker_no = row['WEB-ID'])
            df_history = pd.concat([df_history,df_temp])
        except:
            pass
    # sort index and reverse the order for more processes:
    df_history = df_history.sort_index(ascending=True)
    df_history = df_history.reset_index()
    df_history['J-Date']=df_history['Date'].apply(lambda x: str(jdatetime.date.fromgregorian(date=x.date())))
    df_history['Weekday']=df_history['Date'].dt.weekday
    df_history['Weekday'] = df_history['Weekday'].apply(lambda x: calendar.day_name[x])
    df_history = df_history.set_index('J-Date')
    df_history = df_history[start_date:end_date]
    j_date_list = df_history.index.to_list()
    ticker_no_list = df_history['WEB-ID'].to_list()
    # now we have valid Jalali date list, let's loop over and concat:
    no_days = len(j_date_list)
    if(no_days==0):
        print('There is no trading day between start and end date for this stock!')
        return
    else:
        df_bq_sq_val = pd.DataFrame(columns=['J-Date','Day_UL','Day_LL','Time','BQ_Value','SQ_Value','BQPC','SQPC']).set_index(['J-Date'])
        day_counter = 1
        for j_date in j_date_list:
            df_bq_sq_val = pd.concat([df_bq_sq_val,__Get_Day_MarketClose_BQ_SQ__(ticker_no_list[day_counter-1], j_date)], axis=0)
            if(show_progress):
                clear_output(wait=True)
                print('Progress : ', f'{round((day_counter)/no_days*100,1)} %')
            day_counter+=1
    df_bq_sq_val['Value'] = df_history['Value'].tolist()
    df_bq_sq_val['Date'] = df_history['Date'].tolist()
    df_bq_sq_val['Weekday'] = df_history['Weekday'].tolist()
    df_bq_sq_val['Value'] = df_bq_sq_val['Value'].apply(pd.to_numeric).astype('int64')
    # re-arrange columns order:
    df_bq_sq_val = df_bq_sq_val[['Date','Weekday','Day_UL','Day_LL','Value','Time','BQ_Value','SQ_Value','BQPC','SQPC']]
    if(not show_per_capita):
        df_bq_sq_val.drop(columns=['BQPC','SQPC'],inplace=True)
    if(not show_weekday):
        df_bq_sq_val.drop(columns=['Weekday'],inplace=True)
    if(not double_date):
        df_bq_sq_val.drop(columns=['Date'],inplace=True)
    return df_bq_sq_val

################################################################################################################################################################################
################################################################################################################################################################################

def __Get_Day_LOB__(ticker_no, j_date):
    #convert to desired Cristian data format
    year, month, day = j_date.split('-')
    date = jdatetime.date(int(year), int(month), int(day)).togregorian()
    date = f'{date.year:04}{date.month:02}{date.day:02}'
    # get day upper and lower band prices:
    page = requests.get(f'http://cdn.tsetmc.com/api/MarketData/GetStaticThreshold/{ticker_no}/{date}', headers=headers)
    df_ub_lb = pd.DataFrame(page.json()['staticThreshold'])
    day_ub = df_ub_lb.iloc[-1]['psGelStaMax']    # day upper band price
    day_lb = df_ub_lb.iloc[-1]['psGelStaMin']    # day lower band price
    # get LOB data:
    page = requests.get(f'http://cdn.tsetmc.com/api/BestLimits/{ticker_no}/{date}',headers=headers)
    data = pd.DataFrame(page.json()['bestLimitsHistory'])
    data.drop(columns=['idn','dEven','refID','insCode'],inplace=True)
    data = data.sort_values(['hEven','number'], ascending = (True, True))
    data = data[(data['hEven']>=84500) & (data['hEven']<123000)]  # filter out 8:30 to 12:35
    data.columns = ['Time','Depth','Buy_Vol','Buy_No','Buy_Price','Sell_Price','Sell_No','Sell_Vol']
    data['J-Date'] = j_date
    data['Date'] = date
    data['Date'] = pd.to_datetime(data['Date'])
    # re-arrange columns:
    data['Time'] = data['Time'].astype('str').apply(lambda x :datetime.time(hour=int(x[0]),minute=int(x[1:3]),second=int(x[3:])) if len(x)==5\
                                                    else datetime.time(hour=int(x[:2]),minute=int(x[2:4]),second=int(x[4:])))
    data['Day_UL'] = day_ub
    data['Day_LL'] = day_lb
    data = data[['J-Date','Time','Depth','Sell_No','Sell_Vol','Sell_Price','Buy_Price','Buy_Vol','Buy_No','Day_LL','Day_UL','Date']]
    data = data.set_index(['J-Date','Time','Depth'])
    return data

def Get_IntradayOB_History(stock = '??????????', start_date = '1400-08-01', end_date='1400-08-01', jalali_date = True, combined_datatime = False, show_progress = True):
    """
    ???????????? ?????????????? ???????? ???????????? ???????? ?????? ???????? ???? ???????? ???? ???????????? ???????????????? ?????? ?????????? ???????? ?? ??????????
    ?????? ?????? ???? ???????? ?????? ???? ?????? ???????? ???????? ?????????? ???? ?????????? ???????? ?? ?????????? ?????????? ?????????????? ????????
    """
# a function to get price data from a given page ----------------------------------------------------------------------------------
    def get_price_data_forintraday(ticker_no):
        r = requests.get(f'http://members.tsetmc.com/tsev2/data/InstTradeHistory.aspx?i={ticker_no}&Top=999999&A=0', headers=headers)
        df_history=pd.DataFrame(r.text.split(';'))
        columns=['Date','High','Low','Final','Close','Open','Y-Final','Value','Volume','No']
        #split data into defined columns
        df_history[columns] = df_history[0].str.split("@",expand=True)
        # drop old column 0
        df_history.drop(columns=[0],inplace=True)
        df_history.dropna(inplace=True)
        df_history['Date']=pd.to_datetime(df_history['Date'])
        df_history['WEB-ID'] = ticker_no
        df_history = df_history.set_index('Date')
        return df_history
    # check date validity --------------------------------------------------------------------------------------------------------------
    start_date = __Check_JDate_Validity__(start_date,key_word="'START'")
    if(start_date==None):
        return
    end_date = __Check_JDate_Validity__(end_date,key_word="'END'")
    if(end_date==None):
        return
    start = jdatetime.date(year=int(start_date.split('-')[0]), month=int(start_date.split('-')[1]), day=int(start_date.split('-')[2]))
    end = jdatetime.date(year=int(end_date.split('-')[0]), month=int(end_date.split('-')[1]), day=int(end_date.split('-')[2]))
    if(start>end):
        print('Start date must be a day before end date!')
        return
    #-----------------------------------------------------------------------------------------------------------------------------------
    # find web-ids 
    ticker_no_df = __Get_TSE_WebID__(stock)
    if(type(ticker_no_df)==bool):
        return
    # create an empty dataframe:
    df_history = pd.DataFrame({},columns=['Date','High','Low','Final','Close','Open','Y-Final','Value','Volume','No','WEB-ID']).set_index('Date')
    # loop to get data from different pages of a ticker:
    for index, row in (ticker_no_df.reset_index()).iterrows():
        try:
            df_temp = get_price_data_forintraday(ticker_no = row['WEB-ID'])
            df_history = pd.concat([df_history,df_temp])
        except:
            pass
    # sort index and reverse the order for more processes:
    df_history = df_history.sort_index(ascending=True)
    df_history = df_history.reset_index()
    df_history['J-Date']=df_history['Date'].apply(lambda x: str(jdatetime.date.fromgregorian(date=x.date())))
    df_history = df_history.set_index('J-Date')
    df_history = df_history[start_date:end_date]
    j_date_list = df_history.index.to_list()
    ticker_no_list = df_history['WEB-ID'].to_list()
    # now we have valid Jalali date list, let's loop over and concat:
    no_days = len(j_date_list)
    if(no_days==0):
        print('There is no trading day between start and end date for this stock!')
        return 
    else:
        df_lob = pd.DataFrame(columns=['J-Date','Time','Depth','Sell_No','Sell_Vol','Sell_Price','Buy_Price','Buy_Vol','Buy_No',\
                                       'Day_LL','Day_UL','Date']).set_index(['J-Date','Time','Depth'])
        day_counter = 1
        for j_date in j_date_list:
            df_lob = pd.concat([df_lob,__Get_Day_LOB__(ticker_no_list[day_counter-1], j_date)], axis=0)
            if(show_progress):
                clear_output(wait=True)
                print('Progress : ', f'{round((day_counter)/no_days*100,1)} %')
            day_counter+=1
    if(jalali_date):
        if(combined_datatime):
            df_lob = df_lob.reset_index()
            df_lob['DateTime'] = pd.to_datetime(df_lob['Date'].astype(str) +' '+ df_lob['Time'].astype(str))
            print('Combining Jalali date and time takes a few more seconds!')
            df_lob['J-DateTime'] = df_lob['DateTime'].apply(lambda x: str(jdatetime.datetime.fromgregorian(datetime=x)))
            df_lob.drop(columns=['DateTime','Date','J-Date','Time'],inplace=True)
            df_lob = df_lob.set_index(['J-DateTime','Depth'])
        else:
            df_lob.drop(columns=['Date'],inplace=True)
    else:
        if(combined_datatime):
            df_lob = df_lob.reset_index()
            df_lob['DateTime'] = pd.to_datetime(df_lob['Date'].astype(str) +' '+ df_lob['Time'].astype(str))
            df_lob.drop(columns=['Date','J-Date','Time'],inplace=True)
            df_lob = df_lob.set_index(['DateTime','Depth'])
        else:
            df_lob = df_lob.reset_index()
            df_lob.drop(columns=['J-Date'],inplace=True)
            df_lob = df_lob.set_index(['Date','Time','Depth'])
    return df_lob

################################################################################################################################################################################
################################################################################################################################################################################

def Get_SectorIndex_History(sector = '??????????', start_date='1395-01-01', end_date='1400-12-29', ignore_date = False, \
                            just_adj_close = False, show_weekday = False, double_date = False):
    """
    ???????????? ?????????? ???????? ???????? ???????? ???? ?????? ???? ???????????? ???????????????? ?????? ?????????? ???????? ?? ??????????
    ???????????? ???????????? ?????? ?????????? ???????? ???????? ???????? ???? ?????? ???????? ???????? ???? ?????????? ???????? ?? ??????????
    ???????????? ?????????? ?????????? ???????????? ?????????? ???? ?????????? ?????????? ???????????? ?????????? ???????????? ????????
    ???????????? ???????????? ?????? ?????????? ???????????? ?????? ???????? ???????? ???????? ???????? ???? ??????
    """
    # check date validity --------------------------------------------------------------------------------------------------------------
    if(not ignore_date):
        start_date = __Check_JDate_Validity__(start_date,key_word="'START'")
        if(start_date==None):
            return
        end_date = __Check_JDate_Validity__(end_date,key_word="'END'")
        if(end_date==None):
            return
        start = jdatetime.date(year=int(start_date.split('-')[0]), month=int(start_date.split('-')[1]), day=int(start_date.split('-')[2]))
        end = jdatetime.date(year=int(end_date.split('-')[0]), month=int(end_date.split('-')[1]), day=int(end_date.split('-')[2]))
        if(start>end):
            print('Start date must be a day before end date!')
            return
    # get sector web-id ---------------------------------------------------------------------------------------------------------------------
    try:
        sector_web_id = __Get_TSE_Sector_WebID__(sector_name = sector)
    except:
        print('Please Enter a Valid Name for Sector Index!')
        return
    if(sector_web_id == None):
        return
    r_cl = requests.get(f'http://tsetmc.com/tsev2/chart/data/Index.aspx?i={sector_web_id}&t=value', headers=headers)
    df_sector_cl = pd.DataFrame(r_cl.text.split(';'))
    columns=['J-Date','Adj Close']
    df_sector_cl[columns] = df_sector_cl[0].str.split(",",expand=True)
    df_sector_cl.drop(columns=[0],inplace=True)
    df_sector_cl['J-Date'] = df_sector_cl['J-Date'].apply(lambda x: str(jdatetime.date(int(x.split('/')[0]),int(x.split('/')[1]),int(x.split('/')[2]))))
    df_sector_cl['Date'] = df_sector_cl['J-Date'].apply(lambda x: jdatetime.date(int(x[:4]),int(x[5:7]),int(x[8:])).togregorian())  
    df_sector_cl['Date'] = pd.to_datetime(df_sector_cl['Date'])
    df_sector_cl['Weekday']=df_sector_cl['Date'].dt.weekday
    df_sector_cl['Weekday'] = df_sector_cl['Weekday'].apply(lambda x: calendar.day_name[x])
    df_sector_cl = df_sector_cl.set_index('J-Date')
    df_sector_cl = df_sector_cl[['Date','Weekday','Adj Close']]
    df_sector_cl['Adj Close'] = pd.to_numeric(df_sector_cl['Adj Close'])
    if(not just_adj_close):
        r = requests.get(f'http://www.tsetmc.com/tsev2/chart/data/IndexFinancial.aspx?i={sector_web_id}&t=ph', headers=headers)
        df_sector = pd.DataFrame(r.text.split(';'))
        columns=['Date','High','Low','Open','Close','Volume','D']
        # split data into defined columns
        df_sector[columns] = df_sector[0].str.split(",",expand=True)
        df_sector.drop(columns=[0,'D'],inplace=True)
        df_sector['Date']=pd.to_datetime(df_sector['Date'])
        df_sector['J-Date']=df_sector['Date'].apply(lambda x: str(jdatetime.date.fromgregorian(date=x.date())))
        df_sector = df_sector.set_index('J-Date')
        df_sector.drop(columns=['Date'],inplace=True)
        # now concat:
        df_sector_cl = pd.concat([df_sector,df_sector_cl],axis=1).dropna()
        df_sector_cl = df_sector_cl[['Date','Weekday','Open','High','Low','Close','Adj Close','Volume']]
        cols = ['Open','High','Low','Close','Adj Close','Volume']
        df_sector_cl[cols] = df_sector_cl[cols].apply(pd.to_numeric, axis=1)
        df_sector_cl['Volume'] = df_sector_cl['Volume'].astype('int64')
    if(not show_weekday):
        df_sector_cl.drop(columns=['Weekday'],inplace=True)
    if(not double_date):
        df_sector_cl.drop(columns=['Date'],inplace=True)
    # slice requested time window:
    if(not ignore_date):
        df_sector_cl = df_sector_cl[start_date:end_date]    
    return df_sector_cl


################################################################################################################################################################################
################################################################################################################################################################################

def Get_CWPI_History(start_date='1395-01-01', end_date='1400-12-29', ignore_date = False, just_adj_close = False, show_weekday = False, double_date = False):
    """
    CWPI: Cap-Weighted Price Index = TEPIX = ???????? ???????? (????????-??????????)
    ???????????? ?????????? ???????? ???????? ???? ???????????? ???????????????? ?????? ?????????? ???????? ?? ??????????
    ???????????? ???????????? ?????? ?????????? ???????? ???????? ???????? ???????? ???? ?????????? ???????? ?? ??????????
    ???????????? ?????????? ?????????? ???????????? ?????????? ???? ?????????? ?????????? ???????????? ?????????? ???????????? ????????
    ???????????? ???????????? ?????? ?????????? ???????????? ?????? ???????? ???????? ????????
    """
    # check date validity --------------------------------------------------------------------------------------------------------------
    if(not ignore_date):
        start_date = __Check_JDate_Validity__(start_date,key_word="'START'")
        if(start_date==None):
            return
        end_date = __Check_JDate_Validity__(end_date,key_word="'END'")
        if(end_date==None):
            return
        start = jdatetime.date(year=int(start_date.split('-')[0]), month=int(start_date.split('-')[1]), day=int(start_date.split('-')[2]))
        end = jdatetime.date(year=int(end_date.split('-')[0]), month=int(end_date.split('-')[1]), day=int(end_date.split('-')[2]))
        if(start>end):
            print('Start date must be a day before end date!')
            return
    # get sector web-id ---------------------------------------------------------------------------------------------------------------------
    sector_web_id = 5798407779416661
    # get only close chart data for sector index:
    r_cl = requests.get(f'http://tsetmc.com/tsev2/chart/data/Index.aspx?i={sector_web_id}&t=value', headers=headers)
    df_sector_cl = pd.DataFrame(r_cl.text.split(';'))
    columns=['J-Date','Adj Close']
    df_sector_cl[columns] = df_sector_cl[0].str.split(",",expand=True)
    df_sector_cl.drop(columns=[0],inplace=True)
    df_sector_cl['J-Date'] = df_sector_cl['J-Date'].apply(lambda x: str(jdatetime.date(int(x.split('/')[0]),int(x.split('/')[1]),int(x.split('/')[2]))))
    df_sector_cl['Date'] = df_sector_cl['J-Date'].apply(lambda x: jdatetime.date(int(x[:4]),int(x[5:7]),int(x[8:])).togregorian())  
    df_sector_cl['Date'] = pd.to_datetime(df_sector_cl['Date'])
    df_sector_cl['Weekday']=df_sector_cl['Date'].dt.weekday
    df_sector_cl['Weekday'] = df_sector_cl['Weekday'].apply(lambda x: calendar.day_name[x])
    df_sector_cl = df_sector_cl.set_index('J-Date')
    df_sector_cl = df_sector_cl[['Date','Weekday','Adj Close']]
    df_sector_cl['Adj Close'] = pd.to_numeric(df_sector_cl['Adj Close'])
    if(not just_adj_close):
        r = requests.get(f'http://www.tsetmc.com/tsev2/chart/data/IndexFinancial.aspx?i={sector_web_id}&t=ph', headers=headers)
        df_sector = pd.DataFrame(r.text.split(';'))
        columns=['Date','High','Low','Open','Close','Volume','D']
        # split data into defined columns
        df_sector[columns] = df_sector[0].str.split(",",expand=True)
        df_sector.drop(columns=[0,'D'],inplace=True)
        df_sector['Date']=pd.to_datetime(df_sector['Date'])
        df_sector['J-Date']=df_sector['Date'].apply(lambda x: str(jdatetime.date.fromgregorian(date=x.date())))
        df_sector = df_sector.set_index('J-Date')
        df_sector.drop(columns=['Date'],inplace=True)
        # now concat:
        df_sector_cl = pd.concat([df_sector,df_sector_cl],axis=1).dropna()
        df_sector_cl = df_sector_cl[['Date','Weekday','Open','High','Low','Close','Adj Close','Volume']]
        cols = ['Open','High','Low','Close','Adj Close','Volume']
        df_sector_cl[cols] = df_sector_cl[cols].apply(pd.to_numeric, axis=1)
        df_sector_cl['Volume'] = df_sector_cl['Volume'].astype('int64')
    if(not show_weekday):
        df_sector_cl.drop(columns=['Weekday'],inplace=True)
    if(not double_date):
        df_sector_cl.drop(columns=['Date'],inplace=True)
    # slice requested time window:
    if(not ignore_date):
        df_sector_cl = df_sector_cl[start_date:end_date]    
    return df_sector_cl

################################################################################################################################################################################
################################################################################################################################################################################

def Get_EWPI_History(start_date='1395-01-01', end_date='1400-12-29', ignore_date = False, just_adj_close = False, show_weekday = False, double_date = False):
    """
    EWPI: Equal-Weighted Price Index = ???????? ???????? (???? ??????)
    ???????????? ?????????? ???????? ???????? ???? ?????? ???? ???????????? ???????????????? ?????? ?????????? ???????? ?? ??????????
    ???????????? ???????????? ?????? ?????????? ???????? ???????? ???? ?????? ???????? ???????? ???? ?????????? ???????? ?? ??????????
    ???????????? ?????????? ?????????? ???????????? ?????????? ???? ?????????? ?????????? ???????????? ?????????? ???????????? ????????
    ???????????? ???????????? ?????? ?????????? ???????????? ?????? ???????? ???????? ???????? ???? ??????
    """
    # check date validity --------------------------------------------------------------------------------------------------------------
    if(not ignore_date):
        start_date = __Check_JDate_Validity__(start_date,key_word="'START'")
        if(start_date==None):
            return
        end_date = __Check_JDate_Validity__(end_date,key_word="'END'")
        if(end_date==None):
            return
        start = jdatetime.date(year=int(start_date.split('-')[0]), month=int(start_date.split('-')[1]), day=int(start_date.split('-')[2]))
        end = jdatetime.date(year=int(end_date.split('-')[0]), month=int(end_date.split('-')[1]), day=int(end_date.split('-')[2]))
        if(start>end):
            print('Start date must be a day before end date!')
            return
    # get sector web-id ---------------------------------------------------------------------------------------------------------------------
    sector_web_id = 8384385859414435
    # get only close chart data for sector index:
    r_cl = requests.get(f'http://tsetmc.com/tsev2/chart/data/Index.aspx?i={sector_web_id}&t=value', headers=headers)
    df_sector_cl = pd.DataFrame(r_cl.text.split(';'))
    columns=['J-Date','Adj Close']
    df_sector_cl[columns] = df_sector_cl[0].str.split(",",expand=True)
    df_sector_cl.drop(columns=[0],inplace=True)
    df_sector_cl['J-Date'] = df_sector_cl['J-Date'].apply(lambda x: str(jdatetime.date(int(x.split('/')[0]),int(x.split('/')[1]),int(x.split('/')[2]))))
    df_sector_cl['Date'] = df_sector_cl['J-Date'].apply(lambda x: jdatetime.date(int(x[:4]),int(x[5:7]),int(x[8:])).togregorian())  
    df_sector_cl['Date'] = pd.to_datetime(df_sector_cl['Date'])
    df_sector_cl['Weekday']=df_sector_cl['Date'].dt.weekday
    df_sector_cl['Weekday'] = df_sector_cl['Weekday'].apply(lambda x: calendar.day_name[x])
    df_sector_cl = df_sector_cl.set_index('J-Date')
    df_sector_cl = df_sector_cl[['Date','Weekday','Adj Close']]
    df_sector_cl['Adj Close'] = pd.to_numeric(df_sector_cl['Adj Close'])
    if(not just_adj_close):
        r = requests.get(f'http://www.tsetmc.com/tsev2/chart/data/IndexFinancial.aspx?i={sector_web_id}&t=ph', headers=headers)
        df_sector = pd.DataFrame(r.text.split(';'))
        columns=['Date','High','Low','Open','Close','Volume','D']
        # split data into defined columns
        df_sector[columns] = df_sector[0].str.split(",",expand=True)
        df_sector.drop(columns=[0,'D'],inplace=True)
        df_sector['Date']=pd.to_datetime(df_sector['Date'])
        df_sector['J-Date']=df_sector['Date'].apply(lambda x: str(jdatetime.date.fromgregorian(date=x.date())))
        df_sector = df_sector.set_index('J-Date')
        df_sector.drop(columns=['Date'],inplace=True)
        # now concat:
        df_sector_cl = pd.concat([df_sector,df_sector_cl],axis=1).dropna()
        df_sector_cl = df_sector_cl[['Date','Weekday','Open','High','Low','Close','Adj Close','Volume']]
        cols = ['Open','High','Low','Close','Adj Close','Volume']
        df_sector_cl[cols] = df_sector_cl[cols].apply(pd.to_numeric, axis=1)
        df_sector_cl['Volume'] = df_sector_cl['Volume'].astype('int64')
    if(not show_weekday):
        df_sector_cl.drop(columns=['Weekday'],inplace=True)
    if(not double_date):
        df_sector_cl.drop(columns=['Date'],inplace=True)
    # slice requested time window:
    if(not ignore_date):
        df_sector_cl = df_sector_cl[start_date:end_date]    
    return df_sector_cl

################################################################################################################################################################################
################################################################################################################################################################################

def Get_FFI_History(start_date='1395-01-01', end_date='1400-12-29', ignore_date = False, just_adj_close = False, show_weekday = False, double_date = False):
    """
    FFI: Free-Float Index = ???????? ?????????? ????????
    ???????????? ?????????? ???????? ?????????? ???????? ???? ???????????? ???????????????? ?????? ?????????? ???????? ?? ??????????
    ???????????? ???????????? ?????? ?????????? ???????? ?????????? ???????? ???????? ???????? ???? ?????????? ???????? ?? ??????????
    ???????????? ?????????? ?????????? ???????????? ?????????? ???? ?????????? ?????????? ???????????? ?????????? ???????????? ????????
    ???????????? ???????????? ?????? ?????????? ???????????? ?????? ???????? ???????? ?????????? ????????
    """
    # check date validity --------------------------------------------------------------------------------------------------------------
    if(not ignore_date):
        start_date = __Check_JDate_Validity__(start_date,key_word="'START'")
        if(start_date==None):
            return
        end_date = __Check_JDate_Validity__(end_date,key_word="'END'")
        if(end_date==None):
            return
        start = jdatetime.date(year=int(start_date.split('-')[0]), month=int(start_date.split('-')[1]), day=int(start_date.split('-')[2]))
        end = jdatetime.date(year=int(end_date.split('-')[0]), month=int(end_date.split('-')[1]), day=int(end_date.split('-')[2]))
        if(start>end):
            print('Start date must be a day before end date!')
            return
    # get sector web-id ---------------------------------------------------------------------------------------------------------------------
    sector_web_id = 49579049405614711
    # get only close chart data for sector index:
    r_cl = requests.get(f'http://tsetmc.com/tsev2/chart/data/Index.aspx?i={sector_web_id}&t=value', headers=headers)
    df_sector_cl = pd.DataFrame(r_cl.text.split(';'))
    columns=['J-Date','Adj Close']
    df_sector_cl[columns] = df_sector_cl[0].str.split(",",expand=True)
    df_sector_cl.drop(columns=[0],inplace=True)
    df_sector_cl['J-Date'] = df_sector_cl['J-Date'].apply(lambda x: str(jdatetime.date(int(x.split('/')[0]),int(x.split('/')[1]),int(x.split('/')[2]))))
    df_sector_cl['Date'] = df_sector_cl['J-Date'].apply(lambda x: jdatetime.date(int(x[:4]),int(x[5:7]),int(x[8:])).togregorian())  
    df_sector_cl['Date'] = pd.to_datetime(df_sector_cl['Date'])
    df_sector_cl['Weekday']=df_sector_cl['Date'].dt.weekday
    df_sector_cl['Weekday'] = df_sector_cl['Weekday'].apply(lambda x: calendar.day_name[x])
    df_sector_cl = df_sector_cl.set_index('J-Date')
    df_sector_cl = df_sector_cl[['Date','Weekday','Adj Close']]
    df_sector_cl['Adj Close'] = pd.to_numeric(df_sector_cl['Adj Close'])
    if(not just_adj_close):
        r = requests.get(f'http://www.tsetmc.com/tsev2/chart/data/IndexFinancial.aspx?i={sector_web_id}&t=ph', headers=headers)
        df_sector = pd.DataFrame(r.text.split(';'))
        columns=['Date','High','Low','Open','Close','Volume','D']
        # split data into defined columns
        df_sector[columns] = df_sector[0].str.split(",",expand=True)
        df_sector.drop(columns=[0,'D'],inplace=True)
        df_sector['Date']=pd.to_datetime(df_sector['Date'])
        df_sector['J-Date']=df_sector['Date'].apply(lambda x: str(jdatetime.date.fromgregorian(date=x.date())))
        df_sector = df_sector.set_index('J-Date')
        df_sector.drop(columns=['Date'],inplace=True)
        # now concat:
        df_sector_cl = pd.concat([df_sector,df_sector_cl],axis=1).dropna()
        df_sector_cl = df_sector_cl[['Date','Weekday','Open','High','Low','Close','Adj Close','Volume']]
        cols = ['Open','High','Low','Close','Adj Close','Volume']
        df_sector_cl[cols] = df_sector_cl[cols].apply(pd.to_numeric, axis=1)
        df_sector_cl['Volume'] = df_sector_cl['Volume'].astype('int64')
    if(not show_weekday):
        df_sector_cl.drop(columns=['Weekday'],inplace=True)
    if(not double_date):
        df_sector_cl.drop(columns=['Date'],inplace=True)
    # slice requested time window:
    if(not ignore_date):
        df_sector_cl = df_sector_cl[start_date:end_date]    
    return df_sector_cl

################################################################################################################################################################################
################################################################################################################################################################################

def Get_MKT1I_History(start_date='1395-01-01', end_date='1400-12-29', ignore_date = False, just_adj_close = False, show_weekday = False, double_date = False):
    """
    MKT1I: First Market Index = ???????? ?????????? ??????
    ???????????? ?????????? ???????? ?????????? ?????? ???????? ???? ???????????? ???????????????? ?????? ?????????? ???????? ?? ??????????
    ???????????? ???????????? ?????? ?????????? ???????? ?????????? ?????? ???????? ???????? ???????? ???? ?????????? ???????? ?? ??????????
    ???????????? ?????????? ?????????? ???????????? ?????????? ???? ?????????? ?????????? ???????????? ?????????? ???????????? ????????
    ???????????? ???????????? ?????? ?????????? ???????????? ?????? ???????? ???????? ?????????? ?????? ????????
    """
    # check date validity --------------------------------------------------------------------------------------------------------------
    if(not ignore_date):
        start_date = __Check_JDate_Validity__(start_date,key_word="'START'")
        if(start_date==None):
            return
        end_date = __Check_JDate_Validity__(end_date,key_word="'END'")
        if(end_date==None):
            return
        start = jdatetime.date(year=int(start_date.split('-')[0]), month=int(start_date.split('-')[1]), day=int(start_date.split('-')[2]))
        end = jdatetime.date(year=int(end_date.split('-')[0]), month=int(end_date.split('-')[1]), day=int(end_date.split('-')[2]))
        if(start>end):
            print('Start date must be a day before end date!')
            return
    # get sector web-id ---------------------------------------------------------------------------------------------------------------------
    sector_web_id = 62752761908615603
    # get only close chart data for sector index:
    r_cl = requests.get(f'http://tsetmc.com/tsev2/chart/data/Index.aspx?i={sector_web_id}&t=value', headers=headers)
    df_sector_cl = pd.DataFrame(r_cl.text.split(';'))
    columns=['J-Date','Adj Close']
    df_sector_cl[columns] = df_sector_cl[0].str.split(",",expand=True)
    df_sector_cl.drop(columns=[0],inplace=True)
    df_sector_cl['J-Date'] = df_sector_cl['J-Date'].apply(lambda x: str(jdatetime.date(int(x.split('/')[0]),int(x.split('/')[1]),int(x.split('/')[2]))))
    df_sector_cl['Date'] = df_sector_cl['J-Date'].apply(lambda x: jdatetime.date(int(x[:4]),int(x[5:7]),int(x[8:])).togregorian())  
    df_sector_cl['Date'] = pd.to_datetime(df_sector_cl['Date'])
    df_sector_cl['Weekday']=df_sector_cl['Date'].dt.weekday
    df_sector_cl['Weekday'] = df_sector_cl['Weekday'].apply(lambda x: calendar.day_name[x])
    df_sector_cl = df_sector_cl.set_index('J-Date')
    df_sector_cl = df_sector_cl[['Date','Weekday','Adj Close']]
    df_sector_cl['Adj Close'] = pd.to_numeric(df_sector_cl['Adj Close'])
    if(not just_adj_close):
        r = requests.get(f'http://www.tsetmc.com/tsev2/chart/data/IndexFinancial.aspx?i={sector_web_id}&t=ph', headers=headers)
        df_sector = pd.DataFrame(r.text.split(';'))
        columns=['Date','High','Low','Open','Close','Volume','D']
        # split data into defined columns
        df_sector[columns] = df_sector[0].str.split(",",expand=True)
        df_sector.drop(columns=[0,'D'],inplace=True)
        df_sector['Date']=pd.to_datetime(df_sector['Date'])
        df_sector['J-Date']=df_sector['Date'].apply(lambda x: str(jdatetime.date.fromgregorian(date=x.date())))
        df_sector = df_sector.set_index('J-Date')
        df_sector.drop(columns=['Date'],inplace=True)
        # now concat:
        df_sector_cl = pd.concat([df_sector,df_sector_cl],axis=1).dropna()
        df_sector_cl = df_sector_cl[['Date','Weekday','Open','High','Low','Close','Adj Close','Volume']]
        cols = ['Open','High','Low','Close','Adj Close','Volume']
        df_sector_cl[cols] = df_sector_cl[cols].apply(pd.to_numeric, axis=1)
        df_sector_cl['Volume'] = df_sector_cl['Volume'].astype('int64')
    if(not show_weekday):
        df_sector_cl.drop(columns=['Weekday'],inplace=True)
    if(not double_date):
        df_sector_cl.drop(columns=['Date'],inplace=True)
    # slice requested time window:
    if(not ignore_date):
        df_sector_cl = df_sector_cl[start_date:end_date]    
    return df_sector_cl

################################################################################################################################################################################
################################################################################################################################################################################

def Get_MKT2I_History(start_date='1395-01-01', end_date='1400-12-29', ignore_date = False, just_adj_close = False, show_weekday = False, double_date = False):
    """
    MKT2I: Second Market Index = ???????? ?????????? ??????
    ???????????? ?????????? ???????? ?????????? ?????? ???????? ???? ???????????? ???????????????? ?????? ?????????? ???????? ?? ??????????
    ???????????? ???????????? ?????? ?????????? ???????? ?????????? ?????? ???????? ???????? ???????? ???? ?????????? ???????? ?? ??????????
    ???????????? ?????????? ?????????? ???????????? ?????????? ???? ?????????? ?????????? ???????????? ?????????? ???????????? ????????
    ???????????? ???????????? ?????? ?????????? ???????????? ?????? ???????? ???????? ?????????? ?????? ????????
    """
    # check date validity --------------------------------------------------------------------------------------------------------------
    if(not ignore_date):
        start_date = __Check_JDate_Validity__(start_date,key_word="'START'")
        if(start_date==None):
            return
        end_date = __Check_JDate_Validity__(end_date,key_word="'END'")
        if(end_date==None):
            return
        start = jdatetime.date(year=int(start_date.split('-')[0]), month=int(start_date.split('-')[1]), day=int(start_date.split('-')[2]))
        end = jdatetime.date(year=int(end_date.split('-')[0]), month=int(end_date.split('-')[1]), day=int(end_date.split('-')[2]))
        if(start>end):
            print('Start date must be a day before end date!')
            return
    # get sector web-id ---------------------------------------------------------------------------------------------------------------------
    sector_web_id = 71704845530629737
    # get only close chart data for sector index:
    r_cl = requests.get(f'http://tsetmc.com/tsev2/chart/data/Index.aspx?i={sector_web_id}&t=value', headers=headers)
    df_sector_cl = pd.DataFrame(r_cl.text.split(';'))
    columns=['J-Date','Adj Close']
    df_sector_cl[columns] = df_sector_cl[0].str.split(",",expand=True)
    df_sector_cl.drop(columns=[0],inplace=True)
    df_sector_cl['J-Date'] = df_sector_cl['J-Date'].apply(lambda x: str(jdatetime.date(int(x.split('/')[0]),int(x.split('/')[1]),int(x.split('/')[2]))))
    df_sector_cl['Date'] = df_sector_cl['J-Date'].apply(lambda x: jdatetime.date(int(x[:4]),int(x[5:7]),int(x[8:])).togregorian())  
    df_sector_cl['Date'] = pd.to_datetime(df_sector_cl['Date'])
    df_sector_cl['Weekday']=df_sector_cl['Date'].dt.weekday
    df_sector_cl['Weekday'] = df_sector_cl['Weekday'].apply(lambda x: calendar.day_name[x])
    df_sector_cl = df_sector_cl.set_index('J-Date')
    df_sector_cl = df_sector_cl[['Date','Weekday','Adj Close']]
    df_sector_cl['Adj Close'] = pd.to_numeric(df_sector_cl['Adj Close'])
    if(not just_adj_close):
        r = requests.get(f'http://www.tsetmc.com/tsev2/chart/data/IndexFinancial.aspx?i={sector_web_id}&t=ph', headers=headers)
        df_sector = pd.DataFrame(r.text.split(';'))
        columns=['Date','High','Low','Open','Close','Volume','D']
        # split data into defined columns
        df_sector[columns] = df_sector[0].str.split(",",expand=True)
        df_sector.drop(columns=[0,'D'],inplace=True)
        df_sector['Date']=pd.to_datetime(df_sector['Date'])
        df_sector['J-Date']=df_sector['Date'].apply(lambda x: str(jdatetime.date.fromgregorian(date=x.date())))
        df_sector = df_sector.set_index('J-Date')
        df_sector.drop(columns=['Date'],inplace=True)
        # now concat:
        df_sector_cl = pd.concat([df_sector,df_sector_cl],axis=1).dropna()
        df_sector_cl = df_sector_cl[['Date','Weekday','Open','High','Low','Close','Adj Close','Volume']]
        cols = ['Open','High','Low','Close','Adj Close','Volume']
        df_sector_cl[cols] = df_sector_cl[cols].apply(pd.to_numeric, axis=1)
        df_sector_cl['Volume'] = df_sector_cl['Volume'].astype('int64')
    if(not show_weekday):
        df_sector_cl.drop(columns=['Weekday'],inplace=True)
    if(not double_date):
        df_sector_cl.drop(columns=['Date'],inplace=True)
    # slice requested time window:
    if(not ignore_date):
        df_sector_cl = df_sector_cl[start_date:end_date]    
    return df_sector_cl

################################################################################################################################################################################
################################################################################################################################################################################

def Get_INDI_History(start_date='1395-01-01', end_date='1400-12-29', ignore_date = False, just_adj_close = False, show_weekday = False, double_date = False):
    """
    INDI: Industry Index = ???????? ????????
    ???????????? ?????????? ???????? ???????? ???????? ???? ???????????? ???????????????? ?????? ?????????? ???????? ?? ??????????
    ???????????? ???????????? ?????? ?????????? ???????? ???????? ???????? ???????? ???????? ???? ?????????? ???????? ?? ??????????
    ???????????? ?????????? ?????????? ???????????? ?????????? ???? ?????????? ?????????? ???????????? ?????????? ???????????? ????????
    ???????????? ???????????? ?????? ?????????? ???????????? ?????? ???????? ???????? ???????? ????????
    """
    # check date validity --------------------------------------------------------------------------------------------------------------
    if(not ignore_date):
        start_date = __Check_JDate_Validity__(start_date,key_word="'START'")
        if(start_date==None):
            return
        end_date = __Check_JDate_Validity__(end_date,key_word="'END'")
        if(end_date==None):
            return
        start = jdatetime.date(year=int(start_date.split('-')[0]), month=int(start_date.split('-')[1]), day=int(start_date.split('-')[2]))
        end = jdatetime.date(year=int(end_date.split('-')[0]), month=int(end_date.split('-')[1]), day=int(end_date.split('-')[2]))
        if(start>end):
            print('Start date must be a day before end date!')
            return
    # get sector web-id ---------------------------------------------------------------------------------------------------------------------
    sector_web_id = 43754960038275285
    # get only close chart data for sector index:
    r_cl = requests.get(f'http://tsetmc.com/tsev2/chart/data/Index.aspx?i={sector_web_id}&t=value', headers=headers)
    df_sector_cl = pd.DataFrame(r_cl.text.split(';'))
    columns=['J-Date','Adj Close']
    df_sector_cl[columns] = df_sector_cl[0].str.split(",",expand=True)
    df_sector_cl.drop(columns=[0],inplace=True)
    df_sector_cl['J-Date'] = df_sector_cl['J-Date'].apply(lambda x: str(jdatetime.date(int(x.split('/')[0]),int(x.split('/')[1]),int(x.split('/')[2]))))
    df_sector_cl['Date'] = df_sector_cl['J-Date'].apply(lambda x: jdatetime.date(int(x[:4]),int(x[5:7]),int(x[8:])).togregorian())  
    df_sector_cl['Date'] = pd.to_datetime(df_sector_cl['Date'])
    df_sector_cl['Weekday']=df_sector_cl['Date'].dt.weekday
    df_sector_cl['Weekday'] = df_sector_cl['Weekday'].apply(lambda x: calendar.day_name[x])
    df_sector_cl = df_sector_cl.set_index('J-Date')
    df_sector_cl = df_sector_cl[['Date','Weekday','Adj Close']]
    df_sector_cl['Adj Close'] = pd.to_numeric(df_sector_cl['Adj Close'])
    if(not just_adj_close):
        r = requests.get(f'http://www.tsetmc.com/tsev2/chart/data/IndexFinancial.aspx?i={sector_web_id}&t=ph', headers=headers)
        df_sector = pd.DataFrame(r.text.split(';'))
        columns=['Date','High','Low','Open','Close','Volume','D']
        # split data into defined columns
        df_sector[columns] = df_sector[0].str.split(",",expand=True)
        df_sector.drop(columns=[0,'D'],inplace=True)
        df_sector['Date']=pd.to_datetime(df_sector['Date'])
        df_sector['J-Date']=df_sector['Date'].apply(lambda x: str(jdatetime.date.fromgregorian(date=x.date())))
        df_sector = df_sector.set_index('J-Date')
        df_sector.drop(columns=['Date'],inplace=True)
        # now concat:
        df_sector_cl = pd.concat([df_sector,df_sector_cl],axis=1).dropna()
        df_sector_cl = df_sector_cl[['Date','Weekday','Open','High','Low','Close','Adj Close','Volume']]
        cols = ['Open','High','Low','Close','Adj Close','Volume']
        df_sector_cl[cols] = df_sector_cl[cols].apply(pd.to_numeric, axis=1)
        df_sector_cl['Volume'] = df_sector_cl['Volume'].astype('int64')
    if(not show_weekday):
        df_sector_cl.drop(columns=['Weekday'],inplace=True)
    if(not double_date):
        df_sector_cl.drop(columns=['Date'],inplace=True)
    # slice requested time window:
    if(not ignore_date):
        df_sector_cl = df_sector_cl[start_date:end_date]    
    return df_sector_cl

################################################################################################################################################################################
################################################################################################################################################################################

def Get_LCI30_History(start_date='1395-01-01', end_date='1400-12-29', ignore_date = False, just_adj_close = False, show_weekday = False, double_date = False):
    """
    30LCI: 30 Large-Cap Index = ???????? 30 ???????? ???????? ????????
    ???????????? ?????????? ???????? 30 ???????? ???????? ???????? ???? ???????????? ???????????????? ?????? ?????????? ???????? ?? ??????????
    ???????????? ???????????? ?????? ?????????? ???????? 30 ???????? ???????? ???????? ???????? ???????? ???? ?????????? ???????? ?? ??????????
    ???????????? ?????????? ?????????? ???????????? ?????????? ???? ?????????? ?????????? ???????????? ?????????? ???????????? ????????
    ???????????? ???????????? ?????? ?????????? ???????????? ?????? ???????? ???????? 30 ???????? ???????? ????????
    """
    # check date validity --------------------------------------------------------------------------------------------------------------
    if(not ignore_date):
        start_date = __Check_JDate_Validity__(start_date,key_word="'START'")
        if(start_date==None):
            return
        end_date = __Check_JDate_Validity__(end_date,key_word="'END'")
        if(end_date==None):
            return
        start = jdatetime.date(year=int(start_date.split('-')[0]), month=int(start_date.split('-')[1]), day=int(start_date.split('-')[2]))
        end = jdatetime.date(year=int(end_date.split('-')[0]), month=int(end_date.split('-')[1]), day=int(end_date.split('-')[2]))
        if(start>end):
            print('Start date must be a day before end date!')
            return
    # get sector web-id ---------------------------------------------------------------------------------------------------------------------
    sector_web_id = 10523825119011581
    # get only close chart data for sector index:
    r_cl = requests.get(f'http://tsetmc.com/tsev2/chart/data/Index.aspx?i={sector_web_id}&t=value', headers=headers)
    df_sector_cl = pd.DataFrame(r_cl.text.split(';'))
    columns=['J-Date','Adj Close']
    df_sector_cl[columns] = df_sector_cl[0].str.split(",",expand=True)
    df_sector_cl.drop(columns=[0],inplace=True)
    df_sector_cl['J-Date'] = df_sector_cl['J-Date'].apply(lambda x: str(jdatetime.date(int(x.split('/')[0]),int(x.split('/')[1]),int(x.split('/')[2]))))
    df_sector_cl['Date'] = df_sector_cl['J-Date'].apply(lambda x: jdatetime.date(int(x[:4]),int(x[5:7]),int(x[8:])).togregorian())  
    df_sector_cl['Date'] = pd.to_datetime(df_sector_cl['Date'])
    df_sector_cl['Weekday']=df_sector_cl['Date'].dt.weekday
    df_sector_cl['Weekday'] = df_sector_cl['Weekday'].apply(lambda x: calendar.day_name[x])
    df_sector_cl = df_sector_cl.set_index('J-Date')
    df_sector_cl = df_sector_cl[['Date','Weekday','Adj Close']]
    df_sector_cl['Adj Close'] = pd.to_numeric(df_sector_cl['Adj Close'])
    if(not just_adj_close):
        r = requests.get(f'http://www.tsetmc.com/tsev2/chart/data/IndexFinancial.aspx?i={sector_web_id}&t=ph', headers=headers)
        df_sector = pd.DataFrame(r.text.split(';'))
        columns=['Date','High','Low','Open','Close','Volume','D']
        # split data into defined columns
        df_sector[columns] = df_sector[0].str.split(",",expand=True)
        df_sector.drop(columns=[0,'D'],inplace=True)
        df_sector['Date']=pd.to_datetime(df_sector['Date'])
        df_sector['J-Date']=df_sector['Date'].apply(lambda x: str(jdatetime.date.fromgregorian(date=x.date())))
        df_sector = df_sector.set_index('J-Date')
        df_sector.drop(columns=['Date'],inplace=True)
        # now concat:
        df_sector_cl = pd.concat([df_sector,df_sector_cl],axis=1).dropna()
        df_sector_cl = df_sector_cl[['Date','Weekday','Open','High','Low','Close','Adj Close','Volume']]
        cols = ['Open','High','Low','Close','Adj Close','Volume']
        df_sector_cl[cols] = df_sector_cl[cols].apply(pd.to_numeric, axis=1)
        df_sector_cl['Volume'] = df_sector_cl['Volume'].astype('int64')
    if(not show_weekday):
        df_sector_cl.drop(columns=['Weekday'],inplace=True)
    if(not double_date):
        df_sector_cl.drop(columns=['Date'],inplace=True)
    # slice requested time window:
    if(not ignore_date):
        df_sector_cl = df_sector_cl[start_date:end_date]    
    return df_sector_cl

################################################################################################################################################################################
################################################################################################################################################################################

def Get_ACT50_History(start_date='1395-01-01', end_date='1400-12-29', ignore_date = False, just_adj_close = False, show_weekday = False, double_date = False):
    """
    ACT50: 50 Most Active Stocks Index = ???????? 50 ???????? ???????? ????????
    ???????????? ?????????? ???????? 50 ???????? ???????? ???????? ???? ???????????? ???????????????? ?????? ?????????? ???????? ?? ??????????
    ???????????? ???????????? ?????? ?????????? ???????? 50 ???????? ???????? ???????? ???????? ???????? ???? ?????????? ???????? ?? ??????????
    ???????????? ?????????? ?????????? ???????????? ?????????? ???? ?????????? ?????????? ???????????? ?????????? ???????????? ????????
    ???????????? ???????????? ?????? ?????????? ???????????? ?????? ???????? ???????? 50 ???????? ???????? ????????
    """
    # check date validity --------------------------------------------------------------------------------------------------------------
    if(not ignore_date):
        start_date = __Check_JDate_Validity__(start_date,key_word="'START'")
        if(start_date==None):
            return
        end_date = __Check_JDate_Validity__(end_date,key_word="'END'")
        if(end_date==None):
            return
        start = jdatetime.date(year=int(start_date.split('-')[0]), month=int(start_date.split('-')[1]), day=int(start_date.split('-')[2]))
        end = jdatetime.date(year=int(end_date.split('-')[0]), month=int(end_date.split('-')[1]), day=int(end_date.split('-')[2]))
        if(start>end):
            print('Start date must be a day before end date!')
            return
    # get sector web-id ---------------------------------------------------------------------------------------------------------------------
    sector_web_id = 46342955726788357
    # get only close chart data for sector index:
    r_cl = requests.get(f'http://tsetmc.com/tsev2/chart/data/Index.aspx?i={sector_web_id}&t=value', headers=headers)
    df_sector_cl = pd.DataFrame(r_cl.text.split(';'))
    columns=['J-Date','Adj Close']
    df_sector_cl[columns] = df_sector_cl[0].str.split(",",expand=True)
    df_sector_cl.drop(columns=[0],inplace=True)
    df_sector_cl['J-Date'] = df_sector_cl['J-Date'].apply(lambda x: str(jdatetime.date(int(x.split('/')[0]),int(x.split('/')[1]),int(x.split('/')[2]))))
    df_sector_cl['Date'] = df_sector_cl['J-Date'].apply(lambda x: jdatetime.date(int(x[:4]),int(x[5:7]),int(x[8:])).togregorian())  
    df_sector_cl['Date'] = pd.to_datetime(df_sector_cl['Date'])
    df_sector_cl['Weekday']=df_sector_cl['Date'].dt.weekday
    df_sector_cl['Weekday'] = df_sector_cl['Weekday'].apply(lambda x: calendar.day_name[x])
    df_sector_cl = df_sector_cl.set_index('J-Date')
    df_sector_cl = df_sector_cl[['Date','Weekday','Adj Close']]
    df_sector_cl['Adj Close'] = pd.to_numeric(df_sector_cl['Adj Close'])
    if(not just_adj_close):
        r = requests.get(f'http://www.tsetmc.com/tsev2/chart/data/IndexFinancial.aspx?i={sector_web_id}&t=ph', headers=headers)
        df_sector = pd.DataFrame(r.text.split(';'))
        columns=['Date','High','Low','Open','Close','Volume','D']
        # split data into defined columns
        df_sector[columns] = df_sector[0].str.split(",",expand=True)
        df_sector.drop(columns=[0,'D'],inplace=True)
        df_sector['Date']=pd.to_datetime(df_sector['Date'])
        df_sector['J-Date']=df_sector['Date'].apply(lambda x: str(jdatetime.date.fromgregorian(date=x.date())))
        df_sector = df_sector.set_index('J-Date')
        df_sector.drop(columns=['Date'],inplace=True)
        # now concat:
        df_sector_cl = pd.concat([df_sector,df_sector_cl],axis=1).dropna()
        df_sector_cl = df_sector_cl[['Date','Weekday','Open','High','Low','Close','Adj Close','Volume']]
        cols = ['Open','High','Low','Close','Adj Close','Volume']
        df_sector_cl[cols] = df_sector_cl[cols].apply(pd.to_numeric, axis=1)
        df_sector_cl['Volume'] = df_sector_cl['Volume'].astype('int64')
    if(not show_weekday):
        df_sector_cl.drop(columns=['Weekday'],inplace=True)
    if(not double_date):
        df_sector_cl.drop(columns=['Date'],inplace=True)
    # slice requested time window:
    if(not ignore_date):
        df_sector_cl = df_sector_cl[start_date:end_date]    
    return df_sector_cl

################################################################################################################################################################################
################################################################################################################################################################################
def Get_MarketWatch(save_excel = True, save_path = 'D:/FinPy-TSE Data/MarketWatch'):
    #--------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
    # GET MARKET RETAIL AND INSTITUTIONAL DATA
    #--------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
    r = requests.get('http://www.tsetmc.com/tsev2/data/ClientTypeAll.aspx', headers=headers)
    Mkt_RI_df = pd.DataFrame(r.text.split(';'))
    Mkt_RI_df = Mkt_RI_df[0].str.split(",",expand=True)
    # assign names to columns:
    Mkt_RI_df.columns = ['WEB-ID','No_Buy_R','No_Buy_I','Vol_Buy_R','Vol_Buy_I','No_Sell_R','No_Sell_I','Vol_Sell_R','Vol_Sell_I']
    # convert columns to numeric type:
    cols = ['No_Buy_R','No_Buy_I','Vol_Buy_R','Vol_Buy_I','No_Sell_R','No_Sell_I','Vol_Sell_R','Vol_Sell_I']
    Mkt_RI_df[cols] = Mkt_RI_df[cols].apply(pd.to_numeric, axis=1)
    Mkt_RI_df['WEB-ID'] = Mkt_RI_df['WEB-ID'].apply(lambda x: x.strip())
    Mkt_RI_df = Mkt_RI_df.set_index('WEB-ID')
    # re-arrange the order of columns:
    Mkt_RI_df = Mkt_RI_df[['No_Buy_R','No_Buy_I','No_Sell_R','No_Sell_I','Vol_Buy_R','Vol_Buy_I','Vol_Sell_R','Vol_Sell_I']]
    #--------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
    # GET MARKET WATCH PRICE AND OB DATA
    #--------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
    r = requests.get('http://www.tsetmc.com/tsev2/data/MarketWatchPlus.aspx', headers=headers)
    main_text = r.text
    Mkt_df = pd.DataFrame((main_text.split('@')[2]).split(';'))
    Mkt_df = Mkt_df[0].str.split(",",expand=True)
    Mkt_df.columns = ['WEB-ID','Ticker-Code','Ticker','Name','Time','Open','Final','Close','No','Volume','Value',
                      'Low','High','Y-Final','EPS','Base-Vol','Unknown1','Unknown2','Sector','Day_UL','Day_LL','Share-No','Mkt-ID']
    # re-arrange columns and drop some columns:
    Mkt_df = Mkt_df[['WEB-ID','Ticker','Name','Time','Open','Final','Close','No','Volume','Value',
                      'Low','High','Y-Final','EPS','Base-Vol','Sector','Day_UL','Day_LL','Share-No','Mkt-ID']]
    # Just keep: 300 Bourse, 303 Fara-Bourse, 305 Sandoogh, 309 Payeh, 400 H-Bourse, 403 H-FaraBourse, 404 H-Payeh
    Mkt_ID_list = ['300','303','305','309','400','403','404']
    Mkt_df = Mkt_df[Mkt_df['Mkt-ID'].isin(Mkt_ID_list)]
    Mkt_df['Market'] = Mkt_df['Mkt-ID'].map({'300':'????????','303':'??????????????','305':'?????????? ???????? ????????????','309':'????????','400':'???? ???????? ????????','403':'???? ???????? ??????????????','404':'???? ???????? ????????'})
    Mkt_df.drop(columns=['Mkt-ID'],inplace=True)   # we do not need Mkt-ID column anymore
    # assign sector names:
    r = requests.get('http://www.tsetmc.com/Loader.aspx?ParTree=111C1213', headers=headers)
    sectro_lookup = (pd.read_html(r.text)[0]).iloc[1:,:]
    # convert from Arabic to Farsi and remove half-space
    sectro_lookup[1] = sectro_lookup[1].apply(lambda x: (str(x).replace('??','??')).replace('??','??'))
    sectro_lookup[1] = sectro_lookup[1].apply(lambda x: x.replace('\u200c',' '))
    sectro_lookup[1] = sectro_lookup[1].apply(lambda x: x.strip())
    Mkt_df['Sector'] = Mkt_df['Sector'].map(dict(sectro_lookup[[0, 1]].values))
    # modify format of columns:
    cols = ['Open','Final','Close','No','Volume','Value','Low','High','Y-Final','EPS','Base-Vol','Day_UL','Day_LL','Share-No']
    Mkt_df[cols] = Mkt_df[cols].apply(pd.to_numeric, axis=1)
    Mkt_df['Time'] = Mkt_df['Time'].apply(lambda x: x[:-4]+':'+x[-4:-2]+':'+x[-2:])
    Mkt_df['Ticker'] = Mkt_df['Ticker'].apply(lambda x: (str(x).replace('??','??')).replace('??','??'))
    Mkt_df['Name'] = Mkt_df['Name'].apply(lambda x: (str(x).replace('??','??')).replace('??','??'))
    Mkt_df['Name'] = Mkt_df['Name'].apply(lambda x: x.replace('\u200c',' '))
    #calculate some new columns
    Mkt_df['Close(%)'] = round((Mkt_df['Close']-Mkt_df['Y-Final'])/Mkt_df['Y-Final']*100,2)
    Mkt_df['Final(%)'] = round((Mkt_df['Final']-Mkt_df['Y-Final'])/Mkt_df['Y-Final']*100,2)
    Mkt_df['Market Cap'] = round(Mkt_df['Share-No']*Mkt_df['Final'],2)
    # set index
    Mkt_df['WEB-ID'] = Mkt_df['WEB-ID'].apply(lambda x: x.strip())
    Mkt_df = Mkt_df.set_index('WEB-ID')
    #------------------------------------------------------------------------------------------------------------------------------------------
    # reading OB (order book) and cleaning the data
    OB_df = pd.DataFrame((main_text.split('@')[3]).split(';'))
    OB_df = OB_df[0].str.split(",",expand=True)
    OB_df.columns = ['WEB-ID','OB-Depth','Sell-No','Buy-No','Buy-Price','Sell-Price','Buy-Vol','Sell-Vol']
    OB_df = OB_df[['WEB-ID','OB-Depth','Sell-No','Sell-Vol','Sell-Price','Buy-Price','Buy-Vol','Buy-No']]
    # extract top row of order book = OB1
    OB1_df = (OB_df[OB_df['OB-Depth']=='1']).copy()         # just keep top row of OB
    OB1_df.drop(columns=['OB-Depth'],inplace=True)          # we do not need this column anymore
    # set WEB-ID as index for future joining operations:
    OB1_df['WEB-ID'] = OB1_df['WEB-ID'].apply(lambda x: x.strip())
    OB1_df = OB1_df.set_index('WEB-ID')
    # convert columns to numeric format:
    cols = ['Sell-No','Sell-Vol','Sell-Price','Buy-Price','Buy-Vol','Buy-No']
    OB1_df[cols] = OB1_df[cols].apply(pd.to_numeric, axis=1)
    # join OB1_df to Mkt_df
    Mkt_df = Mkt_df.join(OB1_df)
    # calculate buy/sell queue value
    bq_value = Mkt_df.apply(lambda x: int(x['Buy-Vol']*x['Buy-Price']) if(x['Buy-Price']==x['Day_UL']) else 0 ,axis = 1)
    sq_value = Mkt_df.apply(lambda x: int(x['Sell-Vol']*x['Sell-Price']) if(x['Sell-Price']==x['Day_LL']) else 0 ,axis = 1)
    Mkt_df = pd.concat([Mkt_df,pd.DataFrame(bq_value,columns=['BQ-Value']),pd.DataFrame(sq_value,columns=['SQ-Value'])],axis=1)
    # calculate buy/sell queue average per-capita:
    bq_pc_avg = Mkt_df.apply(lambda x: int(round(x['BQ-Value']/x['Buy-No'],0)) if((x['BQ-Value']!=0) and (x['Buy-No']!=0)) else 0 ,axis = 1)
    sq_pc_avg = Mkt_df.apply(lambda x: int(round(x['SQ-Value']/x['Sell-No'],0)) if((x['SQ-Value']!=0) and (x['Sell-No']!=0)) else 0 ,axis = 1)
    Mkt_df = pd.concat([Mkt_df,pd.DataFrame(bq_pc_avg,columns=['BQPC']),pd.DataFrame(sq_pc_avg,columns=['SQPC'])],axis=1)
    # just keep tickers with Value grater than zero! = traded stocks:
    #Mkt_df = Mkt_df[Mkt_df['Value']!=0]
    #--------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
    # JOIN DATA
    #--------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
    final_df = Mkt_df.join(Mkt_RI_df)
    # add trade types:
    final_df['Trade Type'] = pd.DataFrame(final_df['Ticker'].apply(lambda x: '??????????' if((not x[-1].isdigit())or(x in ['??????????1','??????????2','??????????3'])) 
                                                                   else ('??????????' if(x[-1]=='2') else ('????????' if(x[-1]=='4') else ('????????????' if(x[-1]=='3') else '??????????')))))
    # add update Jalali date and time:
    jdatetime_download = jdatetime.datetime.today().strftime("%Y-%m-%d %H:%M:%S")
    final_df['Download'] = jdatetime_download
    # just keep necessary columns and re-arrange theor order:
    final_df = final_df[['Ticker','Trade Type','Time','Open','High','Low','Close','Final','Close(%)','Final(%)',
                         'Day_UL', 'Day_LL','Value','BQ-Value', 'SQ-Value', 'BQPC', 'SQPC',
                         'Volume','Vol_Buy_R', 'Vol_Buy_I', 'Vol_Sell_R', 'Vol_Sell_I','No','No_Buy_R', 'No_Buy_I', 'No_Sell_R', 'No_Sell_I',
                         'Name','Market','Sector','Share-No','Base-Vol','Market Cap','EPS','Download']]
    final_df = final_df.set_index('Ticker')
    # convert columns to int64 data type:
    """cols = ['Open','High','Low','Close','Final','Day_UL', 'Day_LL','Value', 'BQ-Value', 'SQ-Value', 'BQPC', 'SQPC',
            'Volume','Vol_Buy_R', 'Vol_Buy_I', 'Vol_Sell_R', 'Vol_Sell_I','No','No_Buy_R', 'No_Buy_I', 'No_Sell_R', 'No_Sell_I',
            'Share-No','Base-Vol','Market Cap']
    final_df[cols] = final_df[cols].astype('int64')"""
    #--------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
    # PROCESS ORDER BOOK DATA IF REQUESTED
    #--------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
    final_OB_df = ((Mkt_df[['Ticker','Day_LL','Day_UL']]).join(OB_df.set_index('WEB-ID')))
    # convert columns to numeric int64
    cols = ['Day_LL','Day_UL','OB-Depth','Sell-No','Sell-Vol','Sell-Price','Buy-Price','Buy-Vol','Buy-No']
    final_OB_df[cols] = final_OB_df[cols].astype('int64')
    # sort using tickers and order book depth:
    final_OB_df = final_OB_df.sort_values(['Ticker','OB-Depth'], ascending = (True, True))
    final_OB_df = final_OB_df.set_index(['Ticker','Day_LL','Day_UL','OB-Depth'])
    # add Jalali date and time:
    final_OB_df['Download'] =jdatetime_download
    #--------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
    # SAVE OPTIONS AND RETURNS
    #--------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
    if(save_excel):
        try:
            if(save_path[-1] != '/'):
                save_path = save_path+'/'
            mkt_watch_file_name = 'MarketWatch '+jdatetime.datetime.today().strftime("%Y-%m-%d %H-%M-%S")
            OB_file_name = 'OrderBook '+jdatetime.datetime.today().strftime("%Y-%m-%d %H-%M-%S")
            final_OB_df.to_excel(save_path+OB_file_name+'.xlsx')
            final_df.to_excel(save_path+mkt_watch_file_name+'.xlsx')
        except:
            print('Save path does not exist, you can handle saving this data by returned dataframe as Excel using ".to_excel()", if you will!')  
    return final_df, final_OB_df
################################################################################################################################################################################
################################################################################################################################################################################
def __Save_List__(df_data, bourse, farabourse, payeh, detailed_list, save_excel, save_csv, save_path = 'D:/FinPy-TSE Data/'):
    # find today's j-date ti use in name of the file
    today_j_date = jdatetime.datetime.now().strftime("%Y-%m-%d")
    # select name:
    if(bourse):
        if(farabourse):
            if(payeh):
                if(detailed_list):
                    name = today_j_date+' detailed_stocklist_bfp'
                else:
                    name = today_j_date+' stocklist_bfp'
            else:
                if(detailed_list):
                    name = today_j_date+' detailed_stocklist_bf'
                else:
                    name = today_j_date+' stocklist_bf'
        else:
            if(payeh):
                if(detailed_list):
                    name = today_j_date+' detailed_stocklist_bp'
                else:
                    name = today_j_date+' stocklist_bp'
            else:
                if(detailed_list):
                    name = today_j_date+' detailed_stocklist_b'
                else:
                    name = today_j_date+' stocklist_b'
    else:
        if(farabourse):
            if(payeh):
                if(detailed_list):
                    name = today_j_date+' detailed_stocklist_fp'
                else:
                    name = today_j_date+' stocklist_fp'
            else:
                if(detailed_list):
                    name = today_j_date+' detailed_stocklist_f'
                else:
                    name = today_j_date+' stocklist_f'
        else:
            if(payeh):
                if(detailed_list):
                    name = today_j_date+' detailed_stocklist_p'
                else:
                    name = today_j_date+' stocklist_p'
            else:
                name = None
    #------------------------------------------------
    # modify save path if necessary:
    if(save_path[-1] != '/'):
        save_path = save_path+'/'
    # save Excel file:
    if(save_excel):
        try:
            df_data.to_excel(save_path+name+'.xlsx')
            print('File saved in the specificed directory as: ',name+'.xlsx')
        except:
            print('Save path does not exist, you can handle saving this data by returned dataframe as Excel using ".to_excel()", if you will!')  
    # save Excel file:
    if(save_csv):
        try:
            df_data.to_csv(save_path+name+'.csv')
            print('File saved in the specificed directory as: ',name+'.csv')
        except:
            print('Save path does not exist, you can handle saving this data by returned dataframe as CSV using ".to_csv()", if you will!') 
    return


def Build_Market_StockList(bourse = True, farabourse = True, payeh = True, detailed_list = True, show_progress = True, \
                           save_excel = True, save_csv = True, save_path = 'D:/FinPy-TSE Data/'):
    if(not bourse and not farabourse and not payeh):
        print('Choose at least one market!')
        return
    start_time = time.time()
    http = urllib3.PoolManager()
    look_up = pd.DataFrame({'Ticker':[],'Name':[],'WEB-ID':[],'Market':[]})
    # --------------------------------------------------------------------------------------------------
    if(bourse):
        if(show_progress):
            clear_output(wait=True)
            print('Gathering Bourse market stock list ...')
        r = http.request('GET', "http://www.tsetmc.com/Loader.aspx?ParTree=15131J&i=32097828799138957") 
        soup = BeautifulSoup(r.data.decode('utf-8'), 'html.parser')
        table = soup.findAll("table", {"class": "table1"})
        stock_list = table[0].find_all('a')
        ticker = []
        web_id = []
        name = []
        for stock in stock_list:
            ticker.append(stock.text)
            name.append(stock.attrs["title"])
            web_id.append(stock.attrs["href"].split("&i=")[1])
        bourse_lookup = pd.DataFrame({'Ticker':ticker, 'Name':name,'WEB-ID':web_id}) 
        bourse_lookup['Market'] = '????????'
        look_up = pd.concat([look_up,bourse_lookup],axis=0)
    # --------------------------------------------------------------------------------------------------
    if(farabourse):
        if(show_progress):
            clear_output(wait=True)
            print('Gathering Fara-Bourse market stock list ...')
        r = http.request('GET', 'http://www.tsetmc.com/Loader.aspx?ParTree=15131J&i=43685683301327984') 
        soup = BeautifulSoup(r.data.decode('utf-8'), 'html.parser')
        table = soup.findAll("table", {"class": "table1"})
        stock_list = table[0].find_all('a')
        ticker = []
        web_id = []
        name = []
        for stock in stock_list:
            ticker.append(stock.text)
            name.append(stock.attrs["title"])
            web_id.append(stock.attrs["href"].split("&i=")[1])
        farabourse_lookup = pd.DataFrame({'Ticker':ticker, 'Name':name,'WEB-ID':web_id}) 
        farabourse_lookup['Market'] = '??????????????'
        look_up = pd.concat([look_up,farabourse_lookup],axis=0)
    # --------------------------------------------------------------------------------------------------
    if(payeh):
        if(show_progress):
            clear_output(wait=True)
            print('Gathering Payeh market stock list ...')
        url = "https://www.ifb.ir/StockQoute.aspx"
        header = {"__EVENTTARGET": "exportbtn"}
        response = requests.post(url, header)
        table = pd.read_html(response.content.decode('utf-8'))[0]
        payeh_lookup = table.iloc[2:,:3]
        payeh_lookup.columns = ['Ticker','Name','Market']
        payeh_lookup = payeh_lookup[payeh_lookup['Market'].isin(['?????????? ???????? ??????','?????????? ???????? ????????????','?????????? ???????? ????????'])] 
        payeh_lookup['Market'] = payeh_lookup['Market'].apply(lambda x: (x.replace('??????????','')).strip())
        # drop other than normal trades:
        payeh_lookup = payeh_lookup[payeh_lookup['Ticker'].apply(lambda x: x[-1].isdigit())==False]
        # drop hagh-taghaddom!
        payeh_lookup = payeh_lookup[payeh_lookup['Ticker'].apply(lambda x: x.strip()[-1]!='??')]
        look_up = pd.concat([look_up,payeh_lookup],axis=0)
    # ---------------------------------------------------------------------------------------------------
    if(not detailed_list):
        # convert tickers and names to farsi characters 
        look_up['Ticker'] = look_up['Ticker'].apply(lambda x: (str(x).replace('??','??')).replace('??','??'))
        look_up['Name'] = look_up['Name'].apply(lambda x: (str(x).replace('??','??')).replace('??','??'))
        look_up['Name'] = look_up['Name'].apply(lambda x: x.replace('\u200c',' '))
        look_up = look_up.set_index('Ticker')
        look_up.drop(columns=['WEB-ID'],inplace=True)
        if(show_progress):
            clear_output(wait=True) 
        # save file if necessary
        if(save_excel|save_csv):
            __Save_List__(df_data=look_up, bourse=bourse, farabourse=bourse, payeh=payeh, detailed_list=detailed_list,save_excel=save_excel, save_csv=save_csv, save_path=save_path)
        return look_up
    else:
        if(show_progress):
            clear_output(wait=True)
            print('Searching Payeh market stocks web-pages ...')
        # rearrange columns
        look_up['Ticker'] = look_up['Ticker'].apply(lambda x: (str(x).replace('??','??')).replace('??','??'))
        look_up['Name'] = look_up['Name'].apply(lambda x: (str(x).replace('??','??')).replace('??','??'))
        look_up['Name'] = look_up['Name'].apply(lambda x: x.replace('\u200c',' '))
        look_up = look_up.set_index('Ticker')
        look_up = look_up[['Name','Market','WEB-ID']]
        if(payeh):
            # some minor changes in payeh_lookup
            payeh_lookup['Ticker'] = payeh_lookup['Ticker'].apply(lambda x: characters.ar_to_fa(x))
            payeh_lookup = payeh_lookup.set_index('Ticker')
            # look for payeh market web-ids from market watch
            r = requests.get('http://www.tsetmc.com/tsev2/data/MarketWatchPlus.aspx', headers=headers)
            mkt_watch = pd.DataFrame((r.text.split('@')[2]).split(';'))
            mkt_watch = mkt_watch[0].str.split(",",expand=True)
            mkt_watch = mkt_watch[[0,2]]
            mkt_watch.columns = ['WEB-ID','Ticker']
            mkt_watch['Ticker'] = mkt_watch['Ticker'].apply(lambda x: characters.ar_to_fa(x))
            mkt_watch = mkt_watch.set_index('Ticker')
            # join based on payeh_lookup
            payeh_lookup = payeh_lookup.join(mkt_watch)
            with_web_id = (payeh_lookup[payeh_lookup['WEB-ID'].notnull()]).copy()
            no_web_id = (payeh_lookup[payeh_lookup['WEB-ID'].isnull()]).copy()
            no_web_id.drop(columns=['WEB-ID'],inplace=True)
            # search from google for no web-id stocks:
            web_id = []
            no_stocks = len(no_web_id)
            counter = 1
            for index, row in no_web_id.iterrows():
                if(show_progress):
                    clear_output(wait=True)
                    print('Searching Payeh market stocks web-pages: ', f'{round((counter)/no_stocks*100,1)} %')
                # search with ticker, if you find nothing, then search with name
                code_df = __Get_TSE_WebID__(index)
                code_df = code_df.reset_index()
                try:
                    web_id.append(code_df[code_df['Active']==1].iloc[0]['WEB-ID'])
                    counter+=1
                except:
                    web_id.append(code_df[code_df['Active']==0].iloc[0]['WEB-ID'])
                    counter+=1
                    pass
            # add new codes to dataframe
            no_web_id['WEB-ID'] = web_id 
            # build payeh dataframe with web-ids again:
            payeh_lookup = pd.concat([with_web_id,no_web_id])
            # add to bourse and fara-bourse:
            look_up = pd.concat([look_up[look_up['WEB-ID'].notnull()],payeh_lookup])
            look_up['Name'] = look_up['Name'].apply(lambda x: characters.ar_to_fa(x))
        # read stocks IDs from TSE webpages:
        def get_data_optimaize():
            tracemalloc.start()
            @unsync
            async def get_data_parallel():
                counter = 1
                async with aiohttp.ClientSession() as session:
                    tasks = []
                    for code in look_up['WEB-ID'].to_list():
                        task = asyncio.ensure_future(get_session(session, code))
                        tasks.append(task)
                    view_counts = await asyncio.gather(*tasks)
                    for i in view_counts :
                        if (counter==1):
                            df_final = i.copy()
                        else:
                            df_final = pd.concat([df_final,i])
                        counter+=1
                return df_final
            async def get_session(session, code):
                url = f'http://www.tsetmc.com/Loader.aspx?Partree=15131M&i={code}'
                async with session.get(url, headers=headers) as response:
                    data_text = await response.text()
                    soup = BeautifulSoup(data_text, 'html.parser')
                    table = soup.findAll("table", {"class": "table1"})
                    df_id = pd.read_html(str(table[0]))[0]
                    # rotate data frame:
                    df_id = df_id.T
                    df_id.columns = df_id.iloc[0]
                    df_id = df_id[1:]
                    df_current_stock = look_up[look_up['WEB-ID'] == code]
                    df_id['Ticker'] = df_current_stock.index[0]
                    df_id['Market'] = df_current_stock['Market'][0]
                    df_id['WEB-ID'] = df_current_stock['WEB-ID'][0]
                    return df_id
                return 
            return get_data_parallel().result()
        no_stocks = len(look_up)
        if(show_progress):
            clear_output(wait=True)
            print(f'Be patient! Gathering detailed data of {no_stocks} stocks ...')
        df_final = get_data_optimaize()
        df_final.columns=['Ticker(12)','Ticker(5)','Name(EN)','Ticker(4)','Name','Comment','Ticker(30)','Company Code(12)',
                          'Panel','Panel Code', 'Sector Code','Sector','Sub-Sector Code','Sub-Sector','Ticker','Market','WEB-ID']
        df_final['Comment'] = df_final['Comment'].apply(lambda x: x.split('-')[1] if(len(x.split('-'))>1) else '-')
        df_final = df_final[['Ticker','Name','Market','Panel','Sector','Sub-Sector','Comment','Name(EN)',\
                             'Company Code(12)','Ticker(4)','Ticker(5)','Ticker(12)','Sector Code','Sub-Sector Code','Panel Code','WEB-ID']]
        # change arabic letter to farsi letters nad drop half-spaces:
        df_final['Ticker']=df_final['Ticker'].apply(lambda x: (str(x).replace('??','??')).replace('??','??'))
        df_final['Name']=df_final['Name'].apply(lambda x: (str(x).replace('??','??')).replace('??','??'))
        df_final['Panel']=df_final['Panel'].apply(lambda x: (str(x).replace('??','??')).replace('??','??'))
        df_final['Sector']=df_final['Sector'].apply(lambda x: (str(x).replace('??','??')).replace('??','??'))
        df_final['Sub-Sector']=df_final['Sub-Sector'].apply(lambda x: (str(x).replace('??','??')).replace('??','??'))
        df_final['Name'] = df_final['Name'].apply(lambda x: x.replace('\u200c',' '))
        df_final['Panel'] = df_final['Panel'].apply(lambda x: x.replace('\u200c',' '))
        df_final['Sector'] = df_final['Sector'].apply(lambda x: x.replace('\u200c',' '))
        df_final['Sub-Sector'] = df_final['Sub-Sector'].apply(lambda x: x.replace('\u200c',' '))

        df_final = df_final.set_index('Ticker')
        df_final.drop(columns=['WEB-ID'],inplace=True)
        end_time = time.time()
        if(show_progress):
            clear_output(wait=True)
            print('Progress : 100 % , Done in ' + str(int(round(end_time - start_time,0)))+' seconds!')
        #print(str(int(round(end_time - start_time,0)))+' seconds took to gather detailed data')
        #-------------------------------------------------------------------------------------------------------------------------------------
        # save file if necessary
        if(save_excel|save_csv):
            __Save_List__(df_data=df_final, bourse=bourse, farabourse=bourse, payeh=payeh, detailed_list=detailed_list,save_excel=save_excel, save_csv=save_csv, save_path=save_path)
        return df_final
    
################################################################################################################################################################################
################################################################################################################################################################################
def __get_history_data_group_parallel__(stock_list) :
    # a function for finding web-id of symbols or tickers -------------------------------------------
    def find_code(stock_list) :
        if type(stock_list) != list :
            return False
        tracemalloc.start()
        @unsync
        #?????????? ???????? ?????? ???????? ?????????? ???? ?????? ???? ?? ?????????????????? ?????????? ???? ???????? ???????? ?????? ??????????????  
        async def parallel_request(stock_list):

            async def get_data(session, stock):
                url = f'http://www.tsetmc.com/tsev2/data/search.aspx?skey={stock}'
                #?????????? ??????????????
                async with session.get(url, headers=headers) as response:
                    data_id = await response.text()

                    #?????????? ???? ???????? ???????? ?????????? ???????? ????????
                    data = []
                    for i in data_id.split(';') :
                        try :
                            i = i.split(',')
                            data.append([i[0],i[1],i[2],i[7]])
                        except :
                            pass

                    #???????????????? ????????
                    data = pd.DataFrame(data, columns=['Ticker','Name','WEB-ID','Active'])
                    data['Name'] = data['Name'].apply(lambda x : ''.join(x.split('\u200c')).strip())
                    data['Ticker'] = data['Ticker'].apply(lambda x : ''.join(x.split('\u200c')).strip())
                    data['Name-Split'] = data['Name'].apply(lambda x : ''.join(x.split()).strip())
                    data['Symbol-Split'] = data['Ticker'].apply(lambda x : ''.join(x.split()).strip())
                    data['Active'] = pd.to_numeric(data['Active'])
                    data = data.sort_values('Ticker')
                    data = pd.DataFrame(data[['Name','WEB-ID','Name-Split','Symbol-Split']].values, columns=['Name','WEB-ID',
                                        'Name-Split','Symbol-Split'], index=pd.MultiIndex.from_frame(data[['Ticker','Active']]))

                    return data

            #???????????? ?????????????? ????
            async with aiohttp.ClientSession() as session:
                tasks = []
                for stock in stock_list:
                    #?????????????? ?????????? ???????? ???????? ???????? ?????????? ?????????????? ???? ???????? ???????? 
                    task = asyncio.ensure_future(get_data(session, stock))
                    #?????????? ???????? ???????????????? ???? ???? ????????
                    tasks.append(task)
                view_counts = await asyncio.gather(*tasks)

            return view_counts


        def request(stock_list) :

            view_counts = []
            for stock in stock_list :
                while True :
                    try :
                        data_id = requests.get(f'http://www.tsetmc.com/tsev2/data/search.aspx?skey={stock}', headers=headers).text
                        break
                    except :
                        print('nn')
                        pass

                data = []
                for i in data_id.split(';') :
                    try :
                        i = i.split(',')
                        data.append([i[0],i[1],i[2],i[7]])
                    except :
                        pass

                #???????????????? ????????
                data = pd.DataFrame(data, columns=['Ticker','Name','WEB-ID','Active'])
                data['Name'] = data['Name'].apply(lambda x : ''.join(x.split('\u200c')).strip())
                data['Ticker'] = data['Ticker'].apply(lambda x : ''.join(x.split('\u200c')).strip())
                data['Name-Split'] = data['Name'].apply(lambda x : ''.join(x.split()).strip())
                data['Symbol-Split'] = data['Ticker'].apply(lambda x : ''.join(x.split()).strip())
                data['Active'] = pd.to_numeric(data['Active'])
                data = data.sort_values('Ticker')
                data = pd.DataFrame(data[['Name','WEB-ID','Name-Split','Symbol-Split']].values, columns=['Name','WEB-ID',
                                    'Name-Split','Symbol-Split'], index=pd.MultiIndex.from_frame(data[['Ticker','Active']]))

                view_counts.append(data)

            return view_counts
        #--------------------------------------------------------------------------------------------------------------------------------------
        # cleaning entry list

        #???????????????? ???????? ???????? ??????????????
        list_first_name, stock_list_split = [], []
        for stock in stock_list :
            stock = characters.fa_to_ar(''.join(stock.split('\u200c')).strip())
            list_first_name.append(stock.split()[0])
            stock_list_split.append(''.join(stock.split()))

        #TSE ?????????? ?????????? ?????? ???? 
        while True :
            try :
                data = parallel_request(list_first_name).result()
                break
            except :
                print('n')
                pass



        #?????????????? ?????? ?????? ???????? ???????? ???????? ???? ?????????? ??????
        df_symbols, df_names = pd.DataFrame(), pd.DataFrame()
        for i in list(zip(data,stock_list_split)) :
            df_symbol = i[0][i[0]['Symbol-Split'] == i[1]]
            df_name = i[0][i[0]['Name-Split'] == i[1]]
            df_symbols = pd.concat([df_symbols, df_symbol])
            df_names = pd.concat([df_names, df_name])


        list_first_name_not, stock_list_split_not = [], []
        for i in range(len(data)) :
            if len(data[i]) == 0 :
                list_first_name_not.append(list_first_name[i])
                stock_list_split_not.append(stock_list_split[i])


        data = request(list_first_name_not)
        for i in list(zip(data,stock_list_split_not)) :
            df_symbol = i[0][i[0]['Symbol-Split'] == i[1]]
            df_name = i[0][i[0]['Name-Split'] == i[1]]
            df_symbols = pd.concat([df_symbols, df_symbol])
            df_names = pd.concat([df_names, df_name])



        if len(df_names) > 0 :
            #?????????????? ???????? ???????????????? ???? ?????? ???????? ???????? ??????
            stock_list = [characters.fa_to_ar(''.join(i.split('\u200c')).strip()) for i in 
                          df_names.index[~df_names.index.get_level_values('Ticker').duplicated()].get_level_values('Ticker')]

            #TSE ?????????? ?????????? ?????? ???? 
            while True :
                try :
                    data = parallel_request(stock_list).result()
                    break
                except :
                    print('n')
                    pass



            #?????????????? ?????? ?????? ???????? ???????? ???????? ???? ?????????? ??????
            for i in list(zip(data,stock_list)) :
                df_symbol = i[0][i[0].index.get_level_values('Ticker') == i[1]]
                df_symbols = pd.concat([df_symbols, df_symbol])


            stock_list_not = []
            for i in range(len(data)) :
                if len(data[i]) == 0 :
                    stock_list_not.append(stock_list[i])


            data = request(stock_list_not)
            for i in list(zip(data,stock_list_not)) :
                df_symbol = i[0][i[0].index.get_level_values('Ticker') == i[1]]
                df_symbols = pd.concat([df_symbols, df_symbol])


        if len(df_symbols) == 0 :
            return False

        return df_symbols.drop(['Name-Split','Symbol-Split'], axis=1)
    
    
    
    def get_price(dict_code) :
    
        tracemalloc.start()
        @unsync
        #?????????? ???????? ?????? ???????? ?????????? ???? ?????? ???? ?? ?????????????????? ?????????? ???? ???????? ???????? ?????? ??????????????  
        async def parallel_request(list_code):

            async def get_data(session, code):
                url = f'http://members.tsetmc.com/tsev2/data/InstTradeHistory.aspx?i={code}&Top=999999&A=0'
                #?????????? ??????????????
                async with session.get(url, headers=headers) as response:
                    data_id = await response.text()
                    return [data_id,response.status]

            #???????????? ?????????????? ????
            async with aiohttp.ClientSession() as session:
                tasks = []
                for code in list_code:
                    #?????????????? ?????????? ???????? ???????? ???????? ?????????? ?????????????? ???? ???????? ???????? 
                    task = asyncio.ensure_future(get_data(session, code))
                    #?????????? ???????? ???????????????? ???? ???? ????????
                    tasks.append(task)
                view_counts = await asyncio.gather(*tasks)

            return view_counts


        data = {}
        while True :

            while True :
                try :
                    res = parallel_request(list(dict_code.values())).result()
                    break
                except :
                    pass

            for i in list(zip(res,dict_code.keys())) :     
                try :
                    if i[0][1] == 200 :
                        data[i[1]] = i[0][0]
                        del dict_code[i[1]]                   
                except :
                    pass

            if len(dict_code) == 0 :
                break

        data = dict(sorted(data.items()))

        return data

    
    
    df_total = find_code(stock_list)
    #print('get codes')
    
    df_total['price'] = ''
    dict_code = {}
    for i in range(len(df_total)) :
        dict_code[i] = df_total.iloc[i,1]

    data = get_price(dict_code)

    for index in data.keys() :
        df_total.iloc[index,2] = data[index]

    return df_total
# a function to get price data from a given page ----------------------------------------------------------------------------------
def __process_price_data__(ticker_no, ticker, r, data_part):
    df_history=pd.DataFrame(r.split(';'))
    columns=['Date','High','Low','Final','Close','Open','Y-Final','Value','Volume','No']
    #split data into defined columns
    df_history[columns] = df_history[0].str.split("@",expand=True)
    # drop old column 0
    df_history.drop(columns=[0],inplace=True)
    df_history.dropna(inplace=True)
    df_history['Date']=pd.to_datetime(df_history['Date'])
    df_history['Ticker'] = ticker
    df_history['Part'] = data_part
    df_history = df_history.set_index('Date')
    return df_history
# ----------------------------------------------------------------------------------------------------------------------------------
# process the data: responses might be duplicate
def __build_price_panel_seg__(df_response, param, save_excel = True, save_path = 'D:/FinPy-TSE Data/Price Panel/'):
    # remove empty responses:
    df_response = df_response[df_response['price']!='']
    # drop duplicate indexes (repetitive indexes)
    df_response = pd.concat([df_response[~df_response['WEB-ID'].duplicated(keep=False)],
                             df_response[df_response['WEB-ID'].duplicated(keep='first')]],axis=0)
    # convert ticker from Arabic to Farsi:
    df_response = df_response.reset_index()
    df_response['Ticker'] = df_response['Ticker'].apply(lambda x: (str(x).replace('??','??')).replace('??','??'))
    df_response = df_response.set_index(['Ticker','Active'])
    df_response
    # now loop over and process the data ----------------------------------------------------------------------------------------------
    for ticker, ticker_no_df in df_response.groupby(level=0):
        # create an empty dataframe:
        df_history = pd.DataFrame({},columns=['Date','High','Low','Final','Close','Open','Y-Final',
                                              'Value','Volume','No','Ticker','Part']).set_index('Date')
        # loop to get data from different pages of a ticker:
        for index, row in (ticker_no_df.reset_index()).iterrows():
            df_temp = __process_price_data__(ticker_no = row['WEB-ID'],ticker = row['Ticker'], r = row['price'], data_part = index+1)
            df_history = pd.concat([df_history,df_temp])
        # sort index and reverse the order for more processes:
        df_history = df_history.sort_index(ascending=True)
        df_history = df_history.reset_index()
        # determining week days:
        df_history['Weekday']=df_history['Date'].dt.weekday
        df_history['Weekday'] = df_history['Weekday'].apply(lambda x: calendar.day_name[x])
        df_history['J-Date']=df_history['Date'].apply(lambda x: str(jdatetime.date.fromgregorian(date=x.date())))
        df_history = df_history.set_index('J-Date')
        # rearrange columns:
        df_history=df_history[['Date','Weekday','Y-Final','Open','High','Low','Close','Final','Volume','Value','No','Ticker','Part']]
        cols = ['Y-Final','Open','High','Low','Close','Final','Volume','No','Value','Part']
        df_history[cols] = df_history[cols].apply(pd.to_numeric, axis=1)
        # Y-Final for new part of data could be 0 or 1000, we need to replace them with yesterday's final price:
        df_history['Final(+1)'] = df_history['Final'].shift(+1)        # final prices shifted forward by one day
        df_history['temp'] = df_history.apply(lambda x: x['Y-Final'] if((x['Y-Final']!=0)and(x['Y-Final']!=1000)) 
                                              else (x['Y-Final'] if(pd.isnull(x['Final(+1)'])) else x['Final(+1)']),axis = 1)
        df_history['Y-Final'] = df_history['temp']
        df_history.drop(columns=['Final(+1)','temp'],inplace=True)
        for col in cols:
            df_history[col] = df_history[col].apply(lambda x: int(x)) # convert to int because we do not have less than Rial
        # Adjust price data:
        df_history['COEF'] = (df_history['Y-Final'].shift(-1)/df_history['Final']).fillna(1.0)
        df_history['ADJ-COEF']=df_history.iloc[::-1]['COEF'].cumprod().iloc[::-1]
        df_history['Adj Open'] = (df_history['Open']*df_history['ADJ-COEF']).apply(lambda x: int(x))
        df_history['Adj High'] = (df_history['High']*df_history['ADJ-COEF']).apply(lambda x: int(x))
        df_history['Adj Low'] = (df_history['Low']*df_history['ADJ-COEF']).apply(lambda x: int(x))
        df_history['Adj Close'] = (df_history['Close']*df_history['ADJ-COEF']).apply(lambda x: int(x))
        df_history['Adj Final'] = (df_history['Final']*df_history['ADJ-COEF']).apply(lambda x: int(x))
        df_history.drop(columns=['COEF','ADJ-COEF'],inplace=True)
        # re-arrange again:
        df_history=df_history[['Date','Weekday','Open','High','Low','Close','Final','Volume','Value','No',
                               'Adj Open','Adj High','Adj Close','Adj Final','Ticker','Part']]
        #----------------------------------------------------------------------------------------------------------------------------------------------
        # save data in a given directory:
        if(save_excel):
            try:
                df_history.to_excel(save_path+row['Ticker'].strip()+'.xlsx')
            except:
                pass
        # separate required column for price panel: Adj Final
        df_panel_temp = df_history.reset_index().set_index('Date')
        df_panel_temp = df_panel_temp[[param]]
        df_panel_temp.columns = [row['Ticker'].strip()]
        try:
            df_panel = pd.concat([df_panel,df_panel_temp],axis=1)
        except:  # for first time
            df_panel = df_panel_temp.copy()
    return df_panel

def Build_PricePanel(stock_list, param = 'Adj Final', jalali_date = True, save_excel = True, save_path = 'D:/FinPy-TSE Data/Price Panel/'):
    if(param not in ['Final','Adj Final']):
        print('Invalid Input Error for "param": Valid inputs are "Final" and "Adj Final"')
        return
    segment_size = 25
    # check save path:
    if(save_excel):
        if(save_path[-1] != '/'):
            save_path = save_path+'/' 
        today_j_date = jdatetime.datetime.now().strftime("%Y-%m-%d")
        df_save_test = pd.DataFrame({'Stocks':stock_list})
        try:
            df_save_test.to_excel(save_path+today_j_date+' Price Panel'+'.xlsx')
        except:
            print('Save path does not exist, Please Enter a Valid Destination Path!')
            return   
    # segment data using given segment size:
    segmented_stock_list = [stock_list[i:i + segment_size] for i in range(0, len(stock_list), segment_size)]
    no_segments = len(segmented_stock_list)
    # START -----------------------------------------------------------------------------------------------------------------------------------
    start_time = time.time()
    for i in range(no_segments):
        target_stock_list = segmented_stock_list[i]
        # request for data
        clear_output(wait=True)
        if(save_excel):
            print('Reading Data : ', f'{round((i)/no_segments*100,1)} %', '   Processing and Saving Data : ', f'{round((i)/no_segments*100,1)} %')
        else:
            print('Reading Data : ', f'{round((i)/no_segments*100,1)} %', '   Processing Data : ', f'{round((i)/no_segments*100,1)} %')
        text_resp = __get_history_data_group_parallel__(target_stock_list)
        clear_output(wait=True)
        if(save_excel):
            print('Reading Data : ', f'{round((i+1)/no_segments*100,1)} %', '   Processing and Saving Data : ', f'{round((i)/no_segments*100,1)} %')
        else:
            print('Reading Data : ', f'{round((i+1)/no_segments*100,1)} %', '   Processing Data : ', f'{round((i)/no_segments*100,1)} %')
        # process the data:
        if(i==0):
            df_panel = __build_price_panel_seg__(df_response=text_resp, param = param, save_excel = save_excel, save_path = save_path)
        else:
            df_panel = pd.concat([df_panel,__build_price_panel_seg__(df_response=text_resp, param = param,save_excel = save_excel, save_path = save_path)],axis=1)
        clear_output(wait=True)
        if(save_excel):
            print('Reading Data : ', f'{round((i+1)/no_segments*100,1)} %', '   Processing and Saving Data : ', f'{round((i+1)/no_segments*100,1)} %')
        else:
            print('Reading Data : ', f'{round((i+1)/no_segments*100,1)} %', '   Processing Data : ', f'{round((i+1)/no_segments*100,1)} %')
    # END -----------------------------------------------------------------------------------------------------------------------------------
    # add jalali date and drop date if necessary
    if(jalali_date):
        df_panel = df_panel.reset_index()
        df_panel['J-Date']=df_panel['Date'].apply(lambda x: str(jdatetime.date.fromgregorian(date=x.date())))
        df_panel = df_panel.set_index('J-Date')
        df_panel.drop(columns=['Date'],inplace=True)
    # save options:
    if(save_excel):
        today_j_date = jdatetime.datetime.now().strftime("%Y-%m-%d")
        try:
            df_panel.to_excel(save_path+today_j_date+' Price Panel'+'.xlsx')
        except:
            print('Save path does not exist, you can handle saving this data by returned dataframe as Excel using ".to_excel()", if you will!')
    # final messages to user: time of running:
    end_time = time.time()
    print(str(int(round(end_time-start_time,0)))+ ' Seconds Took to Gather and Process Your Requested Data')
    return df_panel