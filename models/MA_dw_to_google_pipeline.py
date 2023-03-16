#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@author:  marios
"""
import json
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib
import functools
import operator
import os
import os.path as op
import numpy as np
import pytrends
from pytrends.request import TrendReq as UTrendReq
from datetime import date
import datetime as d
from collections import Counter


GET_METHOD='get'
headers = {
    'authority': 'trends.google.com',
    'accept': 'application/json, text/plain, */*',
    'accept-language': 'el-GR,el;q=0.9,en;q=0.8,es;q=0.7',
    'content-type': 'application/json;charset=UTF-8',
    'cookie': '__utma=10102256.1937595387.1677588086.1677588086.1678441622.2; __utmc=10102256; __utmz=10102256.1678441622.2.2.utmcsr=trends.google.com|utmccn=(referral)|utmcmd=referral|utmcct=/; __utmt=1; __utmb=10102256.13.9.1678442016068; CONSENT=YES+GB.en-GB+; HSID=AwrWd8APwv-yBWgzh; SSID=AeXCoum7ArBP5_-aa; APISID=CH4IjtEJhVzXdXGB/AFPE6uuFtOUDiSjnb; SAPISID=FcPgZF83fs0zxFml/Ad59_bwdrgg_kZ4q4; __Secure-1PAPISID=FcPgZF83fs0zxFml/Ad59_bwdrgg_kZ4q4; __Secure-3PAPISID=FcPgZF83fs0zxFml/Ad59_bwdrgg_kZ4q4; SID=TwhPHvTugfJu62Xh-HCJkOIPdoEDrL6q-6Eu9itbEI8mmKw9wzJdgT6c48lYdsNyN4E5xA.; __Secure-1PSID=TwhPHvTugfJu62Xh-HCJkOIPdoEDrL6q-6Eu9itbEI8mmKw9mFTrJ0j2r8zMRcq3v-A7Dg.; __Secure-3PSID=TwhPHvTugfJu62Xh-HCJkOIPdoEDrL6q-6Eu9itbEI8mmKw9xQIlYIR6TyZD2qXkeuopSA.; OGPC=19031986-1:; AEC=ARSKqsLpZW_sbZN2NdijlA8HPzuRHa1TPtYLHLGgaOIZpt8oJZL9PYZZYQ; SEARCH_SAMESITE=CgQI4ZcB; 1P_JAR=2023-03-10-09; NID=511=bYRTpZST7bJyL0z371h4Y79EMA1j9QqQFUpi8vJsSmiWdINx5gKruSDljEBAFfs9FYsxRrmP7vulT_MdtU2xEXQSW837vsgNY9s0i2WZAeFETmMEDrju3d_HgA2Wxy5DrFrIOaOiFu6LkpD7pY4wF4qrMZ38BzvW4NkYX_fUI7bFzHXsg24iHara1hPmPIXOSl6wQgsssfGHUntOI9PgY_eXaAEJbY7VgTr1hjNvEDlFSYOuzLvHSzo9kX9ALXA5-WOICbuLdAucZc3hJKo1dUKM51JCkzLsUHm99MWA86Icz-dmMW8ybQZhEUd2YgsBHHn5MV8uSVpcZ53n4_KL7r6sOpfWZ0ZXairmL3NH-hHz4Vyq; _gid=GA1.3.1682047475.1678441583; OTZ=6935626_48_48_123900_44_436380; _gat_gtag_UA_4401283=1; _ga=GA1.3.1937595387.1677588086; SIDCC=AFvIBn_I_znBUYDEoxfE1jUbrp_F8T607DZhlzI9o_gQoZmA4OxNjglOrH8Q8er3Cv4uzoWYkX9Z; __Secure-1PSIDCC=AFvIBn_Nhc9nywxJ_UrRYogvErcX48ygHEiBzjRRZtPe-mIwBTe_M7UbvKR4d-rAuhYyGJi-Dm0; __Secure-3PSIDCC=AFvIBn8vpeAOp5e0oAWBAETEzSClsyQlm3vQJhAQP7T7Z51q1K7zHDm_-CSGFEPasFw0sRHoJDU; _ga_VWZPXDNJJB=GS1.1.1678441583.2.1.1678442016.0.0.0',
    'origin': 'https://trends.google.com',
    'referer': 'https://trends.google.com/trends/explore?date=now%201-d&q=Adele&hl=en-GB',
    'sec-ch-ua': '"Chromium";v="110", "Not A(Brand";v="24", "Google Chrome";v="110"',
    'sec-ch-ua-arch': '"x86"',
    'sec-ch-ua-bitness': '"64"',
    'sec-ch-ua-full-version': '"110.0.5481.177"',
    'sec-ch-ua-full-version-list': '"Chromium";v="110.0.5481.177", "Not A(Brand";v="24.0.0.0", "Google Chrome";v="110.0.5481.177"',
    'sec-ch-ua-mobile': '?0',
    'sec-ch-ua-model': '',
    'sec-ch-ua-platform': '"macOS"',
    'sec-ch-ua-platform-version': '"13.2.1"',
    'sec-ch-ua-wow64': '?0',
    'sec-fetch-dest': 'empty',
    'sec-fetch-mode': 'cors',
    'sec-fetch-site': 'same-origin',
    'user-agent': 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/110.0.0.0 Safari/537.36',
    'x-client-data': 'CIq2yQEIprbJAQjEtskBCKmdygEIkufKAQiVocsBCPyqzAEI9/XMAQib/swBCI6MzQEIlZbNAQiols0BCOGXzQEI5JfNAQjzl80BCMyYzQEI2JjNAQjzmc0BCLSazQEI0uGsAg==',
}

class TrendReq(UTrendReq):
    def _get_data(self, url, method=GET_METHOD, trim_chars=0, **kwargs):
        return super()._get_data(url, method=GET_METHOD, trim_chars=trim_chars, headers=headers, **kwargs)


def load_data(data_dir, keep=['id', 'keywordStrings', 'lastModifiedDate', 'categories']):
    # Opening JSON file
    f = open(data_dir)
    # returns JSON object as a dictionary
    data = json.load(f)
    df = pd.DataFrame.from_dict(data)
    df = df.sort_values(by= 'lastModifiedDate')
    df = df.reset_index()

    #drop all columns apart from ones necessary
    df = df[keep]

    return df

def truncate_data(df): #TODO needs fixing after sorting by dates
    datetimes = pd.to_datetime(df['lastModifiedDate'])
    df['ts_lastModifiedDate']=datetimes
    #find start index for subset 2019-2022
    ts_start=datetimes[(datetimes > pd.Timestamp(year=2019, month=1, day=1).tz_localize('utc')) 
            & (datetimes < pd.Timestamp(year=2019, month=1, day=2).tz_localize('utc'))].min()
    print('ts_start', ts_start)
    #find end date for subset 2019-2022
    ts_end=datetimes[(datetimes > pd.Timestamp(year=2022, month=1, day=1).tz_localize('utc')) 
            & (datetimes < pd.Timestamp(year=2022, month=1, day=2).tz_localize('utc'))].min()
    print('ts_end', ts_end)
    start_date=datetimes[datetimes == ts_start]
    end_date=datetimes[datetimes == ts_end]
    #find index for the chosen start and end dates
    start_index=start_date.index[0]
    print(start_index)
    df[df.index == start_date.index[0]]
    end_index=end_date.index[0]
    print(end_index)
    df[df.index == end_date.index[0]]
    df_subset=df[start_index:end_index]

    return df_subset

def get_interest_over_time(keywords):#, start_date = '2019-01-01', end_date=f'{date.today()}'):
    #keywords needs to be a list 
    #need to make sure the total number of characters is less than 100 for Google
    print('attempt')
    terms = 0
    chars = 0
    for word in keywords:
        chars += len(word)
        if chars > 99:
            break
        else:
            terms += 1
    if terms > 5:
        terms = 5
    keywords = keywords[:terms]

    #let's get python trends 
    print(keywords)
    print(sum([len(i) for i in keywords]))

    pytrend = TrendReq()
    google_df = pytrend.build_payload(kw_list= keywords)#, timeframe= '{} {}'.format(str(start_date),str(end_date)))
    print('payload done')
    google_df = pytrend.interest_over_time()
    print('innt oveere timee done')
    google_df = google_df.drop('isPartial', axis = 'columns')
    # turn the df into a dict so we can save it in the bigger 
    google_dict = google_df.to_dict()

    return google_dict
    

@click.group()
def cli():
    pass

@click.command(name='dw-to-google')
@click.argument('db_path', type=click.Path(exists=True))
@click.option('--output', '-o', default=None)
def cli_dw_to_google(db_path=None,
                output=None):

    '''This function takes in the DW keywords and iterativeely 
    searchs for them for historical data on the Google API
    
    Then it compares the two signals.
    Need to select an appropriate timescale to compare
    Need to select appropriate temporal resolution 
    '''
    
    df = load_data(data_dir)
    #for now let's not truncate- func needs fixing 
    # df = truncate_data(df)
    df = df.loc[:5,:] #let's just take 5 articles for now 

    #let's get the relevant google search histories (saved as dicts), let's use apply to be faster
    df['googleDict'] = df['keywordStrings'].apply(get_interest_over_time)

    for i, el in enumerate(df['googleDict']):
        


    
cli.add_command(cli_dw_to_google)



if __name__ == '__main__':
    cli()
