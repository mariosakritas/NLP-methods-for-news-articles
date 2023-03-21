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
import itertools
import operator
import os
import os.path as op
import numpy as np
import pytrends
from pytrends.request import TrendReq as UTrendReq
from datetime import date
import datetime as d
from collections import Counter
import click
import pdb
import sys, traceback


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
    data = pd.read_json(data_dir, orient ='split', compression = 'infer')
    df = pd.DataFrame.from_dict(data)
    df = df.sort_values(by= 'lastModifiedDate')
    df = df.reset_index()
    #drop all columns apart from ones necessary
    df = df[keep]
    return df

def truncate_data(df, start_date, end_date): #TODO ?needs fixing to include specific dates 
    df['dt_lastModifiedDate'] = df.lastModifiedDate.apply(lambda x: d.datetime.strptime(x[:10], '%Y-%m-%d') if x is not None else x)
    start_dt = d.datetime.strptime(start_date, '%Y-%m-%d')
    end_dt = d.datetime.strptime(end_date, '%Y-%m-%d')
    mask = np.logical_and(df['dt_lastModifiedDate']>=start_dt, df['dt_lastModifiedDate']<end_dt)
    df_subset = df[mask]
    return df_subset

#TODO fill in this funnction based on Anyas and Magda's input

# def clean_df(df):
#     return clean_df

def extract_keywords(df):
    keywords = [val for sublist in df['cleanKeywords'] for val in sublist]
    return keywords

def get_interest_over_time(keyword, start_date = '2019-01-01', end_date=f'{date.today()}'):
    #keywords needs to be a list 
    #need to make sure the total number of characters is less than 100 for Google and terms are fewer than 5 
    # terms = 0
    # chars = 0
    # for word in keywords:
    #     chars += len(word)
    #     if chars > 99:
    #         break
    #     else:
    #         terms += 1
    # if terms > 5:
    #     terms = 5
    # keywords = keywords[:terms]
    print(keyword)
    if len(keyword)>99:
        print('KEYWORD IS TOO LONG FOR THIS SEARCH')
        return None
    
    #let's get python trends 
    pytrend = TrendReq(                    
    # proxies=['https://34.203.233.13:80','https://35.201.123.31:880'], 
    # #hl='en-US', tz=360, timeout=(10,25)
    retries=2, backoff_factor=0.1, requests_args={'verify':False})
    google_df = pytrend.build_payload(kw_list= [keyword], timeframe= '{} {}'.format(str(start_date),str(end_date)))
    google_df = pytrend.interest_over_time()
    if 'isPartial' in google_df.columns:
        google_df = google_df.drop('isPartial', axis = 'columns')

    return google_df

def get_all_weeks(start_dt, end_dt):
    all_weeks = []
    if end_dt.strftime("%Y") < start_dt.strftime("%Y"):
        print('End date should not be before start date. Please select different dates.')
        return -1
    
    elif end_dt.strftime("%Y") > start_dt.strftime("%Y"):
        #need to loop over years then weeks
        for year in range(int(start_dt.strftime("%Y")), int(end_dt.strftime("%Y"))+1):
            if year in [2006, 2012, 2017, 2023]:
                weeks = 53
            else:
                weeks = 52
            for week in range(weeks+1):
                all_weeks.append(str(year)+str(week).zfill(2))
        #now let's remove the ones before and after the required week
        all_weeks = np.asarray(sorted([int(i) for i in all_weeks]))
        mask = np.logical_and(all_weeks>=int(str(start_dt.strftime("%Y"))+str(start_dt.strftime("%W"))), 
                              all_weeks<int(str(end_dt.strftime("%Y"))+str(end_dt.strftime("%W"))))
        all_weeks = all_weeks[mask]
        return [str(i) for i in all_weeks]
    
    elif end_dt.strftime("%Y") == start_dt.strftime("%Y"):
        #we just need to loop over weeks 
        if int(start_dt.strftime("%Y")) in [2006, 2012, 2017, 2023]:
            weeks = 53
        else:
            weeks = 52
        for week in range(int(start_dt.strftime("%W")), int(end_dt.strftime("%W"))+1):
            all_weeks.append(str(start_dt.strftime("%Y"))+str(week).zfill(2))
        return all_weeks



    return sorted(all_weeks)

def get_dw_timeseries(df_clean, keyword, resolution = 'weekly', start_date = '2019-01-01', end_date=f'{date.today()}'):
    #TODO: check this function after replacing loop 
    
    start_dt = d.datetime.strptime(start_date, '%Y-%m-%d')
    end_dt = d.datetime.strptime(end_date, '%Y-%m-%d')

    not_keyword_indices = [] #TODO: do this without a loop (create extra boolean column and assign True if keyword is there)
    for i, row in enumerate(df_clean['keywordStringsCleanAfterFuzz']):
        if keyword not in row:
            not_keyword_indices.append(i)

    df_clean = df_clean.drop(not_keyword_indices)
    df_clean['datetimes']= pd.to_datetime(df_clean['lastModifiedDate'])
    df_clean['yearweek'] = df_clean['dt_lastModifiedDate'].apply(lambda x: str(x.strftime("%Y"))+str(x.strftime("%W")))

    all_weeks = get_all_weeks(start_dt, end_dt)
    not_in_df = list(set(all_weeks) - set(df_clean['yearweek'].tolist()))
    dw_mentions = dict(Counter(df_clean['yearweek'].tolist()))
    for key_ in not_in_df:
        dw_mentions[key_] = 0 
    
    df_dw_mentions = pd.DataFrame.from_dict(dw_mentions, orient='index', columns=['val'])
    # df_dw_mentions['week_str'] = [str(i) for i in df_dw_mentions.index]
    # df_dw_mentions = df_dw_mentions.sort_values(by='week_str')
    #TODO: can we sort without making a new column etc. this is gonna be computationally expensive
    df_dw_mentions.index = df_dw_mentions.index.astype(int)
    df_dw_mentions = df_dw_mentions.sort_index()

    return df_dw_mentions

def plot_signals(dw, google, ax, keyword):
    ax.bar(dw.index, dw.val, color = 'grey')
    ax.set_xticks(dw.index[::4], rotate = 60)
    ax.set_xlabel('Time', fontsize = 15)
    ax.set_ylabel(f'DW Articles per week with {keyword} in keywords', color = 'grey', fontsize = 15)
    ax2 = ax.twinx()
    ax2.plot(np.arange(0,len(google.values)), google.values, color = 'r')
    ax2.set_ylabel('Relative amount of Google Searches', color = 'r', fontsize = 15)
    return ax

@click.group()
def cli():
    pass

@click.command(name='extract-google-trends')
@click.argument('db_path', type=click.Path(exists=True))
@click.option('--output', '-o', default=None)
@click.option('--overwrite', default=False)
def cli_extract_google_trends(db_path=None,
                        output=None,
                        overwrite=False
                        ):
    '''This function takes in the DW keywords and iterativeely 
    searchs for them for historical data on the Google API
    then outputs a new df containing all of them 
    
    Need to select appropriate temporal resolution. for this amount of time, google gives weekly searches by default
    '''
    #UNCOMMENT WHEN READY TO RUN PROPERLY
    df = pd.read_json(db_path, orient ='split', compression = 'infer')
    if 'keywordStringsCleanAfterFuzz' not in df.columns or overwrite:
        #let's load entire dataset and create keyword list from scratch 
        df = load_data(db_path, keep=['id', 'keywordStrings', 'lastModifiedDate'])
        # df = truncate_data(df) #let's keep entiree dataset for now?
        df = clean_df(df)
        clean_file_name = 'cleaned_df.npz'
        np.save(op.join(output, clean_file_name), df)
    else:
        # df = df[['id', 'cleanedKeywords', 'lastModifiedDate']]
        pass

    #make keywords into df and use .apply to get google dicts 
    df = df.loc[0:5, :] #slicing just to test
    keywords = list(itertools.chain(*list(df['keywordStringsCleanAfterFuzz'])))
    kw_df = pd.DataFrame({'keywords':keywords})
    # pdb.set_trace()
    kw_df['google_trends'] = kw_df['keywords'].apply(get_interest_over_time)#, args =(start_date='2019-01-01', end_date='2022-01-01')

    #TODO: decide if you will make many dataframes inside original df or one big one with dates an index- 
    # thee second is easy to get from the first one
    # e.g.: new_big_df = pd.DataFrame({kw_df['keywords']: kw_df['google_trends']})
    
    google_kw_file_name = 'dw_keywords_google_searches.npz'
    print('DFs with google searches saved at: ', op.join(output,google_kw_file_name))
    np.savez(op.join(output, google_kw_file_name), kw_df)

cli.add_command(cli_extract_google_trends)



@click.command(name='dw-vs-google')
@click.argument('df_clean_path', type=click.Path(exists=True))
@click.option('--output', '-o', default=None)
@click.option('--overwrite', default=False)
def cli_dw_vs_google(df_clean_path=None, # need this to make the timeseries
                        output=None,
                        overwrite=False
                        ):
    '''
    This function takes in a single keyword
    creates a timeseries of mentions across time for them 
    and compares them to google searches also extracted 

    input:
    keyword (str)
    clean df (whole dataset's keywords) 

    output:
    graph with metrics etc. 
    
    TODO select an appropriate timescale to compare --> weeks?
    TODO select/develop multiple metrics to explore 
    TODO develop creative visualization methods 
    '''
    #let's load dataset: this should be clean from previous function
    try:
        df_clean = load_data(df_clean_path, keep=['id', 'lastModifiedDate', 'keywordStringsCleanAfterFuzz'])
    except:
        traceback.print_exc()
        print('This dataset has not been cleaned! Please clean dataset before running this function.')
        return -1

    pre_keyword = str(input('Please input keyword to be analyzed:\n'))
    start_date = str(input('Please input start date (YYYY-MM-DD):\n'))
    end_date = str(input('Please input end date (YYYY-MM-DD):\n'))
    df_clean = truncate_data(df_clean, start_date, end_date)
    # keyword = fuzzy_wuzzy(df_clean, pre_keyword)
    # print(f'searching for: {keyword}')
    keyword = pre_keyword

    dw_mentions = get_dw_timeseries(df_clean, keyword, start_date = start_date, end_date=end_date)
    google_searches = get_interest_over_time(keyword, start_date = start_date, end_date=end_date)


    if not google_searches: #you may need to change this to check if the df is empty 
        print('Exiting...')
        return -1
    #datetime_object saved in column 'dt_lastModifiedDate' of df

    assert dw_mentions.shape[0] == google_searches.values.shape[0]
    #TODO: we need a perfect way for matching dates between the two
    # at the moment we have year+no_of_week in dw and actual date on google 01.01.19, 08.01.2019  etc.)

    #now let's compare for this keyword:
    # 1) let's make plot 
    fig, ax = plt.subplots(figsize(15,10))
    plot_signals(dw_mentions, google, ax)
    
    # ax.plot(dw_mentions.val.values - np.mean(dw_mentions.val.values))
    # ax.plot(google_searches.values - np.mean(google_searches.values))

    # 2) Granger 'Causality': do dw articles follow closely after google searches
    # google_searches = google_searches.values
    # mix_df = pd.DataFrame({'dw':dw, 'google':google})
    # gc_res = grangercausalitytests(mix_df, 5)
        # TODO: find any significant results and print/ report 
        # TODO: implement more metrics to compare 
        # TODO: plot graphs and metrics in a pdf?

    # save it
    output_dir = '/home/marios/S2DS/Spring23_DW/reports/figures'
    file_name = f'{keyword}_dw_vs_google.pdf'
    fig.savefig(op.join(output_dir,file_name))
    plt.show()
    #TODO: decide if you will impose a limit on how many mentions and above you will create timeseries (based on output examples)


cli.add_command(cli_dw_vs_google)




#saving these for latere potentially 

# #let's load and loop over unique cleaned keywords:
#     kw_google_df = np.load(kw_google_path)
#     for i, (keyword, google_searches) in enumerate(zip(kw_google_df['keywords'], kw_google_df['google_trends'])):
#         #TODO: reemove this loop if we can? maybe not                                  
#         print(keyword)
#         #now let's use this word on the original dataset to get time series
#         dw_mentions  = get_dw_timeseries(clean_df, keyword)

#         assert dw_mentions.shape[0] == google_searches.values.shape[0]



if __name__ == '__main__':
    cli()