#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@author:  marios
"""
import json
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import os.path as op
import numpy as np
from pytrends.request import TrendReq as UTrendReq
from datetime import date
import datetime as d
from collections import Counter
import click
import pdb
import traceback
import rapidfuzz.process as rp
#import src functions for getting and cleaning data 
import sys
sys.path.append('../src/')
from data.preprocess_keywords import make_cleaned_keywords_df
from data.make_datasets import get_data

def get_headers():
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
    return headers

class TrendReq(UTrendReq):
    def _get_data(self, url, method='get', trim_chars=0, **kwargs):
        return super()._get_data(url, method='get', trim_chars=trim_chars, headers=get_headers(), **kwargs)

def load_data(data_dir, keep=['id', 'keywordStrings', 'lastModifiedDate', 'categories']):
    # Opening JSON file
    data = pd.read_json(data_dir, orient ='split', compression = 'infer')
    df = pd.DataFrame.from_dict(data)
    df = df.sort_values(by= 'lastModifiedDate')
    df = df.reset_index()
    #drop all columns apart from ones necessary
    df = df[keep]
    return df

def truncate_data(df, start_date, end_date): 
    df['dt_lastModifiedDate'] = df.lastModifiedDate.apply(lambda x: d.datetime.strptime(x[:10], '%Y-%m-%d') if x is not None else x)
    start_dt = d.datetime.strptime(start_date, '%Y-%m-%d')
    end_dt = d.datetime.strptime(end_date, '%Y-%m-%d')
    mask = np.logical_and(df['dt_lastModifiedDate']>=start_dt, df['dt_lastModifiedDate']<end_dt)
    df_subset = df[mask]
    return df_subset

def clean_data(data_file, start_date = '2019-01-01', end_date = '2022-01-01'):
    ##CHANGE THESE DATES IF YOU WANT TO PRE-PROCESS A LARGER PART OF THEE DATASET
    # Load and extract data within time range
    df_subset = get_data(data_file, start_date, end_date)
    # Cleans keywords and saves data as a dataframe
    clean_df = make_cleaned_keywords_df(df_subset, start_date, end_date)
    return clean_df

def extract_keywords(df):
    keywords = np.asarray(list(set([val for sublist in df['keywordStringsCleanAfterFuzz'] for val in sublist])))
    return keywords

def get_interest_over_time(keyword, start_date = '2019-01-01', end_date=f'{date.today()}'):
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

def get_dw_timeseries(df_clean, keyword, google, start_date = '2019-01-01'):
    # df_clean['present'] = df_clean['keywordStringsCleanAfterFuzz'].apply(lambda x:True if keyword in x else False)
    df_clean['present'] = df_clean['keywordStringsCleanAfterFuzz'].apply(lambda x: any(keyword.lower() in word.lower() for word in x))
    df_clean = df_clean.loc[df_clean['present']]
    df_clean['ts'] = pd.to_datetime(df_clean.lastModifiedDate,format= '%Y-%m-%d' )
    dw_mentions = []
    edges = list(google.index) #extract time indices
    edges.insert(0, start_date) #append the first one manually
    edges = pd.to_datetime(edges,format= '%Y-%m-%d' ) # turn them all into pandas timestamps
    for i, (start, end) in enumerate(zip(edges[:-1], edges[1:])):
        mask = np.logical_and(df_clean.ts.dt.date >= start, df_clean.ts.dt.date<end)
        dw_mentions.append(np.sum(mask))
    google['dw'] = dw_mentions
    google = google.rename(columns={keyword: 'google'})

    return google

def plot_signals(mixed_df, fig, ax, keyword = 'keyword'):
    fig.autofmt_xdate(rotation=75)
    ax.plot(mixed_df.index, mixed_df.dw, color = 'k')
    # ax.set_xticks(dw_mentions.index.astype(str)[::8])
    ax.set_xlabel('Time', fontsize = 20)
    ax.set_ylabel(f'DW Articles per week with {keyword} in keywords', color = 'k', fontsize = 20)
    ax2 = ax.twinx()
    ax2.plot(mixed_df.index, mixed_df.google, color = 'r', alpha =0.5) #may need to add .astype(str)
    ax2.set_ylabel(f'Relative amount of Google Searches for {keyword}', color = 'r', fontsize = 20)
    mixed_df.google = mixed_df.google.apply(lambda x: x-mixed_df.google.mean())
    mixed_df.dw = mixed_df.dw.apply(lambda x: x-mixed_df.dw.mean())

    # gc_res = grangercausalitytests(mixed_df, 5)
    #look into gc_res and find the best shift
    #print it onto the plot if it's signifcant 

    return fig, ax

def remove_spaces(keyword):
    while keyword[-1] == ' ':
        keyword = keyword[:-1]
    return keyword

def find_similar_keywords(keyword, df):
    keywords = np.asarray(list(set([val for sublist in df['keywordStringsCleanAfterFuzz'] for val in sublist])))
    similarities = rp.cdist([keyword], keywords).ravel()
    #get all above 90 and return theem if there are some
    similar_keywords = keywords[similarities>=90]
    #append the highest 1 if not  
    if similar_keywords.shape[0]==0:
        similar_keywords = keywords[similarities == np.max(similarities)]
    return list(similar_keywords)


@click.group()
def cli():
    pass

# @click.command(name='extract-google-trends')
# @click.argument('db_path', type=click.Path(exists=True))
# @click.option('--output', '-o', default=None)
# @click.option('--overwrite', default=False)
# def cli_extract_google_trends(db_path=None,
#                         output=None,
#                         overwrite=False
#                         ):
#     '''This function takes in the DW keywords and iterativeely 
#     searchs for them for historical data on the Google API
#     then outputs a new df containing all of them 
    
#     Need to select appropriate temporal resolution. for this amount of time, google gives weekly searches by default
#     '''
#     #UNCOMMENT WHEN READY TO RUN PROPERLY
#     df = pd.read_json(db_path, orient ='split', compression = 'infer')
#     if 'keywordStringsCleanAfterFuzz' not in df.columns or overwrite:
#         #let's load entire dataset and create keyword list from scratch 
#         df = load_data(db_path, keep=['id', 'keywordStrings', 'lastModifiedDate'])
#         df = truncate_data(df) #let's keep entiree dataset for now?
#         df = clean_df(df)
#         clean_file_name = 'cleaned_df.npz'
#         np.save(op.join(output, clean_file_name), df)
#     else:
#         # df = df[['id', 'cleanedKeywords', 'lastModifiedDate']]
#         pass

#     #make keywords into df and use .apply to get google dicts 
#     df = df.loc[0:5, :] #slicing just to test
#     keywords = list(itertools.chain(*list(df['keywordStringsCleanAfterFuzz'])))
#     kw_df = pd.DataFrame({'keywords':keywords})
#     # pdb.set_trace()
#     kw_df['google_trends'] = kw_df['keywords'].apply(get_interest_over_time)#, args =(start_date='2019-01-01', end_date='2022-01-01')

#     #TODO: decide if you will make many dataframes inside original df or one big one with dates an index- 
#     # thee second is easy to get from the first one
#     # e.g.: new_big_df = pd.DataFrame({kw_df['keywords']: kw_df['google_trends']})
    
#     google_kw_file_name = 'dw_keywords_google_searches.npz'
#     print('DFs with google searches saved at: ', op.join(output,google_kw_file_name))
#     np.savez(op.join(output, google_kw_file_name), kw_df)

# cli.add_command(cli_extract_google_trends)



@click.command(name='dw-vs-google')
@click.option('--df_clean_path', type=click.Path(exists=True))
@click.option('--output', '-o', default=None)
@click.option('--overwrite', '-O', default=False)
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
    if df_clean_path is None or overwrite:
        data_dir = '/home/marios/data/CMS_2010_to_June_2022_ENGLISH.json'
        df_clean = clean_data(data_dir)
    elif df_clean_path:
        try:
            df_clean = load_data(df_clean_path, keep=['id', 'lastModifiedDate', 'keywordStringsCleanAfterFuzz'])
        except:
            traceback.print_exc()
            print('This dataset has not been cleaned! Please clean dataset before running this function or add option --overwrite to clean now.')
            return -1

    unique_kws = extract_keywords(df_clean)
    keyword = remove_spaces(str(input('Please input keyword to be analyzed:\n')).lower())
    while keyword not in unique_kws:
        similar_kws = find_similar_keywords(keyword, df_clean)
        keyword = input(f'This keyword is not in the data analyzed. Please select a different one. Suggestions: {similar_kws}\n')
        keyword = remove_spaces(str(keyword.lower()))
    start_date = str(input('Please input start date (YYYY-MM-DD):\n'))
    end_date = str(input('Please input end date (YYYY-MM-DD):\n'))
    df_clean = truncate_data(df_clean, start_date, end_date)

    google_searches = get_interest_over_time(keyword, start_date = start_date, end_date=end_date)
    #theen get dw mentions binned into the google dates output 
    mixed_df = get_dw_timeseries(df_clean, keyword, google_searches, start_date = start_date)

    fig, axar = plt.subplots(nrows =1, ncols =1, figsize=(15,10))
    fig, axar = plot_signals(mixed_df, fig, axar, keyword = keyword)
    
    # 2) Granger 'Causality': do dw articles follow closely after google searches
    # gc_res = grangercausalitytests(mix_df, 5)
    # save it
    file_name = f'{keyword}_dw_vs_google.pdf'
    fig.savefig(op.join(output,file_name))


cli.add_command(cli_dw_vs_google)



if __name__ == '__main__':
    cli()