import pandas as pd 
import json
from pytrends.request import TrendReq as UTrendReq
from datetime import date
from pytrends.exceptions import ResponseError
import time
import numpy as np
from random import randint
import os

def get_data(data_file, start_date, end_date):

    """
    Loads the DW data specified in data_file (json file)
    Returns a data frame with the data between start_date and end_date (format: 'yyyy-mm-dd')

    """

    print('Getting data...')
          
    # Opening JSON file
    f = open(data_file)
    
    # returns JSON object as a dictionary
    data = json.load(f)

    # convert to data frame
    df = pd.DataFrame.from_dict(data)

    print('Loading data DONE. Number of articles is', len(df))

    # sorts dataframe by date
    df = df.sort_values(by='lastModifiedDate') 

    # make a column with only date
    df['Date'] = pd.to_datetime(df['lastModifiedDate']).apply(lambda x: x.date)

    # creates a mask (list of True or False) for indices within the specified date range
    mask = (pd.to_datetime(df['lastModifiedDate']) > start_date) & (pd.to_datetime(df['lastModifiedDate']) <= end_date)
    
    # extracts subset from this dataset
    df_subset = df.loc[mask]

    # Outputs length of unique keywords before and after
    print('Extracting data DONE. Number of articles from', start_date, 'to', end_date, 'is', len(df_subset))

    return df_subset


def get_interest_over_time(keyword, start_date = '2019-01-01', end_date=f'{date.today()}'):
    
    """
    Extracts Google trends over time for a specific keyword, and returns the dataframe

    """

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


class TrendReq(UTrendReq):
    def _get_data(self, url, method='get', trim_chars=0, **kwargs):
        return super()._get_data(url, method='get', trim_chars=trim_chars, headers=get_headers(), **kwargs)

# Header is required to avoid having too many requests on Google
def get_headers():
    headers = {
    'authority': 'trends.google.com',
    'accept': 'application/json, text/plain, */*',
    'accept-language': 'en-US,el;q=0.9,en;q=0.8,es;q=0.7',
    'content-type': 'application/json;charset=UTF-8',
    'cookie': '__utma=10102256.1937595387.1677588086.1677588086.1678441622.2; __utmc=10102256; __utmz=10102256.1678441622.2.2.utmcsr=trends.google.com|utmccn=(referral)|utmcmd=referral|utmcct=/; __utmt=1; __utmb=10102256.13.9.1678442016068; CONSENT=YES+GB.en-GB+; HSID=AwrWd8APwv-yBWgzh; SSID=AeXCoum7ArBP5_-aa; APISID=CH4IjtEJhVzXdXGB/AFPE6uuFtOUDiSjnb; SAPISID=FcPgZF83fs0zxFml/Ad59_bwdrgg_kZ4q4; __Secure-1PAPISID=FcPgZF83fs0zxFml/Ad59_bwdrgg_kZ4q4; __Secure-3PAPISID=FcPgZF83fs0zxFml/Ad59_bwdrgg_kZ4q4; SID=TwhPHvTugfJu62Xh-HCJkOIPdoEDrL6q-6Eu9itbEI8mmKw9wzJdgT6c48lYdsNyN4E5xA.; __Secure-1PSID=TwhPHvTugfJu62Xh-HCJkOIPdoEDrL6q-6Eu9itbEI8mmKw9mFTrJ0j2r8zMRcq3v-A7Dg.; __Secure-3PSID=TwhPHvTugfJu62Xh-HCJkOIPdoEDrL6q-6Eu9itbEI8mmKw9xQIlYIR6TyZD2qXkeuopSA.; OGPC=19031986-1:; AEC=ARSKqsLpZW_sbZN2NdijlA8HPzuRHa1TPtYLHLGgaOIZpt8oJZL9PYZZYQ; SEARCH_SAMESITE=CgQI4ZcB; 1P_JAR=2023-03-10-09; NID=511=bYRTpZST7bJyL0z371h4Y79EMA1j9QqQFUpi8vJsSmiWdINx5gKruSDljEBAFfs9FYsxRrmP7vulT_MdtU2xEXQSW837vsgNY9s0i2WZAeFETmMEDrju3d_HgA2Wxy5DrFrIOaOiFu6LkpD7pY4wF4qrMZ38BzvW4NkYX_fUI7bFzHXsg24iHara1hPmPIXOSl6wQgsssfGHUntOI9PgY_eXaAEJbY7VgTr1hjNvEDlFSYOuzLvHSzo9kX9ALXA5-WOICbuLdAucZc3hJKo1dUKM51JCkzLsUHm99MWA86Icz-dmMW8ybQZhEUd2YgsBHHn5MV8uSVpcZ53n4_KL7r6sOpfWZ0ZXairmL3NH-hHz4Vyq; _gid=GA1.3.1682047475.1678441583; OTZ=6935626_48_48_123900_44_436380; _gat_gtag_UA_4401283=1; _ga=GA1.3.1937595387.1677588086; SIDCC=AFvIBn_I_znBUYDEoxfE1jUbrp_F8T607DZhlzI9o_gQoZmA4OxNjglOrH8Q8er3Cv4uzoWYkX9Z; __Secure-1PSIDCC=AFvIBn_Nhc9nywxJ_UrRYogvErcX48ygHEiBzjRRZtPe-mIwBTe_M7UbvKR4d-rAuhYyGJi-Dm0; __Secure-3PSIDCC=AFvIBn8vpeAOp5e0oAWBAETEzSClsyQlm3vQJhAQP7T7Z51q1K7zHDm_-CSGFEPasFw0sRHoJDU; _ga_VWZPXDNJJB=GS1.1.1678441583.2.1.1678442016.0.0.0',
    'origin': 'https://trends.google.com',
    'referer': 'https://trends.google.com/trends/explore?date=now%201-d&q=Adele&hl=en-US',
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

    # headers = {
    # 'User-Agent': 'Mozilla/5.0 (X11; Linux x86_64; rv:102.0) Gecko/20100101 Firefox/102.0',
    # 'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,image/avif,image/webp,*/*;q=0.8',
    # 'Accept-Language': 'en-US,en;q=0.5',
    # # 'Accept-Encoding': 'gzip, deflate, br',
    # 'Referer': 'https://trends.google.com/trends/explore?geo=US&q=%2Fm%2F0brhd&hl=en-US',
    # 'Alt-Used': 'trends.google.com',
    # 'Connection': 'keep-alive',
    # 'Cookie': 'AEC=AUEFqZcfvz8pPxOry1rz3y1fLbsB5G-C1rUFsrsnGDQmp002SERc73Jw2-Y; CONSENT=PENDING+553; SOCS=CAISHAgCEhJnd3NfMjAyMzAyMjgtMF9SQzIaAmVuIAEaBgiA2pSgBg; NID=511=Vkv-S3JtwGbWx6tl2GL_U1Z4vC_ha0MdnonZ3C4hmswXHdceRg0n8PFadAM6fdUlcqFCCDO1yRTJG95JNy-na-_Qd8nCvepE9o7h8TCljclNySl7yXo3JWe4CLvqPlzqGawItggaqL_GLibBwYqYUTdYm6AhVu-pDnyVYVHh8xsoduPqOynNa552Y7ZikoENOXeN2c3NHtEFIhylA3-x4xHnBvXGhF-bxJaqxqovc_zTd66e0Uy9nA_7Dd14ifp4Nhle8LHUhdGfpbH0PaN3fjLB4sqlcGBT2diKq2ZO; 1P_JAR=2023-3-28-16; SID=VAi_L-D8MkCP55NekTzJ3kdn4YoC1LiPRxuF6xjPuIJYh8apYccZZbqHhtULdlcvflAWBQ.; __Secure-1PSID=VAi_L-D8MkCP55NekTzJ3kdn4YoC1LiPRxuF6xjPuIJYh8apinR4HhLcmjl0MisaG0uLAg.; __Secure-3PSID=VAi_L-D8MkCP55NekTzJ3kdn4YoC1LiPRxuF6xjPuIJYh8apsqFLqC0x_MavCG_fCENcRg.; HSID=A8pYgCcHOR4LMvn4v; SSID=AMWVlViHvnUi59YIo; APISID=kWO6Zgq1x9H43CEW/A5VZDDjcfU3EHo3Fg; SAPISID=dmE2IYITIrnp-sag/AXXyx3jijH-r7Khjd; __Secure-1PAPISID=dmE2IYITIrnp-sag/AXXyx3jijH-r7Khjd; __Secure-3PAPISID=dmE2IYITIrnp-sag/AXXyx3jijH-r7Khjd; SIDCC=AFvIBn_omDtISjcYPIsKjC4he3PtO2Q6Pf5-nzulTmgy0X4AMT8BmibJCDf7AmFcjLNr-UFUnw; __Secure-1PSIDCC=AFvIBn_ItTTD_2tNINCuyb4rST3m1ctLJUy-KDkLQcMkRH15mwXH3pOsVDeDLvhYNOx23bGGlw; __Secure-3PSIDCC=AFvIBn-OQFzfSsmHHUKizPS6NZFRaxaODvuwGhbzjry03Lkose9-DHHZUOR859TkiiyybJp2PA; SEARCH_SAMESITE=CgQI3pcB; _ga_VWZPXDNJJB=GS1.1.1680709849.1.1.1680710271.0.0.0; _ga=GA1.3.491586339.1680709850; _gid=GA1.3.1164332066.1680709850; OTZ=6973431_52_56_123900_52_436380; CONSISTENCY=AKJVzcrhPoZEgK5uS19oLNI-JHA4QUz2s7VVHGQ-xeqd-Hika8X6oIRyXrvTDV9MrmWl4SNqK9ldVLJ-fh4bmtDo22hKZIIhW8XNs2XRAdiIgW6KDxJaj8NJXEMnjWMlOfC8Yb8jsrHI; _gat_gtag_UA_4401283=1',
    # 'Upgrade-Insecure-Requests': '1',
    # 'Sec-Fetch-Dest': 'document',
    # 'Sec-Fetch-Mode': 'navigate',
    # 'Sec-Fetch-Site': 'same-origin',
    # 'Sec-Fetch-User': '?1',
    # # Requests doesn't support trailers
    # # 'TE': 'trailers',
    # }

    return headers
    
def get_daily_trending_searches(filepath = '../data/interim/',start_date = '2019-01-01', end_date = '2019-01-02', geography=''):

    """
    Extracts the most trending topics on Google for a specified time range and saves the dataframe as json file

    """

    start_date_strip = ''.join(start_date.split('-'))
    end_date_strip = ''.join(end_date.split('-'))

    daily_trending_searches = pd.DataFrame()

    pytrend = TrendReq(                    
    # proxies=['https://34.203.233.13:80','https://35.201.123.31:880'], 
    # #hl='en-US', tz=360, timeout=(10,25)
    retries=2, backoff_factor=0.1, requests_args={'verify':False})

    for i in [dr.strftime('%Y-%m-%d') for dr in pd.date_range(start_date_strip, end_date_strip)]:
        try:
            pytrend.build_payload(kw_list=[''], timeframe=i + ' ' + i, geo=geography)
            df_rt_test = pytrend.related_topics()
            data = df_rt_test['']['rising']
            time.sleep(randint(1,100))
            print(i)
        except ResponseError:
            print('Timeout')
        daily_trending_searches = daily_trending_searches.append(data)


    daily_trending_searches.reset_index(drop=True, inplace=True)
    daily_trending_searches['date'] = daily_trending_searches['link'].apply(lambda x: x.split('&')[1].split('date=')[1].split('+')[0])
    daily_trending_searches['location'] = geography if geography != '' else 'World'
    location = geography if geography != '' else 'World' 

    # storing the data in JSON format
    daily_trending_searches.to_json(filepath + start_date + '_' + end_date + '_' + location + '_daily_trending_searches.json', orient = 'split', compression = 'infer', index = 'true')

    return daily_trending_searches




def get_dw_timeseries(df_clean_input, keyword, google, start_date = '2019-01-01'):
    df_clean = df_clean_input.copy()
    df_clean['present'] = df_clean['keywordStringsCleanAfterFuzz'].apply(lambda x: any(keyword.lower() in word.lower() for word in x))
    #df_clean['present'] = df_clean['keywordStringsCleanAfterFuzz'].apply(lambda x:True if keyword in x else False)
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


def save_txt(out_name,data_to_save):
    from tqdm import tqdm
    #!rm $out_name
    if os.path.isfile(out_name):
        os.remove(out_name)
    with open(out_name,'a') as f:
        for i,el in tqdm(enumerate(data_to_save)):
            print(el,file=f)
    input_file=out_name
    return input_file
    