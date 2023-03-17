import pandas as pd
from thefuzz import process


#uni_kw=pd.read_csv('../data/interim/out_2019-2021_keywords_before_FuzzyWuzzy.csv')
df_subset=pd.read_csv('../data/interim/clean_keywords_2019-2021_before_FuzzyWuzzy.csv')

#create series of keywords sets
def get_keywords(row):
    if row is None:
        return None
    else:
        res_set = set()
        for name_dict in row:
            res_set.add(name_dict['name'])
        return res_set

#df['keywords'].apply(get_keywords)
sets=df_subset['keywordStringsClean'].apply(get_keywords)  #2019-2021 subset
#sets=sets[0:10000] #10000 articles
 
unique_keywords=list(functools.reduce(set.union, sets))



ded_kw=process.dedupe(unique_keywords, threshold = 90)

pd.Series(list(ded_kw)).to_csv('../data/interim/out_dedupl_90_1703.csv')