import pandas as pd
from thefuzz import process


uni_kw=pd.read_csv('data/interim/out_2019-2021_keywords_before_FuzzyWuzzy.csv')
unique_keywords=set(uni_kw['0'])
ded_kw=process.dedupe(unique_keywords, threshold = 90)

pd.Series(list(ded_kw)).to_csv('data/interim/out_dedupl_90.csv')