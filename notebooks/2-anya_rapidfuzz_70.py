from rapidfuzz import process as pr
import pandas as pd

uni_kw=pd.read_csv('../data/interim/out_2019-2021_keywords_before_FuzzyWuzzy.csv') 
ratio_array=pr.cdist(list(uni_kw['0']),list(uni_kw['0']),score_cutoff = 70)