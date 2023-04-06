import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
import numpy as np
import pandas as pd


def get_vectors(lst_keywords, wv):
    '''
    Convert the list of keywords to vectors according to word vectors wv
    '''
    
    # Creating the vectorizer 
    vectorizer = CountVectorizer(stop_words='english')

    # Fit the model with our data (each keyword becomes a feature, some are split)
    X = vectorizer.fit_transform(lst_keywords)

    # Make an array and fills it in
    CountVectorizedData=pd.DataFrame(X.toarray(), columns=vectorizer.get_feature_names_out())

    # Words in the vocabulary (some keywords are split)
    WordsVocab=CountVectorizedData.columns

    W2Vec_Data_temp=pd.DataFrame()
    
    Sentence_1=[[wv[word] if word in wv.key_to_index.keys() else np.zeros(300) for word in WordsVocab[CountVectorizedData.iloc[i , :]>=1]] for i in range(CountVectorizedData.shape[0])]
    W2Vec_Data_temp=W2Vec_Data_temp.append(pd.DataFrame(Sentence_1))
    Sentence_2=[[wv[word.capitalize()] if word.capitalize() in wv.key_to_index.keys() and word not in wv.key_to_index.keys()  else np.zeros(300) for word in WordsVocab[CountVectorizedData.iloc[i , :]>=1]] for i in range(CountVectorizedData.shape[0])]
    W2Vec_Data_temp=W2Vec_Data_temp.append(pd.DataFrame(Sentence_2))
    test_sum_df = W2Vec_Data_temp.groupby(W2Vec_Data_temp.index).sum()
    test_sum_df[test_sum_df.applymap(lambda x: np.allclose(x, 0))] = np.nan
    
    W2Vec_Data_mean = test_sum_df.apply(lambda x: np.mean(x[x.notnull()]), axis=1)
    
    W2Vec_Data_df = pd.DataFrame(W2Vec_Data_mean)
    W2Vec_Data = W2Vec_Data_df[0].apply(pd.Series)

    W2Vec_Data.loc[W2Vec_Data.isna().any(axis=1),0:299] = 0

    return W2Vec_Data