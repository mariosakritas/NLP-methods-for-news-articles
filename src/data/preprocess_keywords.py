"""
@author:  magda
"""

from itertools import chain
import collections
from rapidfuzz import process as pr
import numpy as np
import pandas as pd

def get_items_with_substring(lst_lst_keywords, substring):
    
    """
    Takes in a list of list of keywords (lst_lst_keywords) and a sequence of characters (substring) to search for in the list of list of keywords
    Returns the indices of the list of list of keywords, the lists of keywords, and the keywords containing the sequence of characters 

    """
    
    # Indices of rows in the list of list of keywords which have this substring 
    indices_substring = [i for i, lst_kw in enumerate(lst_lst_keywords) if any(list(map(lambda x: substring in x, lst_kw)))]
    
    # Lists of keywords which have this substring 
    lst_keywords_substring = [lst_lst_keywords[i] for i in indices_substring]
    
    # Flattens the list and gets (unique) keywords which have this substring 
    keywords_substring = [kw for kw in list(set(chain(*lst_keywords_substring))) if substring in kw]
    
    return indices_substring, lst_keywords_substring, keywords_substring


def make_cleaned_keywords_df(df_subset, start_date, end_date):
    '''
    Function to get a clean dataframe.
    1. Extracts data within the specified range.
    2. Cleans the keyword column
    3. Cleans the category column
    4. Saves the dataframe as a json file

    Columns in output dataframe
    cleanFocusParentCategory = Only the parent category (from src/preprocess_keywords.py function: clean_categories)
    cleanFocusCategory = all categories
    keywordStrings = raw keywords
    keywordStringsCleanAfterFuzz = Cleaned keywords (from src/preprocess_keywords.py function: make_cleaned_keywords_df)
    lastModifiedDate = raw date
    Date = just the date (in datetime format)

    '''

    # Cleans keywords - ! Runs for 6min on 2019-2022 ! 
    lst_lst_keywords = list(df_subset.keywordStrings)
    lst_lst_keywords_clean = basic_clean_keywords(lst_lst_keywords) # 1st (raw) cleaning
    lst_lst_keywords_clean_replaced = standardize_keywords(lst_lst_keywords_clean) # 2nd cleaning - standardizes keywords (runs rapid fuzz and replaces) 

    # Cleans categories
    df_subset = clean_categories(df_subset)

    # Make a new dataframe
    df_subset_new = df_subset[['id', 'lastModifiedDate', 'Date', 'keywordStrings', 'cleanFocusParentCategory', 'cleanFocusCategory', 'teaser']].copy()
    df_subset_new['keywordStringsCleanAfterFuzz'] = lst_lst_keywords_clean_replaced

    # Storing the data in JSON format
    filepath = '../data/interim/clean_keywords_' + start_date + '_' + end_date + '.json'
    df_subset_new.to_json(filepath, orient = 'split', compression = 'infer', index = 'true')
    
    print('Finished. Data stored in', filepath)


def basic_clean_keywords(lst_lst_keywords):

    """
    Cleans a list of list of keywords
    1. Lower case
    2. Splits keywords that have not been properly split
    3. Removes unicode and other unwanted symbols
    4. Removes leading and trailing whitespaces
    5. Removes long sentences

    """

    lst_lst_keywords_clean = lst_lst_keywords

    # Lower case
    lst_lst_keywords_clean = [list(map(str.casefold, x)) for x in lst_lst_keywords_clean]

    # Split
    lst_lst_keywords_clean = [list(chain(*[kw.split(', ') for kw in lst_kw])) for lst_kw in lst_lst_keywords_clean]
    lst_lst_keywords_clean = [list(chain(*[kw.split(' - ') for kw in lst_kw])) for lst_kw in lst_lst_keywords_clean]

    # Replace unicode and double spaces by a space
    lst_lst_keywords_clean = [list(map(lambda x: x.replace('\xa0', ' '), lst_kw)) for lst_kw in lst_lst_keywords_clean]
    lst_lst_keywords_clean = [list(map(lambda x: x.replace('  ', ' '), lst_kw)) for lst_kw in lst_lst_keywords_clean]

    # Replace unwanted characters
    lst_lst_keywords_clean = [list(map(lambda x: ''.join(filter(str.isprintable, x)), lst_kw)) for lst_kw in lst_lst_keywords_clean]
    lst_lst_keywords_clean = [list(map(lambda x: x.replace('.', ''), lst_kw)) for lst_kw in lst_lst_keywords_clean]
    lst_lst_keywords_clean = [list(map(lambda x: x.replace('" ', ''), lst_kw)) for lst_kw in lst_lst_keywords_clean]
    lst_lst_keywords_clean = [list(map(lambda x: x.replace('"', ''), lst_kw)) for lst_kw in lst_lst_keywords_clean]
    lst_lst_keywords_clean = [list(map(lambda x: x.replace("'", ''), lst_kw)) for lst_kw in lst_lst_keywords_clean]
    lst_lst_keywords_clean = [list(map(lambda x: x.replace('keywords: ', ''), lst_kw)) for lst_kw in lst_lst_keywords_clean]

    # Remove leading and trailing whitespaces
    lst_lst_keywords_clean = [list(map(lambda x: x.strip(), lst_kw)) for lst_kw in lst_lst_keywords_clean]

    # Remove sentences (keywords that have more than 6 spaces)
    n_spaces = 6
    lst_lst_keywords_clean = [[kw for kw in lst_kw if kw.count(' ')<n_spaces] for lst_kw in lst_lst_keywords_clean]

    # Outputs length of unique keywords before and after
    print('Cleaning step 1 out of 2 DONE. Number of unique keywords went from', len(set(list(chain(*lst_lst_keywords)))), \
        'to', len(set(list(chain(*lst_lst_keywords_clean)))))

    return lst_lst_keywords_clean



def standardize_keywords(lst_lst_keywords_clean):

    keywords_flat = list(chain(*lst_lst_keywords_clean)) # Flatten list
    keywords_freq = collections.Counter(keywords_flat)

    # extract unique ones and remove the empty entry
    unique_keywords = list(set(keywords_flat))
    if '' in unique_keywords:
        unique_keywords.remove('')

    ### Run rapid fuzz
    ratio_array= pr.cdist(unique_keywords, unique_keywords, score_cutoff = 90)

    ### Extract list of keywords which have similarity (based on rapidfuzzz cutoff)

    # Convert to dataframe
    df_array = pd.DataFrame(ratio_array, columns = unique_keywords)

    # Save indices of rows with more than 1 non-zero value
    nb_non_zero = np.count_nonzero(np.asarray(ratio_array), axis=1)
    indices_correlating_rows = [i for i, el in enumerate(list(nb_non_zero)) if el>1]

    # Make a list of similar keywords
    all_similar_kw = []
    for i in indices_correlating_rows:
        similar_words = [keyword for val, keyword in zip(list(df_array.iloc[i]), unique_keywords) if val!=0]
        
        # Only adds it if it's not there already
        if similar_words not in all_similar_kw:
            all_similar_kw.append(similar_words)

    # Split in the ones which have equal word number, and the ones which don't, becase they need different processing
    similar_kws_same_word_nb = [sim_kws for sim_kws in all_similar_kw if len(set([kw.count(' ') for kw in sim_kws]))==1]
    similar_kws_diff_word_nb = [sim_kws for sim_kws in all_similar_kw if len(set([kw.count(' ') for kw in sim_kws]))!=1]

    # TODO: Replace: in similar_kws_diff_word_nb

    # Replace: in similar_kws_same_word_nb
    right_kw = [sim_kws[np.argmax([keywords_freq[word] for word in sim_kws])] for sim_kws in similar_kws_same_word_nb]

    keywords_flat = list(chain.from_iterable(lst_lst_keywords_clean))

    replacement_only = [[right_kw[i] for i, j in enumerate(similar_kws_same_word_nb) if word in j] for word in keywords_flat]

    keywords_flat_post = [replacement_only[i][0] if replacement_only[i] != [] else keywords_flat[i] for i in range(len(keywords_flat))]

    def gen_list_of_lists(original_list, new_structure):
        assert len(original_list) == sum(new_structure), \
        "The number of elements in the original list and desired structure don't match"
        list_of_lists = [[original_list[i + sum(new_structure[:j])] for i in range(new_structure[j])] \
                        for j in range(len(new_structure))]
        return list_of_lists

    lst_lst_keywords_replaced = gen_list_of_lists(keywords_flat_post, [len(x) for x in lst_lst_keywords_clean])


    # Outputs length of unique keywords before and after
    print('Cleaning step 2 out of 2 DONE. Number of unique keywords went from', len(set(list(chain(*lst_lst_keywords_clean)))), \
        'to', len(set(list(chain(*lst_lst_keywords_replaced)))))
    
    return lst_lst_keywords_replaced


def clean_categories(df):
    ''' 
    Cleans the category column of data frame df
    1. Gets rid of the dictionary format
    2. Extracts all the main (primary) categories

    '''

    # Makes a new column to get rid of the dictionary format
    df['cleanFocusCategory'] = df['thematicFocusCategory'].apply(lambda x: x['name'] if x is not None else x)

    # Convert all secondary categories into primary categories
    children_dict = {'Architecture':'Culture', 'Design':'Culture', 'Film':'Culture', 'Arts':'Culture', 
                    'Literature':'Culture', 'Music':'Culture', 'Dance':'Culture', 'Theater':'Culture',
                    'Climate':'Nature and Environment',
                    'Conflicts':'Politics', 'Terrorism':'Politics', 
                    'Corruption':'Law and Justice', 'Crime':'Law and Justice', 'Rule of Law':'Law and Justice',
                        'Press Freedom':'Law and Justice', 
                    'Diversity':'Human Rights', 'Freedom of Speech':'Human Rights', 'Equality':'Human Rights', 
                    'Soccer': 'Sports',
                        'Trade':'Business', 'Globalization':'Business', 'Food Security':'Business'
    }

    secondary_cts = [val for val in children_dict.keys()]

    # Replaces
    df['cleanFocusParentCategory'] = df['cleanFocusCategory'].apply(lambda x: children_dict[x] if x in secondary_cts else x)

    return df