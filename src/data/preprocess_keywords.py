"""
@author:  magda
"""

from itertools import chain

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


def clean_keywords(lst_lst_keywords):

    """
    Cleans a list of list of keywords
    1. Lower case
    2. Splits keywords that have not been properly split
    3. Remoes unicode and other unwanted symbols
    4. Removes long sentences

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
    lst_lst_keywords_clean = [list(map(lambda x: x.replace('keywords: ', ''), lst_kw)) for lst_kw in lst_lst_keywords_clean]

    # Remove sentences (keywords that have more than 6 spaces)
    n_spaces = 6
    lst_lst_keywords_clean = [[kw for kw in lst_kw if kw.count(' ')<n_spaces] for lst_kw in lst_lst_keywords_clean]

    # Outputs length of unique keywords before and after
    print('Cleaning DONE. Number of unique keywords went from', len(set(list(chain(*lst_lst_keywords)))), \
        'to', len(set(list(chain(*lst_lst_keywords_clean)))))

    return lst_lst_keywords_clean
