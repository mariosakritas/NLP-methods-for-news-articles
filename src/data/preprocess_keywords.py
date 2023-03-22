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



