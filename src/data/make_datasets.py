import pandas as pd 
import json

def get_data(data_file, start_date, end_date):

    """
    Loads the data specified in data_file (json file)
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

    # creates a mask (list of True or False) for indices within the specified date range
    mask = (pd.to_datetime(df['lastModifiedDate']) > start_date) & (pd.to_datetime(df['lastModifiedDate']) <= end_date)
    
    # extracts subset from this dataset
    df_subset = df.loc[mask]

    # Outputs length of unique keywords before and after
    print('Extracting data DONE. Number of articles from', start_date, 'to', end_date, 'is', len(df_subset))

    return df_subset