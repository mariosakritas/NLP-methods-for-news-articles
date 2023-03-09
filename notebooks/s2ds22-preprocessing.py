import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import SnowballStemmer
import re
from unidecode import unidecode
from bs4 import BeautifulSoup


regxcache={
# precompile regular expressions for faster preprocessing
"<.*?>" : re.compile("<.*?>"),
"&.{1,9};" : re.compile("&.{1,9};"),
"[^a-z]+" : re.compile("[^a-z]+"),
"\s[a-z]\s+" : re.compile("\s[a-z]\s+"),
"\s+" : re.compile("\s+"),
}


def preprocess_CMS_article(doc):
    """
    Normalize articles.

    1. lowercase
    2. only single whitespace
    3. unicode to ascii
    4. delete punctuation and numbers
    5. delete html tags <...>
    6. delete html character (like &nbsp;)
    7. delete stray single characters
    """
    # Lowercase
    doc = doc.lower()
    # expand negations
    doc = doc.replace("n't"," not")
    # Remove leading/trailing whitespace
    doc = doc.strip()
    # Convert Unicode into ASCII
    doc = unidecode(doc)
    # Remove HTML tags:
    doc = regxcache["<.*?>"].sub(" ", doc)
    # remove &nbsp; and other HTML codes up to 9 characters long
    doc = regxcache["&.{1,9};"].sub(" ", doc)
    # Remove punctuation and Numbers
    doc = regxcache["[^a-z]+"].sub( " ", doc)
    # Remove stray single characters
    doc = regxcache["\s[a-z]\s+"].sub( " ", doc)
    # Remove extra whitespace
    doc = regxcache["\s+"].sub( " ", doc)
    return doc


def remove_HTML(html_text):
    """
    Return text without HTML tags ("<...>") and inline code ("&nbsp;").

    Adapted from from Julia Landmann's notebook.
    """
    try:
    # parsing the html file
        htmlParse = BeautifulSoup(html_text, 'html.parser')
        text = htmlParse.get_text()

    except ValueError as e:
    # will get exception for missing texts
        if html_text=="": pass
        else: raise e

    return text


def remove_HTML5_warning(x):
    """
    Remove warning messages from web scraped documents.

    Adapted from Julia Landmann's notebook
    """
    return str(x).replace('To view this video please enable JavaScript, and consider upgrading to a web browser that supports HTML5 video',".") \
            .replace('To play this audio please enable JavaScript, and consider upgrading to a web browser that supports HTML5 audio', '.')
                #.replace("\'s", "'s")


def rename_columns(df):
    """
    Normalize column names to DW standards
    """
    # TODO
    return df


def stem(textcol, language="en"):
    """
    Runs a Snowball Stemmer over a column of text data
    """
    # TODO
    print("Not implemented yet.")
    return textcol


if __name__=="__main__":
    # If run directly, download background data
    nltk.download('stopwords')
    nltk.download('punkt')
