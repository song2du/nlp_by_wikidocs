import kss
from nltk.tokenize import sent_tokenize, TreebankWordTokenizer, word_tokenize, WordPunctTokenizer
from tensorflow.keras.preprocessing.text import text_to_word_sequence

def tokenize_by_word(text : str) -> list:
   """
    Tokenizes the input text into a list of words using NLTK's word_tokenize.
    
    Args:
        text (str): The string to be tokenized.
    
    Returns:
        list: A list of tokenized words.
    
   """
   return word_tokenize(text)

def tokenize_by_word_punctuation(text : str) -> list:
    """
    Tokenizes the input text into the words and punctuation using NLTK's WordPunctTokenizer.
    
    Args:
        text (str) : The string to be tokenized.

    Returns:
        list : A list of tokenized words.
    """
    tokenizer = WordPunctTokenizer()
    return tokenizer.tokenize(text)

def tokenize_with_keras(text : str) -> list:
    """
    Tokenizes the input text into the words using Keras's text_to_word_sequence.
    This function converts all words to lowercase and removes punctuation during tokenization.
    
    Args:
        text (str) : The String to be tokenized.
    
    Returns:
        list : A list of tokenized words.
    """

    return text_to_word_sequence(text)

def tokenize_with_penntreebank(text : str) -> list:
    """
    Tokenizes the input text into the words using Penn Treebank Tokenization.

    This tokenizer:
    - Retains hyphenated words as single token.
    - Sperates clitics and contractions.
    
    Args:
        text (str) : The String to be tokenized.
    
    Returns:
        list : A list of tokenized words.
    """
    tokenizer = TreebankWordTokenizer()
    return tokenizer.tokenize(text)

def tokenize_by_sent(text : str) -> list:
    """
    Tokenizes the input text into a list of sentences.
    
    This function breaks down a larger text into individual sentences.

    Args :
        text (str) : The String to be tokenized.
    
    Returns :
        list : A list of tokenized sentences.
    """
    return sent_tokenize(text)

def tokenize_by_korean_sent(text : str) -> list:
    return kss.split_sentences(text)

