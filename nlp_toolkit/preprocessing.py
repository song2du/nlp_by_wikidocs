from konlpy.tag import Kkma, Okt
import kss
from nltk.tokenize import sent_tokenize, TreebankWordTokenizer, word_tokenize, WordPunctTokenizer
from nltk.tag import pos_tag
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
    """
    Tokenizes the input text into a list of Korean sentences.

    Agrs:
        text (str) : The Korean String to be tokenized.

    Returns:
        list : A list of tokenized sentences.
    """
    return kss.split_sentences(text)

def tag_by_penntreebank(text : str) -> list : 
    """
    POS(Part-of-Speech) tagging on the input text using the Penn Treebank tag set.
    
    Common Penn Treebank POS Tags include : 
    -   PRP : Personal Pronoun
    -   VBP : Verb, non-3rd person singular present
    -   RB : Adverb
    -   VBG : Present Participle
    -   IN : Preposition
    -   NNP : Proper Noun
    -   NNS : Noun, plural
    -   CC : Conjunction
    -   DT : Determiner

    Args:
        text (str) : The String to be POS tagged
    
    Returns:
        list : A list of (word, tag) tuples.
    """
    tokenized_sentence = word_tokenize(text)
    return pos_tag(tokenized_sentence)

def tag_by_Okt(text : str) -> list:
    """
    POS tagging on the input Korean text using Okt(Open Korea Text)

    Args:
        text (str) : Korean text to be POS tagged
    
    Returns:
        list : A list of (word, tag) tuples.
    """
    okt = Okt()
    return okt.pos(text)

def extract_morpheme_by_Okt(text : str) -> list:
    """
    Extracts morphemes from the input korean text using Okt(Open Korea Text)

    Args:
        text (str) : Korean text to be extracted morphemes.
    
    Returns:
        list : A list of extracted morphemes.
    """
    okt = Okt()
    return okt.morphs(text)

def extract_noun_by_Okt(text : str) -> list:
    """
    Extract nouns on the input korean text using Okt(Open Korea Text)

    Args:
        text (str) : Korean text to be extracted nouns.
    
    Returns:
        list : A list of extracted nouns.
    """
    okt = Okt()
    return okt.nouns(text)

def tag_by_Kkma(text : str) -> list:
    """
    POS tagging on the input Korean text using Kkma(꼬꼬마)

    Args:
        text (str) : Korean text to be POS tagged
    
    Returns:
        list : A list of (word, tag) tuples.
    """
    kkma = Kkma()
    return kkma.pos(text)

def extract_morpheme_by_Kkma(text : str) -> list:
    """
    Extracts morphemes from the input korean text using Kkma(꼬꼬마)

    Args:
        text (str) : Korean text to be extracted morphemes.
    
    Returns:
        list : A list of extracted morphemes.
    """
    kkma = Kkma()
    return kkma.morphs(text)

def extract_noun_by_Kkma(text : str) -> list:
    """
    Extract nouns on the input korean text using Kkma(꼬꼬마)

    Args:
        text (str) : Korean text to be extracted nouns.
    
    Returns:
        list : A list of extracted nouns.
    """
    kkma = Kkma()
    return kkma.nouns(text)
