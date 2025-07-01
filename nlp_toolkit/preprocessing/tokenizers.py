from konlpy.tag import Kkma, Okt
import kss
from nltk.tokenize import sent_tokenize, TreebankWordTokenizer, word_tokenize, WordPunctTokenizer
from nltk.tag import pos_tag
from tensorflow.keras.preprocessing.text import text_to_word_sequence

class Tokenizers:
    def __init__(self):
        self.okt_tokenizer = Okt()
        self.kkma_tokenizer = Kkma()
        self.treebank_tokenizer = TreebankWordTokenizer()
        self.wordpunct_tokenizer = WordPunctTokenizer()
        self.word_tokenizer = word_tokenize
        self.text_to_seq = text_to_word_sequence
        

    ## Start Tokenization

    def tokenize_by_word(self, text : str) -> list:
        """
            Tokenizes the input text into a list of words using NLTK's word_tokenize.
            
            Args:
                text (str): The string to be tokenized.
            
            Returns:
                list: A list of tokenized words.
            
        """
        return self.word_tokenize(text)

    def tokenize_by_word_punctuation(self, text : str) -> list:
        """
        Tokenizes the input text into the words and punctuation using NLTK's WordPunctTokenizer.
        
        Args:
            text (str) : The string to be tokenized.

        Returns:
            list : A list of tokenized words.
        """
        return self.wordpunct_tokenizer.tokenize(text)

    def tokenize_with_keras(self, text : str) -> list:
        """
        Tokenizes the input text into the words using Keras's text_to_word_sequence.
        This function converts all words to lowercase and removes punctuation during tokenization.
        
        Args:
            text (str) : The String to be tokenized.
        
        Returns:
            list : A list of tokenized words.
        """

        return self.text_to_seq(text)

    def tokenize_with_penntreebank(self, text : str) -> list:
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
        return self.treebank_tokenizer.tokenize(text)

    def tokenize_by_sent(self, text : str) -> list:
        """
        Tokenizes the input text into a list of sentences.
        
        This function breaks down a larger text into individual sentences.

        Args :
            text (str) : The String to be tokenized.
        
        Returns :
            list : A list of tokenized sentences.
        """
        return sent_tokenize(text)

    def tokenize_by_korean_sent(self, text : str) -> list:
        """
        Tokenizes the input text into a list of Korean sentences.

        Agrs:
            text (str) : The Korean String to be tokenized.

        Returns:
            list : A list of tokenized sentences.
        """
        return kss.split_sentences(text)

    def tag_by_penntreebank(self, text : str) -> list : 
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
        tokenized_sentence = self.word_tokenizer(text)
        return pos_tag(tokenized_sentence)

    def tag_by_Okt(self, text : str) -> list:
        """
        POS tagging on the input Korean text using Okt(Open Korea Text)

        Args:
            text (str) : Korean text to be POS tagged
        
        Returns:
            list : A list of (word, tag) tuples.
        """
        return self.okt_tokenizer.pos(text)

    def extract_morpheme_by_Okt(self, text : str) -> list:
        """
        Extracts morphemes from the input korean text using Okt(Open Korea Text)

        Args:
            text (str) : Korean text to be extracted morphemes.
        
        Returns:
            list : A list of extracted morphemes.
        """
        return self.okt_tokenizer.morphs(text)

    def extract_noun_by_Okt(self, text : str) -> list:
        """
        Extract nouns on the input korean text using Okt(Open Korea Text)

        Args:
            text (str) : Korean text to be extracted nouns.
        
        Returns:
            list : A list of extracted nouns.
        """
        return self.okt_tokenizer.nouns(text)

    def tag_by_Kkma(self, text : str) -> list:
        """
        POS tagging on the input Korean text using Kkma(꼬꼬마)

        Args:
            text (str) : Korean text to be POS tagged
        
        Returns:
            list : A list of (word, tag) tuples.
        """
        return self.kkma_tokenizer.pos(text)

    def extract_morpheme_by_Kkma(self, text : str) -> list:
        """
        Extracts morphemes from the input korean text using Kkma(꼬꼬마)

        Args:
            text (str) : Korean text to be extracted morphemes.
        
        Returns:
            list : A list of extracted morphemes.
        """
        return self.kkma_tokenizer.morphs(text)

    def extract_noun_by_Kkma(self, text : str) -> list:
        """
        Extract nouns on the input korean text using Kkma(꼬꼬마)

        Args:
            text (str) : Korean text to be extracted nouns.
        
        Returns:
            list : A list of extracted nouns.
        """
        return self.kkma_tokenizer.nouns(text)