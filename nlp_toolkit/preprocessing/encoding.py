from collections import Counter
import numpy as np
from nltk import FreqDist
from nltk.corpus import stopwords
from nlp_toolkit.preprocessing.tokenizers import Tokenizers
from tensorflow.keras.preprocessing.text import Tokenizer
class Encoding:
    def __init__(self):
        self.stop_words = set(stopwords.words('english'))
        self.word_tokenizer = Tokenizers().tokenize_by_word
        self.tokenizer = Tokenizer(oov_token = 'OOV')
    
    def int_encoding_by_dic(self, tokens : list) -> list:
        """
        Encodes sentences into sequences of itergers based on word frequency.

        This method tokenizes sentences, removes common and shorts words, and then
        builds a vocabulary sorted by word frequency. Each word is assigned an interger rank,
        which is then userd to convert the senteces into lists of integers.
        An OOV index is used for rare words.

        Args:
            tokens (list) : A list of sentences(strings) to be encoded.
        
        Results:
            list : A list of integer-encoded words.
        """
        vocab = {}
        word_to_index = {}
        preprocessed_sentences = []
        encoded_sentences = []

        # Make words frequency dictionary
        for sentence in tokens:
            tokenized_sentece_to_word = self.word_tokenizer(sentence)
            result = []

            for word in tokenized_sentece_to_word:
                word = word.lower()
                if word not in self.stop_words:
                    if len(word) > 2:
                        result.append(word)
                        if word not in vocab:
                            vocab[word] = 0
                        vocab[word] += 1
            preprocessed_sentences.append(result)
        
        # Sort by frequency
        vocab_sorted = sorted(vocab.items(), key = lambda x:x[1], reverse = True)
        i = 0
        for (word, frequency) in vocab_sorted:
            if frequency > 1:
                i = i + 1
                word_to_index[word] = i
        
        # Add OOV index
        word_to_index['OOV'] = len(word_to_index) + 1
        
        # Encode all senteces in tokens
        for sentence in preprocessed_sentences:
            encoded_sentence = []
            for word in sentence:
                try:
                    encoded_sentence.append(word_to_index[word])
                except KeyError:
                    encoded_sentence.append(word_to_index['OOV'])
            encoded_sentences.append(encoded_sentence)
        
        return encoded_sentences
    
    def int_encoding_by_counter(self, tokens : list) -> dict:
        """
        Process integer encoding using Counter.

        Args:
            tokens (list) : A list of sentences(strings) to be encoded.
        
        Results:
            dict : A dictionary mapping each unique word to an integer index in the format {'word' : index}
        """
        # Make words frequency dictionary
        preprocessed_sentences = []

        for sentence in tokens:
            tokenized_sentece_to_word = self.word_tokenizer(sentence)
            result = []

            for word in tokenized_sentece_to_word:
                word = word.lower()
                if word not in self.stop_words:
                    if len(word) > 2:
                        result.append(word)
            preprocessed_sentences.append(result)
        
        all_words_list = sum(preprocessed_sentences, [])
        vocab = Counter(all_words_list)

        word_to_index = {}
        i = 0
        for (word, frequency) in vocab.items():
            i = i + 1
            word_to_index[word] = i
        
        return word_to_index
    
    def int_encoding_by_nltk(self, tokens : list) -> dict:
        """
        Process integer encoding using FreqDist by nltk.

        Args:
            tokens (list) : A list of sentences(strings) to be encoded.
        
        Results:
            dict : A dictionary mapping each unique word to an integer index in the format {'word' : index}

        """
        preprocessed_sentences = []

        for sentence in tokens:
            tokenized_sentece_to_word = self.word_tokenizer(sentence)
            result = []

            for word in tokenized_sentece_to_word:
                word = word.lower()
                if word not in self.stop_words:
                    if len(word) > 2:
                        result.append(word)
            preprocessed_sentences.append(result)
        
        vocab = FreqDist(np.hstack(preprocessed_sentences))

        word_to_index = {str(word) : index + 1 for index, word in enumerate(vocab)}

        return word_to_index
    
    def int_encoding_by_keras(self, tokens : list) -> dict:
        """
        Process integer encoding using Tokenizer by keras.

        Args:
            tokens (list) : A list of sentences(strings) to be encoded.
        
        Results:
            dict : A dictionary mapping each unique word to an integer index in the format {'word' : index}
        """

        preprocessed_sentences = []

        for sentence in tokens:
            tokenized_sentece_to_word = self.word_tokenizer(sentence)
            result = []

            for word in tokenized_sentece_to_word:
                word = word.lower()
                if word not in self.stop_words:
                    if len(word) > 2:
                        result.append(word)
            preprocessed_sentences.append(result)
        
        self.tokenizer.fit_on_texts(preprocessed_sentences)
        return self.tokenizer.word_index


        


    






        


        


        






