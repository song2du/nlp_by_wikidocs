from collections import Counter
import numpy as np
from nltk import FreqDist
from nltk.corpus import stopwords
from nlp_toolkit.preprocessing.tokenizers import Tokenizers
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.utils import to_categorical


class Encoding:
    def __init__(self):
        self.stop_words = set(stopwords.words('english'))
        self.word_tokenizer = Tokenizers().tokenize_by_word
        self.tokenizer = Tokenizer(oov_token = 'OOV')
    
    def sentence_to_word(self, tokens : list) -> list:
        """
        Tokenizes and preprocesses a list of sentences into words.

        Each sentence is tokenized into a list of words.
        The words are then converted to lowercase, and any stop words or words shorter than three characters are removed.

        Args:
            tokens(list): A list of sentences (strings) to process
        
        Returns:
            list : A list of lists, where each inner list contains the processed word tokens from a sentence.
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
        
        return preprocessed_sentences

    
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
    
    def pad_by_numpy (self, toekns : list) -> np.array:
        """
        Encodes and pads sentences into a NumPy array of uniform length.

        This method preprocesses raw text into tokenized sentences, 
        converts the tokens into integer sequences, and then pads each sequence with zeros to match the length of the longest sentence.

        Args:
            tokens (list): A list of sentences(string) to encode and pad
        
        Returns:
            np.array: A 2D NumPy array of integer-encoded sentences, with each row padded to a uniform length
        """
        preprocessed_sentences = self.sentence_to_word(tokens=toekns)
        self.tokenizer.fit_on_texts(preprocessed_sentences)
        encoded = self.tokenizer.texts_to_sequences(preprocessed_sentences)

        max_len = max(len(item) for item in encoded)

        for sentence in encoded:
            while len(sentence) < max_len:
                sentence.append(0)
        
        padded_np = np.array(encoded)

        return padded_np
    
    def pad_by_keras(self, tokens : list) -> np.array:
        """
        Encodes and pads sentences using the Keras pad_sequences utility.

        This method preprocesses raw text into tokenized sentences, 
        converts them to integer sequences, and then applies post-padding to ensure all sequences have a uniform length.

        Args:
            tokens (list): A list of sentences(string) to encode and pad
        
        Returns:
            np.array: A 2D NumPy array containig the integer-encoded and padded sequences.
        """
        preprocessed_sentences = self.sentence_to_word(tokens=tokens)
        self.tokenizer.fit_on_texts(preprocessed_sentences)
        encoded = self.tokenizer.texts_to_sequences(preprocessed_sentences)

        padded = pad_sequences(encoded, padding='post')

        return padded
    
    def one_hot_encoding(self, tokens : list) -> list:
        """
        Performs one-hot encoding on a list of tokens.

        This function creates a unique vocabulary from the input tokens and then generates a corresponding one-hot vector for each token in the list.

        Args:
            tokens (list): A list of tokens to be one-hot encoded

        Results:
            list: A list of one-hot encoded vectors, where each vector corresponds to a token in the original list.
        """
        result = []
        word_to_index = {word : index for index, word in enumerate(tokens)}
        
        for word in word_to_index:
            one_hot_vector = [0] * (len(word_to_index))
            index = word_to_index[word]
            one_hot_vector[index] = 1
            result.append(one_hot_vector)
        
        return result
    
    def one_hot_encoding_by_keras(self, text : str) -> list:
        """
        Performs one-hot encoding using Keras to_categorical utility.

        Args:
            text (str): A String to be encoded.

        Results:
            list : A list of one-hot encoded vectors where each vector corresponds to a token in the original list.
        """
        self.tokenizer.fit_on_texts([text])
        encoded = self.tokenizer.texts_to_sequences([text])[0]
        one_hot = to_categorical(encoded)

        return one_hot








    
    







        


    






        


        


        






