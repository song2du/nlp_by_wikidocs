from nltk.corpus import stopwords
class StopWords:
    def __init__(self):
        self.stop_words = stopwords

    # Start stopwords

    def remove_stopwords_by_nltk(self, tokens : list) -> list:
        """
        This process removes stopwords.
        Stopwords are words that appear frequently but do not carry significant meaning or information.
        It helps models to focus on more important features.

        Args:
            tokens (list) : The list of tokens to be processed for stopword removal.
        
        Returns:
            list : A new list of words after stopwords have been removed.
        """
        stop_words_set = set(self.stop_words)
        return [word for word in tokens if word not in stop_words_set]
    
    # Stop stopwords
