import re

class TextClean :
    def __init__(self):
        pass

    def remove_short_word_with_regular_expression(self, text : str) -> str:
        """
        Removes words with a length of one or two characters from a string.

        This function uses the regular expression r'W*\b\w{1,2}\b':
        -   \W* : Matches any leading non-word characters (e.g., spaces, punctuation).
        -   \b : Asserts a word boundary to ensure whole words are matched.
        -   \w{1,2} : Matches any word character (alphanumeric & underscore) that occurs one or two times.
        -   \b : Asserts a closing word boundary

        Args:
            text (str): The input string to be cleaned.
        
        Returns:
            str: The text with short words removed.
        """
        
        shortword =  re.compile(r'\W*\b\w{1,2}\b')
        return shortword.sub('', text)

