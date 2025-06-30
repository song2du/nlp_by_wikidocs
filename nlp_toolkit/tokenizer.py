from nltk.tokenize import word_tokenize
from nltk.tokenize import WordPunctTokenizer
from tensorflow.keras.preprocessing.text import text_to_word_sequence


def tokenize_by_word(text):
   """
    단어 토큰화 함수
    입력 : 토큰화할 문자열
    출력 : 토큰화된 단어 리스트
   """
   return word_tokenize(text)

def tokenize_by_word_punctuation(text):
    """
    단어 토큰화 함수, 구두점을 별도로 분리
    입력 : 토큰화할 문자열
    출력 : 토큰화된 단어 리스트
    """
    return WordPunctTokenizer(text)

def tokenize_with_keras(text):
    """
    케라스를 이용한 단어 토큰화 함수, 모든 단어 소문자 변환, 구두점 제거
    입력 : 토큰화할 문자열
    출력 : 토큰화된 단어 리스트
    """

    return text_to_word_sequence(text)