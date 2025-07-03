from nlp_toolkit.preprocessing.tokenizers import Tokenizers
from nlp_toolkit.preprocessing.encoding import Encoding 
if __name__ == '__main__':
    text = "나는 자연어 처리를 배운다"
    result = Encoding().one_hot_encoding_by_keras(text)
    print(result)

