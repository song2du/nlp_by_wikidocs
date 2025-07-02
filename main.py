from nlp_toolkit.preprocessing.tokenizers import Tokenizers
from nlp_toolkit.preprocessing.encoding import Encoding 
if __name__ == '__main__':
    text = "A barber is a person. a barber is good person. a barber is huge person. he Knew A Secret! The Secret He Kept is huge secret. Huge secret. His barber kept his word. a barber kept his word. His barber kept his secret. But keeping and keeping such a huge secret to himself was driving the barber crazy. the barber went up a huge mountain."
    text = Tokenizers().tokenize_by_sent(text)
    result = Encoding().int_encoding_by_keras(text)
    print(result)

