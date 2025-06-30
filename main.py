from nlp_toolkit.tokenizer import tokenize_by_korean_sent

if __name__ == '__main__':
    text = "딥 러닝 자연어 처리가 재미있기는 합니다. 그런데 문제는 영어보다 한국어로 할 때 너무 어렵습니다. 이제 해보면 알걸요?"
    print(tokenize_by_korean_sent(text))