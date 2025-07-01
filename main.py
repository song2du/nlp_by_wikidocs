from nlp_toolkit.preprocessing import remove_short_word_with_regular_expression

if __name__ == '__main__':
    text = "Hi. My name is Yidu Song. I hope to be a billianare."
    print(remove_short_word_with_regular_expression(text))