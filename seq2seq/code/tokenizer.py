from konlpy.tag import Twitter
twitter = Twitter()


def tokenize_kr(text):
    return [i[0] for i in twitter.pos(text, norm=True, stem=True)]
