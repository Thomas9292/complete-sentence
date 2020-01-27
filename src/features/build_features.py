
class LanguageIndex():
    """
    Contains the translation from id to word and vice versa
    """
    def __init__(self, lang):
        self.lang = lang
        self.word2idx = {}
        self.idx2word = {}
        self.vocab = set()
        self.max_length = 0
        self.create_index()

    def create_index(self):
        for phrase in self.lang:
            phrase_list = phrase.split(' ')
            self.max_length = max(self.max_length, len(phrase_list))
            self.vocab.update(phrase_list)
        self.vocab = sorted(self.vocab)
        self.word2idx["<pad>"] = 0
        self.idx2word[0] = "<pad>"
        for i, word in enumerate(self.vocab):
            self.word2idx[word] = i + 1
            self.idx2word[i + 1] = word
