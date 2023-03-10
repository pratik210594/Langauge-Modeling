import math, sys

EPS = 0.0001

# A helper for reading the sentences in a corpus. This is a Python
# generator (note the use of 'yield' instead of 'return'). This allows
# us to encapsulate functionality for reading the file, easily iterate
# through the result (train_sents and test_sents below can be iterated
# through just like a list), and avoid reading the entire file into
# memory at once (not really a big deal with the small corpus sizes
# we're working with here, but it becomes very important when working
# with larger corpora).
def all_sents(infile):
    sent = []
    for token in open(infile, encoding='utf-8'):
        token = token.strip().lower()
        if token:
            sent.append(token)
        if token == '</s>':
            yield sent
            sent = []

# A uniform unigram language model. (Every type has the same
# probability. This is a very bad language model.)
class UniformLM:
    def __init__(self, infile):
        # vocab is a set representing all the types in the training
        # data (i.e., the vocabulary)
        self.vocab = set()
        # V is the number of types (i.e., the vocabulary size)
        self.V = 0
        train_sents = all_sents(infile)
        self.train(train_sents)

    def train(self, sentences):
        for sent in sentences:
            for token in sent:
                # Ignore start of sentence symbols when training the
                # language model. We don't estimate probabilities for
                # the start-of-sentence.
                if(token != '<s>'):
                    # Apply case folding
                    token = token.lower()
                    self.vocab.add(token)
                    
        # Add 1 to the vocabulary size to account for UNK
        self.V = len(self.vocab) + 1

    def logprob(self, w):
        # Compute the log prob for a word. Because this is a uniform
        # language model, w is ignored.
        #
        # We are computing 1/float(self.V), but in logspace
        # 1/float(self.V) = log(1) - log(self.V) = -log(self.V)
        return -math.log(self.V)
        
    def prob(self, w):
        # This is a useful helper for doing some sanity checks below.
        return math.exp(self.logprob(w))

    def check(self):
        # Check that the probability for each type is between 0 and 1,
        # and that the sum of the probability for all types is 1.
        all_types = list(self.vocab) + ['UNK']
        for w in all_types:
            assert 0 - EPS < self.prob(w) < 1 + EPS
        assert 1 - EPS < sum([self.prob(x) for x in all_types]) < 1 + EPS

if __name__ == '__main__':
    # The first argument will be '1', '2', or 'interp' to indicate a
    # 1-gram, 2-gram, or interpolated language model. (Ignored by the
    # starter code)
    n = sys.argv[1]

    # The second and third arguments are the training and testing file
    # names
    train_fname = sys.argv[2]
    test_fname = sys.argv[3]

    # Train a language model on the training data and check that the
    # probabilities are valid.
    lm = UniformLM(train_fname)
    # You can comment this check out to make it run faster once you've
    # got it working.
    lm.check()

    test_sents = all_sents(test_fname)
    for sent in test_sents:
        # We start at index 1 because the first item in a sentence is
        # always <s> and we don't predict a probability for this symbol.
        for i in range(1,len(sent)):
            token = sent[i]
            log_p = lm.logprob(token)
            #print(token, str(log_p))
            print(token.encode('ascii','ignore').decode('utf-8'), str(log_p))
