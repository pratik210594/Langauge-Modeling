import math, sys

EPS = 0.0001
k = 0.002
lambdaa = 0.35
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
class UnigramLM:
    def __init__(self, infile):
        # vocab is a set representing all the types in the training
        # data (i.e., the vocabulary)
        self.vocab = set()
        # V is the number of types (i.e., the vocabulary size)
        self.V = 0
        # N is the number of tokens
        self.N = 0
        self.freq = {}
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
                    self.N+=1

                    if self.freq.get(token) == None:
                        self.freq[token]=1
                    else:
                        self.freq[token]+=1
                   
        # Add 1 to the vocabulary size to account for UNK
        self.V = len(self.vocab) + 1
       

    def logprob(self, w):
        # Compute the log prob for a word. Because this is a uniform
        # language model, w is ignored.
        #
        # We are computing 1/float(self.V), but in logspace
        # 1/float(self.V) = log(1) - log(self.V) = -log(self.V)
        # return -math.log(self.V)
        if self.freq.get(w) == None:
            numerator = 0 + 1
        else:
            numerator = self.freq[w] + 1
        denominator = self.N + self.V
        if n =='interp':
            return float(numerator)/float(denominator)
        else:
            return math.log(numerator) - math.log(denominator)
       
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

class BigramLM:
    def __init__(self, infile):
        # vocab is a set representing all the types in the training
        # data (i.e., the vocabulary)
        self.vocab = set()
        # V is the number of types (i.e., the vocabulary size)
        self.V = 0
        # N is the number of tokens
        self.N = 0
        self.freq = {}
        self.freq2 = {}
        train_sents = all_sents(infile)
        self.train(train_sents)
       

    def train(self, sentences):
        for sent in sentences:
            for token in range(len(sent)):
                # Ignore start of sentence symbols when training the
                # language model. We don't estimate probabilities for
                # the start-of-sentence.
                if(sent[token] != '<s>'):
                    # Apply case folding
                    sent[token] = sent[token].lower()
                    self.vocab.add(sent[token])
                    self.N+=1

                    if self.freq.get(sent[token]) == None:
                        self.freq[sent[token]]=1
                    else:
                        self.freq[sent[token]]+=1
                if(sent[token] != '</s>'):
                    key = sent[token] + ' ' + sent[token+1]
                    if(self.freq2.get(key) == None):
                        self.freq2[key]=1
                    else:
                        self.freq2[key]+=1

                   
        # Add 1 to the vocabulary size to account for UNK
        self.V = len(self.vocab) + 1
       

    def logprob(self, w):
        # Compute the log prob for a word. Because this is a uniform
        # language model, w is ignored.
        #
        # We are computing 1/float(self.V), but in logspace
        # 1/float(self.V) = log(1) - log(self.V) = -log(self.V)
        # return -math.log(self.V)
        num_token = w
        den_token = w.split()[0]
        
        if self.freq2.get(num_token) == None:
            numerator = 0 + k
        else:
            numerator = self.freq2[num_token] + k
        
        if self.freq.get(den_token) == None:
            denominator = 0 + k*self.V
        else:
            denominator = self.freq[den_token] + k*self.V
        if n =='interp':
            return float(numerator)/float(denominator)
        else:
            return math.log(numerator) - math.log(denominator)
       
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
    if n == '1':
        lm = UnigramLM(train_fname)
    elif n == '2':
        lmb = BigramLM(train_fname)
    elif n == 'interp':
        lm = UnigramLM(train_fname)
        lmb = BigramLM(train_fname)

    # You can comment this check out to make it run faster once you've
    # got it working.
    # lm.check()

    test_sents = all_sents(test_fname)
    for sent in test_sents:
        # We start at index 1 because the first item in a sentence is
        # always <s> and we don't predict a probability for this symbol.
        if n == '1':
            for i in range(1,len(sent)):
                token = sent[i]
                log_p = lm.logprob(token)
                print(token.encode('ascii','ignore').decode('utf-8'), str(log_p))
     
        
        elif n == '2':
            for i in range(0,len(sent)-1):
                token = sent[i]+' '+sent[i+1]
                log_p = lmb.logprob(token)
                print(token.replace(' ', ',').encode('ascii','ignore').decode('utf-8'), str(log_p))
        elif n =='interp':

            for i in range(0,len(sent)-1):
                if sent[i]!='</s>':
                    uni_token=sent[i+1]
                    bi_token=sent[i]+ ' ' +sent[i+1]
                #print(bi_token,uni_token)
                probab = ((1-lambdaa)*lmb.logprob(bi_token)) + (lambdaa*lm.logprob(uni_token))
                prob_log = math.log(probab)
                print(bi_token.replace(' ', ',').encode('ascii','ignore').decode('utf-8'), str(prob_log))