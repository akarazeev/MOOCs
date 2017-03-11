# DeepNLP [[oxford-cs-deepnlp-2017]](https://github.com/oxford-cs-deepnlp-2017/lectures)

**TODO:**
- [x] Lecture 1a
- [x] Lecture 1b
- [x] Lecture 2a
- [x] Lecture 2b
- [x] Lecture 3
- [ ] Make quiz

#### Первое занятие [[link]](http://info.deephack.me/?p=572)
Ссылка на лекции - [github](https://github.com/oxford-cs-deepnlp-2017/lectures)

В качестве основного учебника предлагается [Deep Learning Book](http://deeplearningbook.org)

Видео для первого занятия - [link](http://media.podcasts.ox.ac.uk/comlab/deep_learning_NLP/2017-01_deep_NLP_1b_friends.mp4)

#### Второе занятие [[link]](http://info.deephack.me/?p=577)


---

##### Lecture 1a - Introduction

Points:
- I saw her duck.
- It was the intro by the way :)

##### Lecture 1b - Deep Neural Networks Are Our Friends

Points:
- Cost functions
- Optimization
- Derivative of cost functions
- Nonlinear neural models
- Multilayer perceptrons - universal approximators
- Regularization

##### Lecture 2a - Word Level Semantics

Points:
- Naive representation - one hot vectors. Problem: sparse and orthogonal
- Idea: produce dense vector representation
- "You shall know a word by the company he keeps." (c)
- Count-based methods:
    - **basic vocab C** of context words
    - **word window** size w
    - **count basic vocab words** - w words to the left or right from a target word in the corpus
    - form a **vector representation** of the target word based on these counts
- Neural embedding models:
    - count based vectors produce **embedding matrix**
    - CBoW
    - negative sampling
    - Skip-gram
    - Pointwise Mutual Information (PMI) matrix factorization == PMI matrix decomposition
- Classify sentences/documents
    - **bag of vectors**
    - ...

##### Lecture 2b - Overview of the Practicals

Points:
- TED data
- Toolkits
- Computation graphs:
    - Static - TensorFlow, Theano
    - Dynamic - DyNet, pytorch

##### Lecture 3 - Language Modelling and RNNs Part 1

Points:
- p(he likes apples) > p(apples likes he) - Translation
- p(he likes apples) > p(he licks apples) - Speech Recognition
- **Chain rule** for joint distribution: p(w1,w2,...) = p(w1) x p(w2|w1) x p(w3|w1,w2) x ...
- High prob to real utterance. This can be measured with cross entropy
- Time series prediction - train on the past and test on the future
- WikiText datasets are a better option
- Approaches to parametrize language models:
    - **count based n-gram** - approximate the history of previously observed n words
    - **neural n-gram** - embed fixed n-gram history in a _continuous_ space and thus better capture correlations between histories
    - **RNN** - fixed n-gram history -> compress the entire history in a fixed length vector, enabling long range correlations to be captured
- **Count based N-Gram Models**
    - Markov assumption:
        - only prev history matters
        - limited memory: only last k-1 words are included in history (older words are less relevant)
        - k-th order Markov model
    - p(w1,w2,...,wn) = p(w1) x p(w2|w1) x p(w3|w1,w2) x ... ~ p(w1) x p(w2|w1) x p(w3|w2) x ... x p(wn|w(n-1))
    - for 2-gram (bigram) - w_{i-1} is **history** for target word
    - max likelihood estimation: p(w3|w1,w2) = count(w1,w2,w3) / count(w1,w2)
    - Back-Off - to smooth our language model
        - trigram -> bigram
        - **linear interpolation**: p(wn|w(n-2),w(n-1)) = lambda3 x p(wn|w(n-2),w(n-1)) + lambda2 x p(wn|w(n-1)) + lambda1 x p(wn), where lambda3 + lambda2 + lambda1 = 1
- **Neural N-Gram Language Models**
    - wn|w(n-1),w(n-2) ~ pn (using softmax)
    - Gradients ...
- **RNN LMs**
    - Feed Forward: h=g(Vx + c), y=Wh + b
    - Recurrent Network: hn=g(V[xn;h(n-1)] + c), yn=Whn + b
    - **Back Propagation Through Time** Algorithm (BPTT)
    - **Truncated BPTT** (TBPTT) - much simpler gradient formula
- Bias vs Variance in LM Approximations
    - n-gram - only last n words; biased, low variance
    - RNN - unbounded history into a fixed sized vector; decrease the bias, at a cost to variance
