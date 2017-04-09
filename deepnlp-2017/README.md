# DeepNLP [[oxford-cs-deepnlp-2017]](https://github.com/oxford-cs-deepnlp-2017/lectures)

**TODO:**
- [x] Lecture 1a
- [x] Lecture 1b
- [x] Lecture 2a
- [x] Lecture 2b
- [x] Lecture 3
- [x] Make Quiz
- [x] Lecture 4
- [x] Make Quiz
- [x] Lecture 5
- [x] Make Quiz
- [x] Lecture 7
- [x] Lecture 8
- [ ] Lecture 11
- [ ] Lecture 12

#### Первое занятие [[link]](http://info.deephack.me/?p=572)
Ссылка на лекции - [github](https://github.com/oxford-cs-deepnlp-2017/lectures)

В качестве основного учебника предлагается [Deep Learning Book](http://deeplearningbook.org)

Видео для первого занятия - [link](http://media.podcasts.ox.ac.uk/comlab/deep_learning_NLP/2017-01_deep_NLP_1b_friends.mp4)

#### Второе занятие [[link]](http://info.deephack.me/?p=577)

#### Третье занятие [[link]](http://info.deephack.me/?p=580)

#### Четвертое занятие [[link]](http://info.deephack.me/?p=591)

#### Пятое занятие [[link]](http://info.deephack.me/?p=595)

#### Шестое занятие [[link]](http://info.deephack.me/?p=598)

---

##### Lecture 1a - Introduction

- I saw her duck.
- It was the intro by the way :)

##### Lecture 1b - Deep Neural Networks Are Our Friends

- Cost functions
- Optimization
- Derivative of cost functions
- Nonlinear neural models
- Multilayer perceptrons - universal approximators
- Regularization

##### Lecture 2a - Word Level Semantics

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

- TED data
- Toolkits
- Computation graphs:
    - Static - TensorFlow, Theano
    - Dynamic - DyNet, pytorch

##### Lecture 3 - Language Modelling and RNNs Part 1

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

##### Lecture 4 - Language Modelling and RNNs Part 2

- LMs aim to predict next word w_t by the history of observed text (w_1, ..., w_{t-1}):
    - count based n-gram LMs approximate the history with the previous n words
    - neural n-gram LMs embed the fixed n-gram history in a continues space and thus capture correlations between histories
    - RNN LMs use fixed n-gram history and compress the entire history in a foxed length vector, enabling long range correlations to be captured

- Exploding and vanishing gradients
    - partial derivatives ...
    - if the largest eigenvalue of V_h is:
        - 1, gradient will propagate
        - \> 1, explodes
        - < 1, vanishes
    - if it explodes - we can't train our model because of infinity gradient
    - if it vanishes - we are not going to learn long-range dependencies because of zero gradient
- Changing the network architecture
- Long short-term memory (LSTM):
    - key modification of LSTM is replacing multiplication with sum
    - there is a switch
    - forget gate
- Gated Recurrent Network (GRU)
- Deep RNN LMs:
    - how the hidden layer size affects a model size and computation complexity - quadratic
    - skip connection - makes learning much quicker
    - another option is increase depth in time dimension
    - Problem: Large Vocabularies. Solutions:
        - **Short-lists:** only most frequent words and traditional n-gram LM for the rest, but it nullifies the ability to generalize rare events.
        - **Batch local short-lists:**  approximate the full partition function for data instances from a segment of the data with a subset of the vocabulary chosen for that segment.
        - **Approximate the gradient/change the objective:** softmax(x) -> exp(x) * exp(c), where c is some constant (in practice c=0 is effective)
        - **Noise Contrastive Estimation (NCE):** p(D=1 | \hat{p_n}) = \frac{\hat{p\_n}}{\hat{p\_n} + kp_{noise}(w\_n)}, where p_{noise} is unigram distribution for example
        - **Importance Sampling**
        - **Factorise the output vocabulary:** introduce word classes, p(w\_n | \hat{p\_n^{class}}, \hat{p_n^{word}}) = p(class(w_n)). With balanced classes we get sqrt(V) speedup
        - by extending the factorization to a binary tree we can get a log(V) speedup (based Huffman coding is a poor choice): p(w_n | h_n) = \mult_i p(d_i | r_i, h_n) (the better choice is n-ary factorization tree)
    - Sub-Word Level LM:
        - change the input granularity and model text at the morpheme or character level
        - pros: much smaller softmax and no unknown words
        - cons: longer sequences and longer dependencies
        - allows to capture subword structure and morphology
- Regularization: **Dropout**
    - large rnn often overfit their training data my memorizing the sequences they learn
    - how apply dropout - do not zero all hidden units but apply it to non-recurrent connections
    - Bayesian Dropout

#### Lecture 5 - Text Classification

- Classification Tasks
    - pos/neg
    - topic
    - hashtags for twitter
- binary/multi-class/multi-label/clustering classification
- Classification methods:
    - By hand
    - Rule-based
    - Statistical
- Text label - d, Class - c:
    - how to represent d?
    - how to calculate P(c|d)?
- Possible Representations:
    - Bag of words
    - Hand-crafted features (makes use of NLP pipeline - ???)
    - Learned feature representation
- Generative vs Discriminative Models:
    - **Generative** (joint) models: P(c, d)
        - distribution of individual classes
        - n-gram, HMM, IBM translation models, Naive Bayes
        - Naive Bayes classifier
            - the best class if the maximum a posteriori (MAP) class (Laplace smoothing)
                - sentence/document structure not taken into account :(
                - smoothing
    - **Discriminative** (conditional) models: P(c|d)
        - learn boundaries between classes
        - log regression, max entropy models, conditional random fields, svm
- Features Representations
- Logistic Regression
- Due to the Softmax function we not only construct a classifier but learn **probability distributions** over classifications
- Representing Text with RNN
- sigmoid function in text classification task with an RNN
- Loss function for an RNN Classifier
    - Multilayer Perceptron
    - the cross-entropy loss is designed to deal with errors on probabilities
- Dual Objective RNN
- Bi-Directional RNN
- Non-Sequential NNs:
    - Recursive NNs
        - Autoencoder Signals
    - CNNs
        - Convolutional layer with multiple filters
        - Max-Pooling Layer
        - R^{M x N x K} where M - number of input words, N - size of the input embeddings and K - number of feature maps

#### Lecture 7 - Conditional Language Models

- Unconditional LM assigns probs to seqs of words w=(w1, ..., wl)
- Task: modeling prob of the next word by the history of prevs words
- Conditional LM assigns probs to seqs of words by the conditioning context x
- to train we need paired samples {(x, w)}
- w^\* = arg max p(w | x). Approximate it using a **beam search**.
- Evaluating conditional LMs - using cross-entropy or perplexity. Or easier and interpretable - Task-specific evaluation: **BLEU**, **METEOR**, **WER**, **ROUGE** - metric L(w^\*, w_{ref})
- Encoder-Decoder model
    - how should we define c=embed(x)?
    - Convolutional sentence model (CSM)
    - Good - learn interactions among local features in context, long dependencies can be learnt. Bad - sentences have different lengths
    - RNN Decoder - p(tom|s,&lt;s&gt;) x p(likes|s,&lt;s&gt;,tom) x p(beer|s,&lt;s&gt;,tom,likes) x p(<\s>|s,&lt;s&gt;,tom,likes,beer)
    - Sutskever et al. (2014) - LSTM Encoder-Decoder
    - Good - RNNs deal with seqs of various lengths, LSTMs can prop grads a long distance, simple architecture. Bad - hidden state has to remember a lot of information!
    - Trick#1 - read the input seq "backwards": +4 BLEU
    - Trick#2 - use an ensemble of J independently trained models: ensemble of 2 models gives +3 BLEU, ensemble of 5 models gives +4.5 BLEU
    - We want to fine the most probable (MAP) output given the input: w^\* = arg max p(w|x). It's a hard (undecidable) problem for RNNs, approximate it with a **greedy search**: w^\*\_1 = arg max p(w_1 | x), w^\*\_2 = arg max p(w_2 | x, w^\*\_1), ...
    - Better approximation - **beam search** (keep track of top b hypothesis - beam size b).
    - Use beam search: +1 BLEU
- Image cation generation
- Kiros et al. (2013) - Innovation: **multiplicative interactions** in the decoder n-gram model
    - Encoder: x = embed(x)
    - Unconditional n-gram model: h=W[w, ...]
    - Simple conditional n-gram LM: h=W[w, ...] + Cx

#### Lecture 8 - Generating Language with Attention

- Compressing a lot of information in a finite-sized vector
- “You can't cram the meaning of a whole %&!$# sentence into a single $&!#* vector!"
- Represent a source sentence as a matrix
    - vectors have the same size but the matrices have different amount of columns
    - with Convolutional Nets
    - most widely used matrix representation - Bidirectional RNNs
        - forward representation - word and its left context
        - reverse representation - word and its right context
        - GRUs or LSTMs
        - about dimension: 500-1000-2000 dimensions are plenty for most of this problems
        - if you use very large vectors and you don't have much training data you tend to overfit. if you don't use enough dimensions you are not gonna learn the function good enough. we don't do a lot of pretraining so we don't often tun word2vec and don't use word embeddings of word2vec. it doesn't seem to help in very many cases. what w2v learns isn't good enough for this type of task.
        - is word2vec a good representation of words for this task?
        - in what types of tasks the pretraining is much effective - if we were working on very lower-resource machine translation task pretraining is probably more effective
        - "Multi-word expressions are a pain in the neck." - maybe LSTMs will help...
- Generation from Matrices
- High-level idea:
    - generate output sentence word by word using an RNN
    - at each position t the RNN receives **two** inputs:
        - a vector of prevs output
        - a vector "view" of the input matrix
    - getting a fixed-sized vector from a matrix that changes over time:
        - weighted sum of the columns of **F** (words) based on how important they are at the current time step (just a matrix-vector product **Fa_t**)
        - the weighting if the input columns at each time-step (a_t) is called **attention**
- What is attention component responsible for? How do we compute attention vector?
- ... compute the **attention energy** ...
- Nonlinear Attention-Energy Model
- `BLEU doesn't always correlate well with human judgements. You can get perfectly good translation with bad BLEU score. It depends a lot on the language that you use. From Chinese to English - you have a lot of optionality. Translation evaluation is a huge problem`
- `If you are comparing two different models on the same text set, you will be able to distinguish models by theirs quality of translation - so BLEU score is meaningful here. In this type of task.`
- Model Variant
- Early binding vs Late binding
- Attention and Translation
- **Image caption generation with attention**
    - ConvNets
    - attention vectors for images
        - Deterministic soft attention
        - Stochastic hard attention
        - analogy with Reinforcement learning...
