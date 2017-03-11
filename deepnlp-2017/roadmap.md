# DeepNLP [[oxford-cs-deepnlp-2017]](https://github.com/oxford-cs-deepnlp-2017/lectures)

**TODO:**
- [x] Lecture 1a
- [x] Lecture 1b
- [ ] Lecture 2a
- [ ] Lecture 2b
- [ ] Lecture 3
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
- "you shall know a person by the company he keeps"
- Count-based methods:
    - **basic vocab C** of context words
    - **word window** size w
    - **count basic vocab words** - w words to the left or right from a target word in the corpus
    - form a **vector representation** of the target word based on these counts
- Neural embedding models
    - count based vectors produce **embedding matrix**
    - CBoW
    - negative sampling (???)
    - Skip-gram
    - PMI matrix factorization (???)
