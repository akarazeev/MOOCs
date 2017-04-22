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
