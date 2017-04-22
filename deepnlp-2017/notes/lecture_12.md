### Lecture 12 - Memory

Incorporating information. Draw conclusion. Memory is important aspect of NLP and NL understanding.

`RNNs:`
- Unfold to feedforward net with shared weights
- Applications:
    - Language modelling
    - Sequence labelling
    - Sentence classification

Transduction with Conditional modelling.

Deep LSTMs for Translation [Sutskever et al. NIPS 2014]: Source sequence -> Target sequence.

Use the same architecture from Machine translation task for calculations [Zaremba and Sutskever, 2014].

Bottleneck between Source and Target sequence when doing a transduction. If you translate more and more longer sentence you have to compress more information into the same dimensions.

#### Limitations of RNNs

Finite State Machines (regular languages) -> Pushdown Automata (context free languages) -> Turing Machines (computable functions)

RNN models history sequence and generates output sequence.

"Question: RNNs are computing/outputs target sequence from source.. Who thinks RNN is close to Finite State Machine? Pushdown Automata? Turing machines? RNNs are close to Turing Machines, they are Turing complete."
