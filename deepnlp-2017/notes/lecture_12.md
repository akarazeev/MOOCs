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

`Finite State Machines` (regular languages) **->** `Pushdown Automata` (context free languages) **->** `Turing Machines` (computable functions)

RNN models history sequence and generates output sequence.

"Question: RNNs are computing/outputs target sequence from source... Who thinks RNN is close to Finite State Machine? Pushdown Automata? Turing machines? RNNs are close to Turing Machines, they are Turing complete."

RNNs and Turing Machines [Sieglemann & Sontag 1995]

By extension RNNs can express/approximate a set of Turing machines. But simple RNNs (basic, GRU, LSTM) cannot learn Turing Machines.

`RNNs as Finite State Machines` [[interesting post](https://stats.stackexchange.com/questions/220907/meaning-of-and-proof-of-rnn-can-approximate-any-algorithm)]

#### Natural Language is at least Context Free.

Consider model (a^n b^n), n is never more than N.

| Regular language (N+1 rules) | Context Free Grammar (CFG) (2 rules) |
| :------------- | :------------- |
| E(ab)(aabb).. | S -> a S b </br> S -> E   |

Express solution to the problem with just a few rules.

"Where is Boltzmann machine in this hierarchy?"

"learnable =?= representable ?"

#### RNNs: More API than Model

RNN: X x P -> Y x N where X is input, P - previous recurrent state, Y is an output, N - updated recurrent state

#### The Controller-Memory Split

Memory <--> Controller

#### Attention

- `Early` Fusion - the controller is updated and conditioned on inputs, previous state and some read from memory.
- `Late` Fusion - internal state will tell the model which memory to look at, and use that to influence the outputs of controller.

Read Only Memory (ROM) for Encoder-Decoder Models

`Recognizing Textual Entailment (RTE)`

Word-by-Word Attention

... pointer networks.

#### Neural RAM

- Controller is responsible for every memory [not only] operation.
- Experiments show better generalisation on symbolic tasks [Graves at al. 2014].

`Register Machines and NL understanding (NLU)`

- Complex reasoning probably requires smt both more expressive and more structured than RNNs + Attention.
- These architectures are complex, hard to train, many design choices.
- Make research here :) But don't forget about simpler models.

`Neural Stack` -> Controller API

RNN -> Neural Stack

- Example: A Continuous Stack
- Synthetic Transduction Tasks
- Neural Stack/Queue/DeQue

**Neural PDA Summary**
- Decent approx of classical PDA
- architectural bias towards recursive/nested dependencies
- Should be useful for syntactically rich natural language
    - Parsing
    - Compositionality
