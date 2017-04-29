### Lecture 13 - Linguistic Knowledge in Neural Networks

Sentences  are hierarchical.

Generalization: `not` must "structurally precede" `anybody`

RNNs are capable enough to compute any desired function [because they are Turing complete].

**Inductive bias**

Chomsky: "Sequential recency is not the right bias for effective learning of human language."

Representing a sentence:
- with RNNs - not good
- Syntax as Trees - better
- **Recursive** Neural Network on Syntax Trees - Socher et al. (ICML 2011)

Representing Sentences:
- Bag of words/n-grams
- CNN
- RNN
- Recursive NN

Stanford Sentiment Treebank [Socher et al. (2013)]

Internal Supervision

Many Extensions:
- ...
- Improved gradient dynamics using tree cells defined in terms of LSTM updates with gating instead of RNN. **Exercise**: _generalize the definition of a sequential LSTM to the tree case. Check the paper._

**Recursive vs. Recurrent**

| Advantages | Disadvantages     |
| :------------- | :------------- |
| meaning decomposes roughly according to the syntax of a sentence -- better inductive bias       | we need parse trees      |
| shorter gradient paths on average (log(n) in the best case) | trees tend to be right-branching -- gradients still have a long way to go!|
| internal supervision of the node representations ("auxiliary objectives") is sometimes available |more difficult to batch than RNNs|

**Recurrent Neural Net _Grammars_**

- Generate symbols sequentially using an RNN
- Add some control symbols to rewrite the history occasionally
    - Occasionally compress a sequence into a constituent
    - RNN predicts next terminal/control symbol based on the history of compressed elements and non-compressed terminals
- This is a top-down, left-to-right generation of a tree+sequence

| stack | action     | probability |
| :------------- | :------------- | :------------- |
| ...       | ...  | ... |
| (S(NP _The hungry_  | GEN(_cat_) | p(GEN(_cat_) |
|...  | .. | ... |
| (S(NP _The hungry cat_) |  .. | ..  |

Compress "The hungry cat" into a single composite symbol.

Stack RNNs operation:
- Augment RNN with a `stack pointer`
- Two constant-time operations:
    - `Push` - read input, add to top of stack
    - `Pop` - move stack pointer back
- A summary of stack contents is obtained by accessing the output of the RNN at location of the stack pointer

RNNGs Inductive bias

- `Discriminative` PTB - models sentence x and its parsing tree y jointly
- `Generative` PTB - models parsing tree y by given sentence x

Transition-based parsing

| stack | buffer     | action |
| :------------- | :------------- | :------------- |
| ...       | ...       | ... |

Word representation:
- Arbitrariness (cat - c + b = bat)
- Opportunity (cool | coooool)

Word as structured objects:
- Normal word vector
- Morphological word vector
- Character-based word vector

Generating new word forms
