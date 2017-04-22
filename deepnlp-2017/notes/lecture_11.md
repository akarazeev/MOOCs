#### Lecture 11 - Question Answering

- Source of answer
- Semantic Parsing: Q -> Logical Form -> `KB` (Knowledge Base) Query -> Answer
- (relation, entity1, entity2); Google Knowledge Graph, WikiData
- KBs are cheap but Supervised Data is expensive
- Semantic parsing can be viewed as seq2seq model
    - using attention can be useful
    - Pointer Networks [[link](https://arxiv.org/pdf/1506.03134.pdf)]
- Each of these examples [some questions and statements] requires a different underlying KB.
- Corpora for `Reading Comprehension`
    - CNN/DailyMail - replace all entities with anonymized markers. It reduces vocab size
    - Generic Neural Model for Reading Comprehension: p(a|q,d) ~ exp(W(a)g(q,d)), a \in vocab
    - [using Bidirectional RNN] RC with Attention
- `Attentive Reader Training` (using RMSprop)
- `Attention Sum Reader` (Kadlec et. al. 2016, Text Understanding with the ASR network [[link](https://arxiv.org/pdf/1603.01547.pdf)])

| pros     | cons     |
| :------------- | :------------- |
| ask questions in context  | constraint on context often artificial  |
| easily used in discriminative and generative fashion | many types of questions are unanswerable |
| large dataset available  | _  |

- **Answer Sentence Selection**: the Answer is guaranteed to be _extracted_, while in `reading comprehension` it could be either _generated_ or _extracted_.

| _     | _     |
| :------------- | :------------- |
| **Questions**      | Factual questions, possibly with context       |
| **Data Source** | "The Web" or the output of some `IR` (information retrieval) system |
| **Answer**  | One or several excerpts pertinent to the answer |

- Neural Model for Answer Sentence Selection (Yu et. al. [[link](https://arxiv.org/pdf/1412.1632.pdf)])
    - we need to compute the prob of an answer candidate `a` and a question `q` matching
    - p(y=1|q,a) = \sigma (q M a + b)
- **Evaluation**

| Measure | Description     |
| :------------- | :------------- |
| Accuracy       | Binary measure   |
| Mean Reciprocal Rank  | Measures position of first relevant document in return set  |
| BLEU score | Machine Translation measure for translation accuracy |

- **Answer Selection Summary**

| pros | cons     |
| :------------- | :------------- |
| designed to deal with large amounts of context    | does not provide answers, provides context only   |
| more robust than 'true' QA systems as it turns provides context with its answers  | real-world use depends on underlying IR pipeline |
| obvious pipeline step between IR and QA | _ |

- Visual Question Answering
- Attention Methods for Visual QA (Yang et. al. (2015): Stacked Attention Networks for Image Question Answering)
- **Visual Question Answering Summary**

| pros | cons  |
| :------------- | :------------- |
| extra modality 'for free'    |  currently quite gimmicky  |
| plenty of training data available as of recently | still a long way to go |
