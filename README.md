# Project-22: Feedback and Memory in Transformers
My final project submission for the Meta Learning course at BITS Goa (conducted by TCS Research & BITS Goa). The project can be run as a colab notebook [![colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/rajaswa/feedback-and-memory-in-transformers/blob/main/Feedback_and_Memory_in_Transformers.ipynb)

## Key Contributions
The key contributions of this project can be listed as follows:
1. [Implementing and Open-sourcing a modular customizable Feedback Transformer Model in PyTorch](https://github.com/rajaswa/feedback-and-memory-in-transformers#feedback-transformer-implementation)
2. [Experimenting the Feedback Transformer Model with COGS Benchmark (Compositional Generalization)](https://github.com/rajaswa/feedback-and-memory-in-transformers#solving-cogs-with-feedback-transformer)
3. [Implementing the Sequence Copy & Reverse Task from the original Feedback Transformer Paper](https://github.com/rajaswa/feedback-and-memory-in-transformers/blob/main/README.md#sequence-copy--reverse-task)

### Feedback Transformer Implementation
The Feedback Transformer Model has been implemented as PyTorch model class in the given notebook. You can adjust the various hyperparameters and turn the feedback ON/OFF in the Encoder and Decoder of the Model independently. Use the model in the following manner:
```python
model = FeedbackTransformerModel(
            encoder_feedback = False,   # Disable Feedback Mehancism in the Encoder
            decoder_feedback = True,    # Enable Feedback Mehancism in the Decoder
            memory_context = 8,         # How long to look in the past for Memory-attention
            input_vocab_size = 800,     # Input Vocabulary Size
            output_vocab_size = 800,    # Output Vocabulary Size
            d_model = 128,              # Model Embedding Dimension
            nhead = 8,                  # Number of Heads in Multi-head Cross-attention and Memory-attention
            num_layers = 4,             # Number of Encoder and Decoder blocks
            dim_feedforward = 256,      # Feedforward Dimension
            max_seq_length = 1000,      # Maximum Sequence Length in Data
            dropout = 0.1,              # Model Dropout Probability 
            PAD_IDX = 0,                # PAD Token ID to Mask Padding tokens for Attention
            activation = "gelu",        # Model Activation Function: "gelu" / "relu"
    )
```


### Solving COGS with Feedback Transformer
The [COGS Benchmark](https://github.com/najoungkim/COGS) is a benchmark for evaluating **compositional generalization & reasoning** in natural language. The COGS task is that of mapping a **natural language sentence to a lambda-expression based semantic logical form**:

```python
input_sentence = "The moose wanted to read ."
output_logical_form = "* moose ( x _ 1 ) ; want . agent ( x _ 2 , x _ 1 ) AND want . xcomp ( x _ 2 , x _ 4 ) AND read . agent ( x _ 4 , x _ 1 )"
```

This can be treated as a **sequence-to-sequence semantic-parsing** task. What makes this task challenging is its **Generalization test set**. The following points make it quite challenging:
1. Novel (unseen in training) Combination of Familiar Primitives and Grammatical Roles
2. Novel (unseen in training) Combination Modified Phrases and Grammatical Roles
3. Deeper Recursion (results in longer sentences and deeper lingusitic strucutre i.e. parse tree)
4. Verb Argument Structure Alternation
5. Verb Class Alteration

You can check the [COGS Paper](https://www.aclweb.org/anthology/2020.emnlp-main.731.pdf) for more details on the benchmark.

**NOTE**: _This is the first attempt (to the best of my knowledge) to inspect the effect of incoroporating feedback and memory based architectural biases in solving compositional generalization problem in natural language._


### Sequence Copy & Reverse Task


## Citations
If you use the code for Feedback Transfomer or the Sequence Copy & Reverse task, cite the Feedback Transformer paper:
```
@misc{fan2021addressing,
      title={Addressing Some Limitations of Transformers with Feedback Memory}, 
      author={Angela Fan and Thibaut Lavril and Edouard Grave and Armand Joulin and Sainbayar Sukhbaatar},
      year={2021},
      eprint={2002.09402},
      archivePrefix={arXiv},
      primaryClass={cs.LG}
}
```

If you use the code from COGS Benchmark data processing and loading, cite the COGS paper:
```
@inproceedings{kim-linzen-2020-cogs,
    title = "{COGS}: A Compositional Generalization Challenge Based on Semantic Interpretation",
    author = "Kim, Najoung  and
      Linzen, Tal",
    booktitle = "Proceedings of the 2020 Conference on Empirical Methods in Natural Language Processing (EMNLP)",
    month = nov,
    year = "2020",
    address = "Online",
    publisher = "Association for Computational Linguistics",
    url = "https://www.aclweb.org/anthology/2020.emnlp-main.731",
    doi = "10.18653/v1/2020.emnlp-main.731",
    pages = "9087--9105",
    abstract = "Natural language is characterized by compositionality: the meaning of a complex expression is constructed from the meanings of its constituent parts. To facilitate the evaluation of the compositional abilities of language processing architectures, we introduce COGS, a semantic parsing dataset based on a fragment of English. The evaluation portion of COGS contains multiple systematic gaps that can only be addressed by compositional generalization; these include new combinations of familiar syntactic structures, or new combinations of familiar words and familiar structures. In experiments with Transformers and LSTMs, we found that in-distribution accuracy on the COGS test set was near-perfect (96{--}99{\%}), but generalization accuracy was substantially lower (16{--}35{\%}) and showed high sensitivity to random seed (+-6{--}8{\%}). These findings indicate that contemporary standard NLP models are limited in their compositional generalization capacity, and position COGS as a good way to measure progress.",
}
```

## Contact
Submit an issue.
