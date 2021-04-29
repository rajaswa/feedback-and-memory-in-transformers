# feedback-and-memory-in-transformers
My final project submission for the Meta Learning course at BITS Goa (conducted by TCS Research & BITS Goa)

## Feedback Transformer Implementation
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

## Citation
If you use this code, cite the original paper:
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

## Contact
Submit an issue.
