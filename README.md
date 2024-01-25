# Transformer and Character-Level Language Model

## Overview

This repository contains four implementations for text generation:
1. **Transformer Model**: A text generation model based on the Transformer architecture.
2. **Character-Level Language Models**: Custom character-level language models using a basic architecture.

For detailed explanations and insights, refer to the accompanying [explanation](https://github.com/khethan123/Makemore/blob/main/NLP/explanation.pdf) file.

## Files

- [`Reccurent`](https://github.com/khethan123/Makemore/blob/main/NLP/recurrent.py): Implementation of the Recurrent architecture text generation model.
- [`bigram`](https://github.com/khethan123/Makemore/blob/main/NLP/bigram.py): Implementation of the character-level language model.
- [`wavenet`](https://github.com/khethan123/Makemore/blob/main/NLP/wavenet.py): Implementation of the character-level language model using wavenet logic.
- [`transformer`](https://github.com/khethan123/Makemore/blob/main/NLP/transformer.py): Implementation of the character-level language model using transformer architecture.


## Important Hyperparameters

- `BATCH_SIZE`: The number of independent sequences processed in parallel.
- `BLOCK_SIZE`: The maximum context length for predictions.
- `N_EMBD`: Dimensionality of the character embedding vectors.
- `N_HEAD`, `N_LAYER`: Parameters for the Transformer model.
- `DROPOUT`: Dropout rate for regularization.
- `LEARNING_RATE`: Learning rate for the optimization algorithm.
- `MAX_ITERS`: Maximum number of training iterations.

Adjust hyperparameters in the script, as needed and run them using the FILE_NAME to train these models.
```bash
python FILE_NAME.py
```

Remember to adjust `max_new_tokens` for controlling the length of generated text.

## Output 

<div style="display: flex;">
  
  <div style="flex: 50%; padding: 5px; border: 1px solid black;">
    
    juli
    alayah
    maycen
    angelyn
    kaisha
    merick
    kamar
    jayana
    jemir
    alyaan
    mareli
    maycie
    angely
    anniyah
    shayde
    sona
    adian
    
  </div>

  <div style="flex: 50%; padding: 5px; border: 1px solid black;">
    
    KING RICHARD III:
    The law be made of boar
    Upon my ready that hath bar happily quick.
  
    SICINIUS:
    If, I can my fached, when must outs
    Make it is covert now in a grief,
    Evencious time in overs, and the news!
    
    SICINIUS:
    Nay, meaning she course to up; and the tyrant in's norfolk.
  </div>
</div>
The names were generated after training the model with the names.txt file and the Shakespeare text was generated after training the model with input.txt file.
