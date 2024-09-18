# Makemore: Autoregressive Character-Level Language Model

Makemore is a project aimed at building an autoregressive character-level language model for generating text data. This repository contains five versions of the model, each improving upon the previous one with advancements in architecture, optimization, and training. One key aspect of this project is that many of the foundational libraries and layers were **manually implemented**, offering deeper insights into how neural networks operate under the hood, without relying on pre-built deep learning modules.

## Highlights
- **Manual Implementations**: Instead of relying on high-level libraries like PyTorch or TensorFlow's pre-built layers, key components such as the `Linear`, `BatchNorm1d`, and `Embedding` layers were manually coded. This gives a detailed, ground-up approach to understanding the mechanics of a neural network.
- **Progressive Model Versions**: The project is organized into five incremental versions, each with performance improvements and architecture enhancements, demonstrating how small modifications can yield substantial improvements.

Current implementation follows a few key papers:

- Bigram (one character predicts the next one with a lookup table of counts)
- MLP, following [Bengio et al. 2003](https://www.jmlr.org/papers/volume3/bengio03a/bengio03a.pdf)
- CNN, following [DeepMind WaveNet 2016](https://arxiv.org/abs/1609.03499) (in progress...)
- RNN, following [Mikolov et al. 2010](https://www.fit.vutbr.cz/research/groups/speech/publi/2010/mikolov_interspeech2010_IS100722.pdf)
- LSTM, following [Graves et al. 2014](https://arxiv.org/abs/1308.0850)
- GRU, following [Kyunghyun Cho et al. 2014](https://arxiv.org/abs/1409.1259)
- Transformer, following [Vaswani et al. 2017](https://arxiv.org/abs/1706.03762)

## Key Features

- **Manual Implementation of Key Layers**: The project demonstrates a hands-on approach to deep learning by manually implementing key neural network components such as `Linear`, `BatchNorm1d`, `Embedding`, and `Tanh` layers, as well as custom layers like `FlattenConsecutive`.

- **Autoregressive Nature**: Makemore is an autoregressive model, meaning each character is predicted based on the previously generated characters. The architecture is designed to optimize this process.

- **Hierarchical Model Structure**: As the project progresses, hierarchical structures are introduced to improve the model's performance when generating longer sequences.

- **Contextual Character Prediction**: The model is designed to predict the next character based on an increasing context window, improving the fidelity and coherence of generated names.

- **Batch Normalization**: The project uses batch normalization to help the model train more effectively and generalize better, especially for larger networks.

- **Customizable Embeddings**: The embedding layer used in the models is manually defined, providing flexibility in embedding dimensionality and understanding of how the input data is processed.


## Overview of Versions

### Version 1: Basic Character-Level Model
- **Goal**: Build a simple autoregressive model to predict the next character in a sequence.
- **Manual Implementations**:
  - Custom `Linear` layer for fully connected layers.
  - Simple SGD optimizer coded manually for parameter updates.
- **Performance**:
  - Training Loss: 2.058
  - Validation Loss: 2.105

### Version 2: Context Size Expansion
- **Goal**: Increase the context size to improve the model’s ability to generate coherent text.
- **Manual Implementations**:
  - Custom `Embedding` layer for character embeddings.
  - Handcrafted context windowing mechanism.
- **Performance**:
  - Training Loss: 1.918
  - Validation Loss: 2.027

### Version 3: Hierarchical Network
- **Goal**: Transition from a flat model to a hierarchical network architecture for better representation learning.
- **Manual Implementations**:
  - Custom hierarchical `FlattenConsecutive` layers to handle multiple timesteps of input.
  - Manually implemented `BatchNorm1d` for layer normalization.
- **Performance**:
  - Training Loss: 1.941
  - Validation Loss: 2.029

### Version 4: Batch Normalization Bug Fix
- **Goal**: Fix a bug in the `BatchNorm1d` implementation that hindered performance.
- **Manual Implementations**:
  - Corrected and fine-tuned the Batch Normalization layer.
- **Performance**:
  - Training Loss: 1.912
  - Validation Loss: 2.022

### Version 5: Network Scaling
- **Goal**: Scale up the network for improved performance with a larger capacity.
- **Manual Implementations**:
  - More complex hierarchical layers and network scaling through increased embeddings and neurons.
  - Implemented learning rate decay with manual adjustments based on training steps.
- **Performance**:
  - Training Loss: 1.769
  - Validation Loss: 1.993

### Final Version (makemore.py)
- The final version further improves efficiency and model scaling while keeping training stable.

## Model Architecture
The model’s architecture is built using a combination of manually coded layers, offering insight into the intricate workings of neural networks:
- **Custom Embedding Layer**: Embeds input characters into dense vector spaces.
- **Linear Layer**: A fully connected layer that manually handles matrix multiplications and bias adjustments.
- **BatchNorm1d**: Manually implemented batch normalization for stabilizing activations.
- **Tanh Activation**: Non-linear activation manually integrated into the network.
- **FlattenConsecutive**: Custom function to flatten input for hierarchical processing.

These components, when combined, form a robust autoregressive model capable of character-level text generation.

## Training Procedure
- **Loss Function**: Cross-entropy loss function implemented using PyTorch's lower-level API.
- **Optimizer**: Custom SGD optimizer with a learning rate schedule to adapt during training.
- **Data**: The dataset consists of character sequences from a text file (`names.txt`), split into training, validation, and test sets.

## Sampling Procedure
After training, new sequences are generated character by character. The model autoregressively predicts the next character based on the previously generated ones, until an end-of-sequence token is encountered.

## Usage
To train and use the model, follow these steps:

1. Clone the repository and ensure you have PyTorch and Matplotlib installed.
2. Place the dataset (`names.txt`) in the project directory.
3. Run any of the five versions by executing the corresponding Python file.

The included `names.txt` dataset, as an example, has the most common 32K names takes from [ssa.gov](https://www.ssa.gov/oact/babynames/) for the year 2018. It looks like:

```
emma
olivia
ava
isabella
sophia
charlotte
...
```

Let's point the script at it:

```bash
$ python Final_Makemore.py -i names.txt -o names
```

Training progress and logs and model will all be saved to the working directory `names`. The default model is a super tiny 200K param transformer; Many more training configurations are available - see the argparse and read the code. Training does not require any special hardware, it runs on my Macbook Air and will run on anything else, but if you have a GPU then training will fly faster. As training progresses the script will print some samples throughout. However, if you'd like to sample manually, you can use the `--sample-only` flag, e.g. in a separate terminal do:

```bash
$ python Final_Makemore.py -i names.txt -o names --sample-only
```

This will load the best model so far and print more samples on demand. Here are some unique baby names that get eventually generated from current default settings (test logprob of ~1.92, though much lower logprobs are achievable with some hyperparameter tuning):

```
dontell
khylum
camatena
aeriline
najlah
sherrith
ryel
irmi
taislee
mortaz
akarli
maxfelynn
biolett
zendy
laisa
halliliana
goralynn
brodynn
romima
chiyomin
loghlyn
melichae
mahmed
irot
helicha
besdy
ebokun
lucianno
```

The model will train and output loss values at regular intervals, and once trained, it will be capable of generating new text samples.

## Performance Summary
Here is a summary of performance across all five versions:

| Version | Training Loss | Validation Loss | Parameters |
|---------|---------------|-----------------|------------|
| V1      | 2.058         | 2.105           | 12K        |
| V2      | 1.918         | 2.027           | 22K        |
| V3      | 1.941         | 2.029           | 22K        |
| V4      | 1.912         | 2.022           | 22K        |
| V5      | 1.769         | 1.993           | 76K        |

## Manual Implementation Details

This project is unique because it manually implements much of the deep learning stack, including:

- **Linear Layers**: Manually defined linear layers using basic matrix multiplication.
- **Batch Normalization**: Batch normalization layers, implemented with full control over momentum, running mean, and variance.
- **Embedding Layer**: Embedding of input characters into vector representations is manually handled.
- **Loss and Optimization**: Cross-entropy loss and a basic SGD optimizer with learning rate scheduling have been implemented manually.

This approach adds to the educational value of the project, as it doesn't rely on high-level deep learning frameworks for these components, providing an in-depth understanding of the underlying mechanisms.

## Why Manual Implementations?
This project emphasizes the manual implementation of key neural network components for several reasons:
- **Learning Experience**: Coding components from scratch offers a deep understanding of how neural networks function beyond what high-level APIs provide.
- **Control & Flexibility**: Manually implementing layers and optimizers allows for fine-tuning and custom modifications tailored to the specific use case.
- **Educational Value**: This project serves as a learning tool for those who want to understand how popular deep learning layers and mechanisms, such as batch normalization, embeddings, and linear layers, work at a lower level.

