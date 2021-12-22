# Variational Auto-Encoder - PyTorch (lightning) implementation

For more info, see the [original paper](https://arxiv.org/abs/1312.6114) and my [notes](https://timothydelille.github.io/content/stanford_cs228_probabilistic_graphical_modeling.html#variational-auto-encoder) based on Stanford CS228 lecture notes.

# Auto-encoding crypto punks

The crypto-punks are stored in an image containing 10,000 crypto punks with varying traits. As they are already randomly sampled, I reserve punks 1 to 6000 for the training set, 6001 to 8000 to the validation set and punk 8001 to 10000 to the test set.

# Notes

- Using ReLU activations and Adam optimizer (as opposed to tanh and Adagrad in the original paper) speeds up training.
- Increasing depth instead of width seems to result in better performance for very low latent space dimensions (e.g. 2D latent space).
