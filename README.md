# Spatial-Temporal and Text Representation

# Run the code

### Dependencies
* Python 3.7.
* [Anaconda](https://www.anaconda.com/) contains all the required packages.
* [PyTorch](https://pytorch.org/) version 1.4.0.

### Instructions


### Note
* Right now the code only supports single GPU training, but an extension to support multiple GPUs should be easy.
* The reported event time prediction RMSE and the time stamps provided in the datasets are not of the same unit, i.e., the provided time stamps can be in minutes, but the reported results are in hours.
* There are several factors that can be changed, beside the ones in **run.sh**:
  * In **Main.py**, function **train\_epoch**, the event time prediction squared error needs to be properly scaled to stabilize training.
  * In **Utils.py**, function **log_likelihood**, users can select whether to use numerical integration or Monte Carlo integration.
  * In **transformer/Models.py**, class **Transformer**, parameter **alpha** controls the weight of the time difference factor. This parameter can be added into the computation graph, i.e., changeable during training, but the gain is marginal.
  * In **transformer/Models.py**, class **Transformer**, there is an optional recurrent layer. This  is inspired by the fact that additional recurrent layers can better capture the sequential context, as suggested in [this paper](https://arxiv.org/pdf/1904.09408.pdf). In reality, this may or may not help, depending on the dataset.
