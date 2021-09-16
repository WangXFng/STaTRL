# Spatial-Temporal and Text Representation (STaTR)

# Run the code

### Dependencies
* Python 3.7.
* [Anaconda](https://www.anaconda.com/) contains all the required packages.
* [PyTorch](https://pytorch.org/) version 1.4.0.

### Instructions

### 1 preprocessing
* If you already have the processed data which are [train_6.pkl](https://drive.google.com/file/d/17hVpGDsRuRnocdaLHxFHvO3cvmdgi_ZA/view?usp=sharing) and [test_6.pkl](https://drive.google.com/file/d/1Nt_zKTWYmIPZbS1AlLDeDiUJl1xIjVZn/view?usp=sharing) , skip to step 2.
* otherwise
  * In **albert**, there are five steps to preprocess the original [Yelp open review data](https://www.yelp.com/dataset), which we used is version-2020.
  * step 1 training the albert-model.
  * step 2 filtering the POIs which have less than 10 vivits and users who has visited less than 10 POIs, and obtain the numeric ids of POIs and users.
  * step 3 to generate the dictionary which you can obtain the top category through the numeric ids of the POIs.
  * step 4 to obtain the text visit sequence -- Yelp_reviews_test.txt.
  * step 5 to compute the distance matrix of all POIs.
  * step 6 generate the input data, train_6.pkl and test_6.pkl

### 2 train STaTR
#> sh run.sh

### Note
* Right now the code only supports single GPU training, but an extension to support multiple GPUs should be easy.
* The reported event time prediction RMSE and the time stamps provided in the datasets are not of the same unit, i.e., the provided time stamps can be in minutes, but the reported results are in hours.
* There are several factors that can be changed, beside the ones in **run.sh**:
  * In **Main.py**, function **train\_epoch**, the event time prediction squared error needs to be properly scaled to stabilize training.
  * In **Utils.py**, function **log_likelihood**, users can select whether to use numerical integration or Monte Carlo integration.
  * In **transformer/Models.py**, class **Transformer**, parameter **alpha** controls the weight of the time difference factor. This parameter can be added into the computation graph, i.e., changeable during training, but the gain is marginal.
  * In **transformer/Models.py**, class **Transformer**, there is an optional recurrent layer. This  is inspired by the fact that additional recurrent layers can better capture the sequential context, as suggested in [this paper](https://arxiv.org/pdf/1904.09408.pdf). In reality, this may or may not help, depending on the dataset.
