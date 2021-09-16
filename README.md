# Spatial-Temporal and Text Representation (STaTR)


### Dependencies
* Python 3.7.
* [Anaconda](https://www.anaconda.com/) contains all the required packages.
* [PyTorch](https://pytorch.org/) version 1.4.0.

### Instructions

### 1. preprocessing
* If you have already got the processed data which are [train_6.pkl](https://drive.google.com/file/d/17hVpGDsRuRnocdaLHxFHvO3cvmdgi_ZA/view?usp=sharing) and [test_6.pkl](https://drive.google.com/file/d/1Nt_zKTWYmIPZbS1AlLDeDiUJl1xIjVZn/view?usp=sharing) , skip to step 2.
* otherwise
  * In **albert**, there are five steps to preprocess the original [Yelp open review data](https://www.yelp.com/dataset), which we used is version-2020.
  * step 1 training the albert-model.
  * step 2 filtering the POIs which have less than 10 vivits and users who has visited less than 10 POIs, and obtain the numeric ids of POIs and users. In this step you can obtain the number of POIs and users, then you should mortify the setup of **TYPE_NUMBER** and **USER_NUMBER** in **transformer.Constants**.
  * step 3 to generate the dictionary which you can obtain the top category through the numeric ids of the POIs.
  * step 4 to obtain the text visit sequence -- Yelp_reviews_test.txt.
  * step 5 to compute the distance matrix of all POIs.
  * step 6 generate the input data, train_6.pkl and test_6.pkl

### 2. train STaTR
#> sh run.sh

### Note
* Right now the code only supports single GPU training, but an extension to support multiple GPUs should be easy.
