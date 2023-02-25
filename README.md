# Spatial-Temporal and Text Representation Learning (STaTRL) for Point-of-Interest (POI) recommendation

### Applied Intelligence 2022. IF=5.019 JCR分区（Q2）

### [STaTRL without Text Representation Learning（STRL）](https://github.com/WangXFng/STRL) could be easily conducted, without processing Text.

| dataset | # Users | # POI |
|---------|---------|---------|
| Yelp2020     | 28038     |# 15745 |
| Yelp    | 30887     |# 18995 |
| Gowalla     | 18737     |# 32510 |
| Foursquare     | 7642     |# 28483 |

### Dependencies
* Python 3.7.6
* [Anaconda](https://www.anaconda.com/) 4.8.2 contains all the required packages.
* [PyTorch](https://pytorch.org/) version 1.7.1.


### Instructions

### 1. Preprocessing
* If you have already got the processed data which are [train_6.pkl](https://drive.google.com/file/d/17hVpGDsRuRnocdaLHxFHvO3cvmdgi_ZA/view?usp=sharing) and [test_6.pkl](https://drive.google.com/file/d/1Nt_zKTWYmIPZbS1AlLDeDiUJl1xIjVZn/view?usp=sharing), skip to 2. (Training).
* otherwise
  * In the folder of **albert**, there are five steps to preprocess the original [Yelp open review data](https://www.yelp.com/dataset), which we used is version-2020.
  * step 1 training the albert-model.
  * step 2 filtering the POIs which have less than 10 visits and users who have visited less than 10 POIs, and obtain the numeric ids of POIs and users. In this step you can obtain the number of POIs and users, then you may need modify **TYPE_NUMBER** and **USER_NUMBER** in **transformer.Constants**.
  * step 3 to generate the dictionary which you can obtain the top category through the numeric ids of the POIs.
  * step 4 to obtain the text visit sequence -- Yelp_reviews_test.txt.
  * step 5 to compute the distance matrix of all POIs.
  * step 6 generate the input data, train_6.pkl and test_6.pkl

### 2. Training
> sh run.sh

### Note
* Right now the code only supports single GPU training, but an extension to support multiple GPUs should be easy.
* If there is any problem, please contact to kaysen@hdu.edu.cn or fukumoto@yamanashi.ac.jp.
