# Model training on IMDb dataset using pretrained embeddings from GloVe Database

Maximum length of review - 100 words\
Number of training samples - 8000\
Number of validation samples - 10000

## 1. Train the model using pretrained word embeddings

![alt text](https://github.com/nrajmalwar/Project-1/blob/master/Images/pretrained_glove.jpg)

Model quickly starts overfitting and gives a validation accuracy of ~68%.\
Test set accuracy - **68.9%**

## 2. Train the model without using pretrained word embeddings (without freezing the embedding layer)

![alt text](https://github.com/nrajmalwar/Project-1/blob/master/Images/without_pretrained_glove.jpg)

Here, we learn a task specifc embedding of the input tokens, which is generally more powerful than pretrained word embeddings
when lots of data is available.\
Test set accuracy - **82.12%**
