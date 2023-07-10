# Amazon-Review-Sentiment-Analysis

The project focuses on sentiment analysis of Amazon reviews. The goal is to develop a model that can accurately classify reviews as positive or negative.

- Preprocessing: Various techniques were applied to preprocess the data. Special characters were removed and tokenization was performed to break down the reviews into individual words or tokens.
- Word2Vec Model: A word2vec model was implemented to effectively represent the vocabulary of the Amazon Reviews dataset. Word2vec helps capture semantic relationships between words by generating word embeddings.
- Neural Network Model: A neural network model was developed to classify the sentiment of the reviews. The model was trained using the preprocessed data and the word embeddings obtained from the word2vec model. The aim was to achieve high accuracy in determining whether a review is positive or negative.
- Test Accuracy: The performance of the developed model was evaluated using a test dataset. The model achieved a test accuracy of 80%, indicating its effectiveness in sentiment analysis of Amazon reviews.


## Classification Accuracy Results

Activation Fn | Dropout | Test Accuracy |
---------- | ---------- | ---------- |
ReLU | 0.1 | 80.72% | 
Sigmoid | 0.1 | 81.21% |
Tanh | 0.3 | 80.82% |

The best performance is achieved on using Sigmoid activation function followed by ReLU and Tanh. This is because Sigmoid activation provides output between 0 to 1 which acts as a probability and is apt for classification problem.