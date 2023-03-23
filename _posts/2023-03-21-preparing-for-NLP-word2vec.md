---
layout: paper-note
title: "Preparing notes for NLP (word2vec)"
description: Understanding of skip-gram model with formulas and image illustrations
date: 2023-03-21

paper_type: None
paper_url: None
code_type: None
code_url: None

bibliography: paper-notes.bib

# Optionally, you can add a table of contents to your post.
# NOTES:
#   - make sure that TOC names match the actual section names
#     for hyperlinks within the post to work correctly.
#   - we may want to automate TOC generation in the future using
#     jekyll-toc plugin (https://github.com/toshimaru/jekyll-toc).
toc:
  - name: Takeaways
  - name: Introduction
    subsections:
    - name: What is Word2Vec?
    - name: Skip-gram model
  - name: The model architectture
    subsections:
    - name: The language model task
    - name: The Skip-gram model in details
  - name: Conclusion
  - name: References

---
## Takeaways

- Understanding the motivation of using neural network to create word embeddings.
- Understanding the dataset and how predicting neighboring-word model works.
- The row of updating weight matrix in the hidden layer is the word embedding corresponding to the word in the vocabulary.
- Continue reading the next post of **Negative Sampling** and **Comparison of Skipgram and CBOW**.


## Introduction

### What is Word2Vec?
- That computers can only understand the numbers( let's say array of numbers) motivated us to create the word embeddings for all the words in the vocabulary. 

- Other approaches for representing words we might have used in other tasks are **one-hot encoding vectors** or **bag-of-words representations**. But all of those suffer from several drawbacks that we don't talk here (It doesn't matter!!!).

### Skip-gram model
- There are two main techniques of modern word embeddings: **Skip-gram** and **Continous bag of Words (CBOW)**. But we only talk about **Skip-gram** here because **CBOW** is pretty much the same except for the goal of the model (we will talk about it later.) 

- The **Skip-gram** model have expanded name which is **Skip gram neural network model**. To be simplified, we are going to train a simple neural network with a single hidden layer to perform such a weird task. The "weird task" we mention is that we don't use that neural network to output something as we expect, instead we use the **weights** of the hidden layer to create our word embeddings. 

- Believe me, at first I don't believe it works but that is the interesting thing, let's dive right into the model.

## The model architecture

### Overview of the language model task
- One of the best example of language model tasks is the next-word prediction of a smartphone keyboard. 

<div class="l-body" style="text-align:center;">
  <img src="https://ugtechmag.com/wp-content/uploads/2022/03/w5J1V.png" width="50%" style="margin-bottom: 12px; background-color: white;">
  <p>Figure 1: Next word prediction task</p>
</div>

- The above task is something like we take the input of 4 words **I**, **had**, **such**, **a**, then put it to the language model and the output will be all probabilities of possible words in the vocabularies (here **great**, **great time**, **lovely** are 3 words with highest probabilities).

- So the motivation for **Skip-gram model** is the same but reverse. So for example, we have a sentence 

                                "I had such a great time"

- The **Skip-gram model** will take the input of the center word (let'say **a**) and predicting the neighboring words with the size of slicing window. The **window size** is how many words you want to be in the output (if window size to 2, four words **had**, **such**, **great**, **time** will be the outputs) 

- The **training sample** will be like this:

<div class="l-body" style="text-align:center;">
  <img src="https://media.geeksforgeeks.org/wp-content/uploads/word2vec_diagram-1.jpg" width="70%" style="margin-bottom: 12px; background-color: white;">
  <p>Figure 2: Example of training sample</p>
</div>

- The network is going to learn the statistics from the number of times each pairing shows up. So, for example, the network is probably going to get many more training samples of (“fox”, “jumps”) than it is of (“fox”, “flies”). When the training is finished, if you give it the word “fox” as input, then it will output a much higher probability for “jumps” or “brown” than it will for “flies”.

### The Skip-gram model in details

- First, we all know that we cannot put a word in type of string to the model. So, we have to put this in type of number, and one-hot encoding vector is the easiest way. Let's say we have the vocabulary of 10000 unique words.

- So in the model below, we have to represent the input word "beautiful" as an array of 10000 dimension with 1 element value equals to 1, others will be 0. It is simply the position of the word ""beautiful"" in the vocabulary.

<div class="l-body" style="text-align:center;">
  <img src="https://i0.wp.com/towardsmachinelearning.org/wp-content/uploads/2022/04/SkipGram-Model-Architecture-1.webp?resize=648%2C593&ssl=1" width="70%" style="margin-bottom: 12px; background-color: white;">
  <p>Figure 3: Model architecture</p>
</div>

So questions are raised from here:

- How about the hidden layer?
- How about the output layer?
- What is the dimension of weight matrix and also the input, hidden layer and output?

So I will give the step-by-step in details from the input to the output through the hidden layer and how we can get the word embedding matrices from the weights matrix

1) Let's first define some variables:
- **V** is the size of the vocabulary (i.e., the number of unique words in the corpus)
- **N** is the dimensionality of the word embeddings
- **W** is the embedding matrix of shape (V, N)
- **C** is the context matrix of shape (N, V)

2) The weight matrix and the output of hidden layer
- The input of word "beautiful" with dimension of (1, V)
- Then multiply with a weight matrix of (V, N), it produces the output of dimension: (1,V) * (V,N) = (1, N). This is exactly of dimension of word embedding we want. 
- Let's understand it in another way
<div class="l-body" style="text-align:center;">
  <img src="https://miro.medium.com/v2/resize:fit:1400/0*6DOQn6gxvEoix0yn.png" width="60%" style="margin-bottom: 12px; background-color: white;">
  <p>Figure 4: Example of hidden layer weight matrix</p>
</div>
<div class="l-body" style="text-align:center;">
  <img src="https://i0.wp.com/towardsmachinelearning.org/wp-content/uploads/2022/04/image-6.webp?ssl=1" width="60%" style="margin-bottom: 12px; background-color: white;">
  <p>Figure 5: The output of hidden layer</p>
</div>

- Here we can see that each row of the weight matrix works like the word embedding for the one-hot encoding vector input, so the out put vector after doing the multiplication is already the the word embedding vector that we want.

3) The output layer
- After getting the output of hidden layer with the dimension of (1, N), we have to multiply this output with the context matrix of dimension (N, V) to get back to the size of word in vocabulary: (1, N) . (N, V) = (1, V)
- Then we use softmax function to display the vector related to probabilities of all words in the vocab. 
- Imagine that we have to calculate the loss with the true label be one-hot encoding vector like we have in the initial step. 
- Then we do several steps related to calculating the gradient, update parameter, and doing the new step.

4) The last result

- After several steps, we will get the best weight matrix.
This is exactly what we want.
- Each row of this weight matrix is the word embedding for the corresponding word in the vocabulary.

## Conclusion

- We have all went through all the steps to create the word embedding. That is absolutely interesting idea to using weight matrix to create word embedding.
- Understanding the motivation of algorithms under the hood will make us be more active in the field of AI and Deep Learning.
- The next part, we will talk about a method to optimize the training process of generating word embedding which is **negative sampling**. 

See you guys in the next post!

## References

- Visualize the Word2Vec: https://jalammar.github.io/illustrated-word2vec/
- Explanation of Word2Vec: https://towardsdatascience.com/word2vec-explained-49c52b4ccb71
- Original paper: https://arxiv.org/abs/1301.3781





