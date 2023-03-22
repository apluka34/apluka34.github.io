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
    # subsections:
    # - name: Image Synthesis
  - name: 
    subsections:
    # - name: Perceptual Image Compression (Autoencoder)
  - name: Model
    # subsections:
    # - name: Evaluation
  - name: References

---
## Takeaways

## Introduction
### What is Word2Vec?
- That computers can only understand the numbers( let's say array of numbers) motivated us to create the word embeddings for all the words in the vocabulary. 

- Other approaches for representing words we might have used in other tasks are one-hot encoding vectors or bag-of-words representations. But all of those suffer from several drawbacks that we don't talk here (It doesn't matter!!!).
### Skip-gram model
- There are two main techniques of modern word embeddings: **Skip-gram** and **Continous bag of Words (CBOW)**. But we only talk about **Skip-gram** here because **CBOW** is pretty much the same except for the goal of the model (we will talk about it later.) 

- The **Skip-gram** model have expanded name which is **Skip gram neural network model**. To be simplified, we are going to train a simple neural network with a single hidden layer to perform such a weird task. The "weird task" we mention is that we don't use that neural network to output something as we expect, instead we use the **weights** of the hidden layer to create our word embeddings. 

- Believe me, at first I don't believe it works but that is the interesting thing, let's dive right into the model.

## The model architectture

### Overview of the language model task
- One of the best example of language model tasks is the next-word prediction of a smartphone keyboard. 

<div class="l-body" style="text-align:center;">
  <img src="https://ugtechmag.com/wp-content/uploads/2022/03/w5J1V.png" width="50%" style="margin-bottom: 12px; background-color: white;">
  <p>Example of next word prediction task</p>
</div>

- The above task is something like we take the input of 4 words **I**, **had**, **such**, **a**, then put it to the language model and the output will be all probabilities of possible words in the vocabularies (here **great**, **great time**, **lovely** are 3 words with highest probabilities).

- So the motivation for **Skip-gram model** is the same but reverse. So for example, we have a sentence 

                    "I had such a great time"

- The **Skip-gram model** will take the input of the center word (let'say **a**) and predicting the neighboring words with the size of slicing window. The **window size** is how many words you want to be in the output (if window size to 2, four words **had**, **such**, **great**, **time** will be the outputs) 

- The **training sample** will be like this:

<div class="l-body" style="text-align:center;">
  <img src="https://media.geeksforgeeks.org/wp-content/uploads/word2vec_diagram-1.jpg" width="70%" style="margin-bottom: 12px; background-color: white;">
  <p>Example of training sample</p>
</div>

- The network is going to learn the statistics from the number of times each pairing shows up. So, for example, the network is probably going to get many more training samples of (“fox”, “jumps”) than it is of (“fox”, “flies”). When the training is finished, if you give it the word “fox” as input, then it will output a much higher probability for “jumps” or “brown” than it will for “flies”.

### The Skip-gram model in details

- First, we all know that we cannot put a word in type of string to the model. So, we have to put this in type of number, and one-hot encoding vector is the easiest way. Let's say we have the vocabulary of 10000 unique words.

- So in the model below, we have to represent the input word "beautiful" as an array of 10000 dimension with 1 element value equals to 1, others will be 0. It is simply the position of the word ""beautiful"" in the vocabulary.

<div class="l-body" style="text-align:center;">
  <img src="https://i0.wp.com/towardsmachinelearning.org/wp-content/uploads/2022/04/SkipGram-Model-Architecture-1.webp?resize=648%2C593&ssl=1" width="70%" style="margin-bottom: 12px; background-color: white;">
  <p>Model architecture</p>
</div>

So questions are raised from here:

- How about the hidden layer?
- How about the output layer?
- What is the dimension of weight matrix and also the input, hidden layer and output?

So I will give the step-by-step in details from the input to the output through the hidden layer and how we can get the word embedding matrices from the weights matrix

- **Step 1**:  