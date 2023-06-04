---
layout: paper-note
title: "Captionize-it app with Pytorch and Flask"
description: Details of model architectures and web app
date: 2023-06-4

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
    - name: Dataset
    - name: Encoder-Decoder architecture
    - name: Image captioning architecture
  - name: Dataset
    subsections:
    - name: Flickr8k
  - name: The model architectture
    subsections:
    - name: Encoder
    - name: Decoder
  - name: The training phase
    subsections:
    - name: Default parameters
    - name: Utilize vast.ai GPU
  - name: The application of Flask and UI using HTML and CSS
    subsections:
    - name: Flask
    - name: HTML and CSS
  - name: Conclusion
  - name: References

---
## Takeaways

- Understanding the encoder-decoder deep learning model.
- Understanding and utilizing training techniques.
- Deploying app using Flask.
- Basic HTML and CSS.


## Introduction

### Dataset
- The **Flickr8k dataset** consists of a diverse collection of **8,000 images**, each accompanied by five different textual captions. These captions are written by human annotators to describe the content and context of the corresponding image.

- Link to the dataset: **[dataset link](https://www.kaggle.com/datasets/adityajn105/flickr8k)**

### Encoder-Decoder architecture
- This architecture has another name as **sequence-to-sequence model**. This model can solve variety of tasks such as machine translation, image captioning, and also generative model like cycleGAN...

- There are 3 main blocks in the encoder-decoder architecture: **encoder**, **hidden vector**, **decoder**.
  - The **encoder** will convert the input sequence to into single-dimensional vector
  - The **decoder** will take this vector as input, and then generate the output 

<div class="l-body" style="text-align:center;">
  <img src="https://miro.medium.com/v2/resize:fit:515/0*bM9oRET5AGEdpaRv.png" width="50%" style="margin-bottom: 12px; background-color: white;">
  <p>Figure 1: encoder-decoder model architecture</p>
</div>

### This project
- We aim to develop an image captioning app using PyTorch and Flask by implementing an encoder-decoder model. The app will leverage a pre-trained model for the encoder component, while the decoder part will be trained from scratch. To enhance UI and accessibility, we will integrate a web interface.

## Dataset and how to create vocab
- First, in the **vocab.py** file, we define the **Vocabulary** class that handles word-to-index and index-to-word mappings.

```bash
class Vocabulary():
    def __init__(self, freq_threshold):
        self.freq_threshold = freq_threshold
        self.itos = {0: "<PAD>", 1:"<SOS>", 2:"<EOS>", 3:"<UNK>"}
        self.stoi = {"<PAD>":0, "<SOS>":1, "<EOS>":2, "<UNK>":3}
```
- We also use **nltk.word_tokenize** for tokenizing sentence.
- If a word's frequency reaches the specified **freq_threshold**, it is added to the vocabulary.
- Second, To make the **vocab.json**, simply do like this

```bash
vocab_file = "vocab.json"
with open(vocab_file, "w") as f:
    json.dump(vocab.stoi, f)
```
- Finally, the **get_loader()** will return the loader and dataset

## The model architecture

### Inception V3 for the encoder
- An **Inception Module** is an image model block that aims to approximate an optimal local sparse structure in a CNN using different types of filter sizes. 
- Back to previous models, we have GoogleNet as **Inception V1** with **factorization technique**, then Google introduce update **Inception V2** with the techniques of **factorization intro symmetric concolutions**. This project, we will use utilize the **Inception V3** with the update of **promoting high dimensional representations**.

<div class="l-body" style="text-align:center;">
  <img src="https://miro.medium.com/v2/resize:fit:828/format:webp/1*gqKM5V-uo2sMFFPDS84yJw.png" width="80%" style="margin-bottom: 12px; background-color: white;">
  <p>Figure 2: Inception V3 architecture</p>
</div>

- Using the pretrained models from Pytorch, the **features vector** is obtained. After, we will design a **fully-connected layers** to map the vector size to desired size before feeding it to the decoder

```bash
class Encoder(nn.Module):
    def __init__(self, embed_size, trainEncoder=False):
        super(Encoder, self).__init__()
        self.trainEncoder = trainEncoder
        self.inception = models.inception_v3(weights="DEFAULT")
        self.inception.fc = nn.Linear(self.inception.fc.in_features, embed_size)
        self.relu = nn.ReLU()
        self.times = []
        self.dropout = nn.Dropout(0.5)
```
### LSTM for the decoder

- **LSTM** stands for **long short-term memory networks**. It is a variety of recurrent neural networks (RNNs) that are capable of learning long-term dependencies in sequence data. The problem with **vanilla RNN** is that it has **long term dependency** and **gradient vanishing** problems. The update of **forget gate, input gate and output gate** of LSTM make it an improvement over vanilla RNN because it can control the memory flow more effectively and somehow minimize the risk of gradient vanish.

<div class="l-body" style="text-align:center;">
  <img src="https://databasecamp.de/wp-content/uploads/lstm-architecture-1024x709.png" width="60%" style="margin-bottom: 12px; background-color: white;">
  <p>Figure 3: LSTM architecture</p>
</div>

```bash
class Decoder(nn.Module):
    def __init__(self, embed_size, hidden_size, vocab_size, num_layers):
        super(Decoder, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embed_size)
        self.lstm = nn.LSTM(embed_size, hidden_size, num_layers)
        self.linear = nn.Linear(hidden_size, vocab_size)
        self.dropout = nn.Dropout(0.5)
```

**So questions are raised from here:**

- The relation of hidden state, output, and cell state in the training process?
- How we calculate the loss?

So I will give the step-by-step in details from the features vector to the output of each time step, and how we can update the parameters.

**1) Let's derive how we feed the first vector to decoder:**
- **The features vector** after going through **FC layers** will be concatenated with an initializing embedded **START** token.
- **The concatenated vector** then will be feeded into the LSTM layer, then receive hidden_states and cell_states for each timesteps.
- **Output** of each timestep will be calculated by feeding **hidden state** into linear layers to match the vocab size.
- **Loss** will be calculated.

**2) Calculate the loss using cross-entropy loss**
- For each timestep, we will get the **output word** and **ground-truth word**. The cross-entropy loss will be applied

**3) Teacher-forcing technique**
- Teacher forcing is a method for quickly and efficiently training RNN models that use the **ground truth** from a prior time step as input.
- As we know that there will be a **hallucination/mismatch** between the training part and the inference part. Traditionally, the training should not use ground-truth, but here is the problem. If first timestep of decoder generate the wrong output, it will lead to the chain of wrong output, which makes the computation cost bigger. 
- We can also apply the **partial teacher-force**. That will reduce both mismatch between training and inference and computation cost.

### Training phase
- First, we have to specify out default parameters and transform

```bash
# Hyperparameters
embed_size = 256
hidden_size = 256
vocab_size = len(dataset.vocab)
num_layers = 1
learning_rate = 3e-4
num_epochs = 150
```
- Second, initialize model and loss

```bash
# Initialize model, loss etc
model = EncoderDecoder(embed_size, hidden_size, vocab_size, num_layers).to(device)
criterion = nn.CrossEntropyLoss(ignore_index=dataset.vocab.stoi["<PAD>"])
optimizer = optim.Adam(model.parameters(), lr=learning_rate)
```

- Lastly, the code for training with all teacher-force and cross-entropy loss above.

```bash
for epoch in range(num_epochs):
    for idx, (imgs, captions) in tqdm(enumerate(train_loader), total=len(train_loader), leave=False):
        imgs, captions = imgs.to(device), captions.to(device)
        # Zero the gradient
        optimizer.zero_grad()
        # Feed forward
        outputs = model.forward(imgs, captions[:-1])
        # Calculate batch loss
        target = captions.reshape(-1)
        predict = outputs.reshape(-1, outputs.shape[2])
        loss = criterion(predict, target)
        # Write to tensorboard
        writer.add_scalar("Training loss", loss.item(), global_step=step)
        # Update step
        step += 1
        #Eval loss
        if step % print_every == 0:
            print("Epoch: {} loss: {:.5f}".format(epoch,loss.item()))
            # Generate the caption
            model.eval()
            print_examples(model, device, dataset)
            # Back to train mode
            model.train()
        loss.backward(loss)
        optimizer.step()
        if step % 5000 == 0:
            save_model(model, optimizer, step)
```
## The application of Flask and UI using HTML and CSS
### The inference
- The **inference()** function will be utilized to make predicted captions of input images. It takes the path of an input image, the path of a trained model checkpoint, the target device for inference (CPU or GPU), and the path to the vocabulary JSON file. 

- This inference() will be leveraged in the Flask app to enable users to upload images and receive corresponding captions as output.

### Flask
- The Flask application defines the / route to handle both GET and POST requests. In the GET request, the user is presented with a simple web interface where they can upload an image. Upon submitting the image, the POST request is triggered. 

- The application checks if the uploaded file is valid and saves it to the designated upload folder. 

- Then, the captions, along with the image filename and path, are rendered back to the user interface for display.

### Result with UI

<!-- <img src="https://drive.google.com/uc?id=1T4Vy96unIIluEa4rID9JsZuDs0EzQuxw" width="120%" style="margin-bottom: 12px; background-color: white;">
<p>Figure 4: Result</p> -->

<div class="l-body" style="text-align:center;">
  <img src="https://drive.google.com/uc?id=1T4Vy96unIIluEa4rID9JsZuDs0EzQuxw" width="100%" style="margin-bottom: 12px; background-color: white;">
  <p>Figure 4: Result</p>
</div>

## Conclusion
- We have all went through all the steps to create an image caption app. That is absolutely interesting application of encoder-decoder architecture. 
- Understanding the model architectures under the hood by coding it from scratch makes us be more active in the next development of the app.
- Understand how to use Flask, HTML and CSS to make a webapp interface
- How to improve: 
  - Add cross attention
  - Package to docker
  - Deploy to cloud server

See you guys in the next post!

## References

- Image captioning paper: https://paperswithcode.com/task/image-captioning
- Flask: https://flask.palletsprojects.com/en/2.3.x/
- Encoder Decoder architecture: https://arxiv.org/abs/2110.15253
- How to use vast.ai GPU: https://vast.ai/faq





