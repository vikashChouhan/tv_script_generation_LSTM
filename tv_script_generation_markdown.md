
# TV Script Generation

In this project, we'll generate our own [Seinfeld](https://en.wikipedia.org/wiki/Seinfeld) TV scripts using RNNs.  we'll be using part of the [Seinfeld dataset](https://www.kaggle.com/thec03u5/seinfeld-chronicles#scripts.csv) of scripts from 9 seasons.  The Neural Network we'll build will generate a new ,"fake" TV script, based on patterns it recognizes in this training data.

## Get the Data

The data is already provided for us in `./data/Seinfeld_Scripts.txt` and we're encouraged to open that file and look at the text. 
>* As a first step, we'll load in this data and look at some samples. 
* Then, you'll be tasked with defining and training an RNN to generate  a new script!


# load in data



```python
import helper
data_dir = './data/Seinfeld_Scripts.txt'
text = helper.load_data(data_dir)
```

## Explore the Data



```python
view_line_range = (0, 20)


import numpy as np

print('Dataset Stats')
print('Roughly the number of unique words: {}'.format(len({word: None for word in text.split()})))

lines = text.split('\n')
print('Number of lines: {}'.format(len(lines)))
word_count_line = [len(line.split()) for line in lines]
print('Average number of words in each line: {}'.format(np.average(word_count_line)))

print()
print('The lines {} to {}:'.format(*view_line_range))
print('\n'.join(text.split('\n')[view_line_range[0]:view_line_range[1]]))
```

    Dataset Stats
    Roughly the number of unique words: 46367
    Number of lines: 109233
    Average number of words in each line: 5.544240293684143
    
    The lines 0 to 20:
    jerry: do you know what this is all about? do you know, why were here? to be out, this is out...and out is one of the single most enjoyable experiences of life. people...did you ever hear people talking about we should go out? this is what theyre talking about...this whole thing, were all out now, no one is home. not one person here is home, were all out! there are people trying to find us, they dont know where we are. (on an imaginary phone) did you ring?, i cant find him. where did he go? he didnt tell me where he was going. he must have gone out. you wanna go out you get ready, you pick out the clothes, right? you take the shower, you get all ready, get the cash, get your friends, the car, the spot, the reservation...then youre standing around, what do you do? you go we gotta be getting back. once youre out, you wanna get back! you wanna go to sleep, you wanna get up, you wanna go out again tomorrow, right? where ever you are in life, its my feeling, youve gotta go. 
    
    jerry: (pointing at georges shirt) see, to me, that button is in the worst possible spot. the second button literally makes or breaks the shirt, look at it. its too high! its in no-mans-land. you look like you live with your mother. 
    
    george: are you through? 
    
    jerry: you do of course try on, when you buy? 
    
    george: yes, it was purple, i liked it, i dont actually recall considering the buttons. 
    
    jerry: oh, you dont recall? 
    
    george: (on an imaginary microphone) uh, no, not at this time. 
    
    jerry: well, senator, id just like to know, what you knew and when you knew it. 
    
    claire: mr. seinfeld. mr. costanza. 
    
    george: are, are you sure this is decaf? wheres the orange indicator? 
    
    

---
## Implement Pre-processing Functions
The first thing to do to any dataset is pre-processing.  Implement the following pre-processing functions below:
- Lookup Table
- Tokenize Punctuation

### Lookup Table
To create a word embedding, we first need to transform the words to ids.  In this function, we create two dictionaries:
- Dictionary to go from the words to an id, we'll call `vocab_to_int`
- Dictionary to go from the id to word, we'll call `int_to_vocab`

Return these dictionaries in the following **tuple** `(vocab_to_int, int_to_vocab)`


```python
import problem_unittests as tests

def create_lookup_tables(text):
    """
    Create lookup tables for vocabulary
    :param text: The text of tv scripts split into words
    :return: A tuple of dicts (vocab_to_int, int_to_vocab)
    """
    
    # TODO: Implement Function
    vocab_to_int = {word:ii for ii,word in enumerate(set(text),1)}
    
    int_to_vocab = dict(zip(vocab_to_int.values(),vocab_to_int.keys()))
    
    # return tuple
    return (vocab_to_int, int_to_vocab)


tests.test_create_lookup_tables(create_lookup_tables)
```

    Tests Passed
    

### Tokenize Punctuation
We'll be splitting the script into a word array using spaces as delimiters.  However, punctuations like periods and exclamation marks can create multiple ids for the same word. For example, "bye" and "bye!" would generate two different word ids.

We'll implement the function `token_lookup` to return a dict that will be used to tokenize symbols like "!" into "||Exclamation_Mark||".  We'll create a dictionary for the following symbols where the symbol is the key and value is the token:
- Period ( **.** )
- Comma ( **,** )
- Quotation Mark ( **"** )
- Semicolon ( **;** )
- Exclamation mark ( **!** )
- Question mark ( **?** )
- Left Parentheses ( **(** )
- Right Parentheses ( **)** )
- Dash ( **-** )
- Return ( **\n** )

This dictionary will be used to tokenize the symbols and add the delimiter (space) around it.  This separates each symbols as its own word, making it easier for the neural network to predict the next word. Make sure you don't use a value that could be confused as a word; for example, instead of using the value "dash", try using something like "||dash||".


```python
def token_lookup():
    """
    Generate a dict to turn punctuation into a token.
    :return: Tokenized dictionary where the key is the punctuation and the value is the token
    """
    # TODO: Implement Function
    symbol_to_name = {'.':'Period',
                      ',':'Comma',
                      '"':'Quotation_Mark',
                      ';':'Semicolon',
                      '!':'Exclamation_Mark',
                      '?':'Question_Mark',
                      '(':'Left_Parentheses',
                      ')':'Right_Parentheses',
                      '-':'Dash',
                      '\n':'Return'}
    
    return symbol_to_name


tests.test_tokenize(token_lookup)
```

    Tests Passed
    

## Pre-process all the data and save it

Running the code cell below will pre-process all the data and save it to file.


```python

# pre-process training data
helper.preprocess_and_save_data(data_dir, token_lookup, create_lookup_tables)
```

# Check Point
This is our first checkpoint. The preprocessed data has been saved to disk.


```python

import helper
import problem_unittests as tests

int_text, vocab_to_int, int_to_vocab, token_dict = helper.load_preprocess()


```

## Build the Neural Network
In this section, we'll build the components necessary to build an RNN by implementing the RNN Module and forward and backpropagation functions.

### Check Access to GPU


```python

import torch

# Check for a GPU
train_on_gpu = torch.cuda.is_available()

if not train_on_gpu:
    print('No GPU found. Please use a GPU to train your neural network.')
```

## Input
Let's start with the preprocessed input data. We'll use [TensorDataset](http://pytorch.org/docs/master/data.html#torch.utils.data.TensorDataset) to provide a known format to our dataset; in combination with [DataLoader](http://pytorch.org/docs/master/data.html#torch.utils.data.DataLoader), it will handle batching, shuffling, and other dataset iteration functions.

We can create data with TensorDataset by passing in feature and target tensors. Then create a DataLoader as usual.
```
data = TensorDataset(feature_tensors, target_tensors)
data_loader = torch.utils.data.DataLoader(data, 
                                          batch_size=batch_size)
```

### Batching
Implement the `batch_data` function to batch `words` data into chunks of size `batch_size` using the `TensorDataset` and `DataLoader` classes.

>We can batch words using the DataLoader, but it will be up to you to create `feature_tensors` and `target_tensors` of the correct size and content for a given `sequence_length`.

For example, say we have these as input:
```
words = [1, 2, 3, 4, 5, 6, 7]
sequence_length = 4
```

Your first `feature_tensor` should contain the values:
```
[1, 2, 3, 4]
```
And the corresponding `target_tensor` should just be the next "word"/tokenized word value:
```
5
```
This should continue with the second `feature_tensor`, `target_tensor` being:
```
[2, 3, 4, 5]  # features
6             # target
```


```python
from torch.utils.data import TensorDataset, DataLoader


def batch_data(words, sequence_length, batch_size):
    """
    Batch the neural network data using DataLoader
    :param words: The word ids of the TV scripts
    :param sequence_length: The sequence length of each batch
    :param batch_size: The size of each batch; the number of sequences in a batch
    :return: DataLoader with batched data
    """
    # TODO: Implement function
    
    x = np.zeros((len(words)-sequence_length,sequence_length))
    y = np.zeros(len(words)-sequence_length)
    
    for ii, batch in enumerate(range(len(words)-sequence_length)):
      x[ii,:] = words[batch:batch+sequence_length]
      y[ii] = words[batch+sequence_length]
    
    
    
    x,y = torch.from_numpy(x),torch.from_numpy(y)
    
    features_train = TensorDataset(x,y)
    dataloader = DataLoader(features_train,batch_size=batch_size,shuffle=True)
   
    # return a dataloader
    return dataloader
    

# there is no test for this function, but you are encouraged to create
# print statements and tests of your own

```

### Test your dataloader 

We'll have to modify this code to test a batching function, but it should look fairly similar.

Below, we're generating some test text data and defining a dataloader using the function you defined, above. Then, we are getting some sample batch of inputs `sample_x` and targets `sample_y` from our dataloader.

Our code should return something like the following (likely in a different order, if you shuffled your data):

```
torch.Size([10, 5])
tensor([[ 28,  29,  30,  31,  32],
        [ 21,  22,  23,  24,  25],
        [ 17,  18,  19,  20,  21],
        [ 34,  35,  36,  37,  38],
        [ 11,  12,  13,  14,  15],
        [ 23,  24,  25,  26,  27],
        [  6,   7,   8,   9,  10],
        [ 38,  39,  40,  41,  42],
        [ 25,  26,  27,  28,  29],
        [  7,   8,   9,  10,  11]])

torch.Size([10])
tensor([ 33,  26,  22,  39,  16,  28,  11,  43,  30,  12])
```

### Sizes
Our sample_x should be of size `(batch_size, sequence_length)` or (10, 5) in this case and sample_y should just have one dimension: batch_size (10). 

### Values

We should also notice that the targets, sample_y, are the *next* value in the ordered test_text data. So, for an input sequence `[ 28,  29,  30,  31,  32]` that ends with the value `32`, the corresponding output should be `33`.


```python
# test dataloader

test_text = range(50)
t_loader = batch_data(test_text, sequence_length=5, batch_size=10)

data_iter = iter(t_loader)

sample_x, sample_y = data_iter.next()

#for sample_x,sample_y in t_loader:
print(sample_x.shape)
print(sample_x)
print()
print(sample_y.shape)
print(sample_y)
```

    torch.Size([10, 5])
    tensor([[20., 21., 22., 23., 24.],
            [19., 20., 21., 22., 23.],
            [38., 39., 40., 41., 42.],
            [ 9., 10., 11., 12., 13.],
            [42., 43., 44., 45., 46.],
            [29., 30., 31., 32., 33.],
            [27., 28., 29., 30., 31.],
            [18., 19., 20., 21., 22.],
            [ 2.,  3.,  4.,  5.,  6.],
            [41., 42., 43., 44., 45.]], dtype=torch.float64)
    
    torch.Size([10])
    tensor([25., 24., 43., 14., 47., 34., 32., 23.,  7., 46.], dtype=torch.float64)
    

---
## Build the Neural Network
Implement an RNN using PyTorch's [Module class](http://pytorch.org/docs/master/nn.html#torch.nn.Module). We may choose to use a GRU or an LSTM. To complete the RNN, we'll have to implement the following functions for the class:
 - `__init__` - The initialize function. 
 - `init_hidden` - The initialization function for an LSTM/GRU hidden state
 - `forward` - Forward propagation function.
 
The initialize function should create the layers of the neural network and save them to the class. The forward propagation function will use these layers to run forward propagation and generate an output and a hidden state.

**The output of this model should be the *last* batch of word scores** after a complete sequence has been processed. That is, for each input sequence of words, we only want to output the word scores for a single, most likely, next word.

### We will keep an eye on these

1. Make sure to stack the outputs of the lstm to pass to your fully-connected layer, you can do this with `lstm_output = lstm_output.contiguous().view(-1, self.hidden_dim)`
2. We can get the last batch of word scores by shaping the output of the final, fully-connected layer like so:

```
# reshape into (batch_size, seq_length, output_size)
output = output.view(batch_size, -1, self.output_size)
# get last batch
out = output[:, -1]
```


```python
import torch.nn as nn

class RNN(nn.Module):
    
    def __init__(self, vocab_size, output_size, embedding_dim, hidden_dim, n_layers, dropout=0.5):
        """
        Initialize the PyTorch RNN Module
        :param vocab_size: The number of input dimensions of the neural network (the size of the vocabulary)
        :param output_size: The number of output dimensions of the neural network
        :param embedding_dim: The size of embeddings, should you choose to use them        
        :param hidden_dim: The size of the hidden layer outputs
        :param dropout: dropout to add in between LSTM/GRU layers
        """
        super(RNN, self).__init__()
        # TODO: Implement function
        
        # set class variables
        self.vocab_size = vocab_size
        self.output_size = output_size
        self.hidden_dim = hidden_dim
        self.embed_dim = embedding_dim
        self.n_layers = n_layers
        
        
        # define model layers
        self.embed = nn.Embedding(vocab_size,self.embed_dim)
        #initializing weights of embedding layer
        #self.embed.weight.data.normal_(mean=0,std=0.001)
        
        self.lstm = nn.LSTM(self.embed_dim,self.hidden_dim,self.n_layers,dropout=dropout,batch_first=True)
        
        self.drpout = nn.Dropout(p=0.6)
        self.fc = nn.Linear(self.hidden_dim,self.output_size)
        
    
    def forward(self, nn_input, hidden):
        """
        Forward propagation of the neural network
        :param nn_input: The input to the neural network
        :param hidden: The hidden state        
        :return: Two Tensors, the output of the neural network and the latest hidden state
        """
        # TODO: Implement function   
        batch_size = nn_input.size(0)
        
        #print(nn_input.shape)
        nn_output = self.embed(nn_input.long())
        
        #print(nn_output.shape)
        nn_output,hidden_t = self.lstm(nn_output,hidden)
        
        #print(nn_output.shape)
        #nn_output = self.drpout(nn_output)
        
        
        nn_output = nn_output.contiguous().view(-1,self.hidden_dim)
        #print(nn_output.shape)
        
        nn_output = self.fc(nn_output)
        #print(nn_output.shape)
        
        nn_output = nn_output.reshape(batch_size,-1,self.output_size)
        #print(nn_output.shape)
        output = nn_output[:,-1]
        
        #print(nn_input.shape,output.shape,hidden[0].shape)
        #print(self.hidden_dim,self.vocab_size,self.embed_dim,self.n_layers)
        
        # return one batch of output word scores and the hidden state
        return output,hidden_t
    
    
    def init_hidden(self, batch_size):
        '''
        Initialize the hidden state of an LSTM/GRU
        :param batch_size: The batch_size of the hidden state
        :return: hidden state of dims (n_layers, batch_size, hidden_dim)
        '''
        # Implement function
        
        # initialize hidden state with zero weights, and move to GPU if available
        weight = next(self.parameters()).data
        
        if (train_on_gpu):
            hidden = (weight.new(self.n_layers, batch_size, self.hidden_dim).zero_().cuda(),
                  weight.new(self.n_layers, batch_size, self.hidden_dim).zero_().cuda())
        else:
            hidden = (weight.new(self.n_layers, batch_size, self.hidden_dim).zero_(),
                      weight.new(self.n_layers, batch_size, self.hidden_dim).zero_())


        return hidden


tests.test_rnn(RNN, train_on_gpu)
```

    Tests Passed
    

### Define forward and backpropagation

Use the RNN class you implemented to apply forward and back propagation. This function will be called, iteratively, in the training loop as follows:
```
loss = forward_back_prop(decoder, decoder_optimizer, criterion, inp, target)
```

And it should return the average loss over a batch and the hidden state returned by a call to `RNN(inp, hidden)`. Recall that you can get this loss by computing it, as usual, and calling `loss.item()`.

**If a GPU is available, you should move your data to that GPU device, here.**


```python
def forward_back_prop(rnn, optimizer, criterion, inp, target, hidden):
    """
    Forward and backward propagation on the neural network
    :param rnn: The PyTorch Module that holds the neural network
    :param optimizer: The PyTorch optimizer for the neural network
    :param criterion: The PyTorch loss function
    :param inp: A batch of input to the neural network
    :param target: The target output for the batch of input
    :return: The loss and the latest hidden state Tensor
    """
    
    # TODO: Implement Function
    
    # move data to GPU, if available
    if(train_on_gpu):
      inp,target = inp.cuda(),target.cuda()
      
     
      
    #print(inp.dtype,hidden[0].dtype)
    optimizer.zero_grad()
    
    output,hidden = rnn(inp,hidden)
    hidden = tuple([h.detach() for h in hidden])
    #print('input=',inp.device,'output=',output.device,'hidden=', hidden[0].device)
    
    #print(output.dtype,target.dtype)
    #print('input - ',inp.shape)
    #print('target - ',target.shape)
    #print('output - ',output.shape)
    loss = criterion(output,target.long())
    #print('loss=',loss.device)
    
    
    #print(output.type(),hidden[0].dtype,target.dtype)
    
    # perform backpropagation and optimization
    loss.backward()
    nn.utils.clip_grad_norm_(rnn.parameters(), 10)
    optimizer.step()

    # return the loss over a batch and the hidden state produced by our model
    return loss.item(),hidden

# Note that these tests aren't completely extensive.
# they are here to act as general checks on the expected outputs of your functions

tests.test_forward_back_prop(RNN, forward_back_prop, train_on_gpu)
```

    Tests Passed
    

## Neural Network Training

With the structure of the network complete and data ready to be fed in the neural network, it's time to train it.

### Train Loop

The training loop is implemented  in the `train_decoder` function. This function will train the network over all the batches for the number of epochs given. The model progress will be shown every number of batches. This number is set with the `show_every_n_batches` parameter. You'll set this parameter along with other parameters in the next section.


```python

def train_rnn(rnn, batch_size, optimizer, criterion, n_epochs, show_every_n_batches=100):
    batch_losses = []
    
    rnn.train()
    minLoss = np.inf
    print("Training for %d epoch(s)..." % n_epochs)
    
    
    
    for epoch_i in range(1, n_epochs + 1):
        
        # initialize hidden state
        hidden = rnn.init_hidden(batch_size)
        print()
        for batch_i, (inputs, labels) in enumerate(train_loader, 1):
            print(f'\rTraining : {batch_i}/{len(train_loader)}',end='')
            # make sure you iterate over completely full batches, only
            n_batches = len(train_loader.dataset)//batch_size
            if(batch_i > n_batches):
                break
            
            # forward, back prop
            loss, hidden = forward_back_prop(rnn, optimizer, criterion, inputs, labels, hidden)          
            # record loss
            batch_losses.append(loss)

            # printing loss stats
            if batch_i % show_every_n_batches == 0:
                print()
                avg_loss = np.average(batch_losses)
                print('Epoch: {:>4}/{:<4}  Loss: {}\n'.format(epoch_i, n_epochs, avg_loss))
             
                
                batch_losses = []
                  
                print()
        
       
    # returns a trained rnn
    return rnn
```

### Hyperparameters

Set and train the neural network with the following parameters:
- Set `sequence_length` to the length of a sequence.
- Set `batch_size` to the batch size.
- Set `num_epochs` to the number of epochs to train for.
- Set `learning_rate` to the learning rate for an Adam optimizer.
- Set `vocab_size` to the number of unique tokens in our vocabulary.
- Set `output_size` to the desired size of the output.
- Set `embedding_dim` to the embedding dimension; smaller than the vocab_size.
- Set `hidden_dim` to the hidden dimension of your RNN.
- Set `n_layers` to the number of layers/cells in your RNN.
- Set `show_every_n_batches` to the number of batches at which the neural network should print progress.

If the network isn't getting the desired results, tweak these parameters and/or the layers in the `RNN` class.


```python
# Data params
# Sequence Length
sequence_length = 16
# of words in a sequence
# Batch Size

batch_size = 128

# data loader - do not change
train_loader = batch_data(int_text, sequence_length, batch_size)

```


```python
# Training parameters
# Number of Epochs
num_epochs = 10
# Learning Rate
learning_rate = 0.0005

# Model parameters
# Vocab size
vocab_size = len(vocab_to_int)+1

# Output size
output_size = vocab_size
# Embedding Dimension
embedding_dim = 300
# Hidden Dimension
hidden_dim = 512
# Number of RNN Layers
n_layers = 2

# Show stats for every n number of batches
show_every_n_batches = 500

```

### Train
In the next cell, we'll train the neural network on the pre-processed data.  If we have a hard time getting a good loss, we may consider changing your hyperparameters. In general, you may get better results with larger hidden and n_layer dimensions, but larger models take a longer time to train. 
> **We should aim for a loss less than 3.5.** 

We should also experiment with different sequence lengths, which determine the size of the long range dependencies that a model can learn.


```python

# create model and move to gpu if available
rnn = RNN(vocab_size, output_size, embedding_dim, hidden_dim, n_layers, dropout=0.5)

#rnn.load_state_dict(torch.load('rnn_model.pt'))
if train_on_gpu:
    rnn.cuda()

# defining loss and optimization functions for training
optimizer = torch.optim.Adam(rnn.parameters(), lr=learning_rate)
criterion = nn.CrossEntropyLoss()

# training the model

trained_rnn = train_rnn(rnn, batch_size, optimizer, criterion, num_epochs, show_every_n_batches)

# saving the trained model
helper.save_model('./save/trained_rnn1', trained_rnn)
print('Model Trained and Saved')
```

    Training for 10 epoch(s)...
    
    Training : 500/6970
    Epoch:    1/10    Loss: 5.443206458091736
    
    
    Training : 1000/6970
    Epoch:    1/10    Loss: 4.848888091087341
    
    
    Training : 1500/6970
    Epoch:    1/10    Loss: 4.606449417114257
    
    
    Training : 2000/6970
    Epoch:    1/10    Loss: 4.486605880737304
    
    
    Training : 2500/6970
    Epoch:    1/10    Loss: 4.38608249092102
    
    
    Training : 3000/6970
    Epoch:    1/10    Loss: 4.33639565372467
    
    
    Training : 3500/6970
    Epoch:    1/10    Loss: 4.284657090187073
    
    
    Training : 4000/6970
    Epoch:    1/10    Loss: 4.252015771389008
    
    
    Training : 4500/6970
    Epoch:    1/10    Loss: 4.225338705539704
    
    
    Training : 5000/6970
    Epoch:    1/10    Loss: 4.193158923625946
    
    
    Training : 5500/6970
    Epoch:    1/10    Loss: 4.141554294109344
    
    
    Training : 6000/6970
    Epoch:    1/10    Loss: 4.097915284633636
    
    
    Training : 6500/6970
    Epoch:    1/10    Loss: 4.099655900001526
    
    
    Training : 6970/6970
    Training : 500/6970
    Epoch:    2/10    Loss: 4.004449606433864
    
    
    Training : 1000/6970
    Epoch:    2/10    Loss: 3.9022494673728945
    
    
    Training : 1500/6970
    Epoch:    2/10    Loss: 3.8768889503479005
    
    
    Training : 2000/6970
    Epoch:    2/10    Loss: 3.871685335159302
    
    
    Training : 2500/6970
    Epoch:    2/10    Loss: 3.8695617833137512
    
    
    Training : 3000/6970
    Epoch:    2/10    Loss: 3.85710277223587
    
    
    Training : 3500/6970
    Epoch:    2/10    Loss: 3.8576703271865846
    
    
    Training : 4000/6970
    Epoch:    2/10    Loss: 3.842003134250641
    
    
    Training : 4500/6970
    Epoch:    2/10    Loss: 3.8383171529769897
    
    
    Training : 5000/6970
    Epoch:    2/10    Loss: 3.855919692516327
    
    
    Training : 5500/6970
    Epoch:    2/10    Loss: 3.8416240401268005
    
    
    Training : 6000/6970
    Epoch:    2/10    Loss: 3.8396801595687866
    
    
    Training : 6500/6970
    Epoch:    2/10    Loss: 3.808434310913086
    
    
    Training : 6970/6970
    Training : 500/6970
    Epoch:    3/10    Loss: 3.724915117179154
    
    
    Training : 1000/6970
    Epoch:    3/10    Loss: 3.6291332688331606
    
    
    Training : 1500/6970
    Epoch:    3/10    Loss: 3.630673463344574
    
    
    Training : 2000/6970
    Epoch:    3/10    Loss: 3.6252119898796082
    
    
    Training : 2500/6970
    Epoch:    3/10    Loss: 3.61202201461792
    
    
    Training : 3000/6970
    Epoch:    3/10    Loss: 3.6409706892967226
    
    
    Training : 3500/6970
    Epoch:    3/10    Loss: 3.6535669422149657
    
    
    Training : 4000/6970
    Epoch:    3/10    Loss: 3.6744433312416076
    
    
    Training : 4500/6970
    Epoch:    3/10    Loss: 3.633676655769348
    
    
    Training : 5000/6970
    Epoch:    3/10    Loss: 3.629742596626282
    
    
    Training : 5500/6970
    Epoch:    3/10    Loss: 3.638074182033539
    
    
    Training : 6000/6970
    Epoch:    3/10    Loss: 3.644261281490326
    
    
    Training : 6500/6970
    Epoch:    3/10    Loss: 3.6565587577819825
    
    
    Training : 6970/6970
    Training : 500/6970
    Epoch:    4/10    Loss: 3.535166019010593
    
    
    Training : 1000/6970
    Epoch:    4/10    Loss: 3.4327659134864805
    
    
    Training : 1500/6970
    Epoch:    4/10    Loss: 3.4542546830177305
    
    
    Training : 2000/6970
    Epoch:    4/10    Loss: 3.4544861354827883
    
    
    Training : 2500/6970
    Epoch:    4/10    Loss: 3.4795235228538512
    
    
    Training : 3000/6970
    Epoch:    4/10    Loss: 3.4586186752319334
    
    
    Training : 3500/6970
    Epoch:    4/10    Loss: 3.4838649849891663
    
    
    Training : 4000/6970
    Epoch:    4/10    Loss: 3.4738333144187927
    
    
    Training : 4500/6970
    Epoch:    4/10    Loss: 3.497898551940918
    
    
    Training : 5000/6970
    Epoch:    4/10    Loss: 3.503476919174194
    
    
    Training : 5500/6970
    Epoch:    4/10    Loss: 3.4977535648345945
    
    
    Training : 6000/6970
    Epoch:    4/10    Loss: 3.49973184633255
    
    
    Training : 6500/6970
    Epoch:    4/10    Loss: 3.506080183029175
    
    
    Training : 6970/6970
    Training : 500/6970
    Epoch:    5/10    Loss: 3.383412648034662
    
    
    Training : 1000/6970
    Epoch:    5/10    Loss: 3.3167228679656984
    
    
    Training : 1500/6970
    Epoch:    5/10    Loss: 3.2879762868881226
    
    
    Training : 2000/6970
    Epoch:    5/10    Loss: 3.325356276512146
    
    
    Training : 2500/6970
    Epoch:    5/10    Loss: 3.3336277313232423
    
    
    Training : 3000/6970
    Epoch:    5/10    Loss: 3.34119540643692
    
    
    Training : 3500/6970
    Epoch:    5/10    Loss: 3.346028685569763
    
    
    Training : 4000/6970
    Epoch:    5/10    Loss: 3.3552881927490232
    
    
    Training : 4500/6970
    Epoch:    5/10    Loss: 3.3672088952064514
    
    
    Training : 5000/6970
    Epoch:    5/10    Loss: 3.372416317462921
    
    
    Training : 5500/6970
    Epoch:    5/10    Loss: 3.35188969707489
    
    
    Training : 6000/6970
    Epoch:    5/10    Loss: 3.374532573223114
    
    
    Training : 6500/6970
    Epoch:    5/10    Loss: 3.386392236709595
    
    
    Training : 6970/6970
    Training : 500/6970
    Epoch:    6/10    Loss: 3.2781247871209964
    
    
    Training : 1000/6970
    Epoch:    6/10    Loss: 3.18343750333786
    
    
    Training : 1500/6970
    Epoch:    6/10    Loss: 3.1789568667411805
    
    
    Training : 2000/6970
    Epoch:    6/10    Loss: 3.196244062900543
    
    
    Training : 2500/6970
    Epoch:    6/10    Loss: 3.208189133644104
    
    
    Training : 3000/6970
    Epoch:    6/10    Loss: 3.2090008721351624
    
    
    Training : 3500/6970
    Epoch:    6/10    Loss: 3.229842489719391
    
    
    Training : 4000/6970
    Epoch:    6/10    Loss: 3.2405106053352357
    
    
    Training : 4500/6970
    Epoch:    6/10    Loss: 3.250214470386505
    
    
    Training : 5000/6970
    Epoch:    6/10    Loss: 3.2495397238731383
    
    
    Training : 5500/6970
    Epoch:    6/10    Loss: 3.2850586552619934
    
    
    Training : 6000/6970
    Epoch:    6/10    Loss: 3.2836332902908327
    
    
    Training : 6500/6970
    Epoch:    6/10    Loss: 3.2776124992370606
    
    
    Training : 6970/6970
    Training : 500/6970
    Epoch:    7/10    Loss: 3.16419457971004
    
    
    Training : 1000/6970
    Epoch:    7/10    Loss: 3.0985522465705873
    
    
    Training : 1500/6970
    Epoch:    7/10    Loss: 3.0686220088005065
    
    
    Training : 2000/6970
    Epoch:    7/10    Loss: 3.0918169078826905
    
    
    Training : 2500/6970
    Epoch:    7/10    Loss: 3.1416888055801393
    
    
    Training : 3000/6970
    Epoch:    7/10    Loss: 3.117724885940552
    
    
    Training : 3500/6970
    Epoch:    7/10    Loss: 3.13076327419281
    
    
    Training : 4000/6970
    Epoch:    7/10    Loss: 3.1632449049949645
    
    
    Training : 4500/6970
    Epoch:    7/10    Loss: 3.185392683506012
    
    
    Training : 5000/6970
    Epoch:    7/10    Loss: 3.1747667698860167
    
    
    Training : 5500/6970
    Epoch:    7/10    Loss: 3.1776210260391236
    
    
    Training : 6000/6970
    Epoch:    7/10    Loss: 3.1918694615364074
    
    
    Training : 6500/6970
    Epoch:    7/10    Loss: 3.1986109404563905
    
    
    Training : 6970/6970
    Training : 500/6970
    Epoch:    8/10    Loss: 3.097798932078453
    
    
    Training : 1000/6970
    Epoch:    8/10    Loss: 2.9873950128555298
    
    
    Training : 1500/6970
    Epoch:    8/10    Loss: 3.0092983527183534
    
    
    Training : 2000/6970
    Epoch:    8/10    Loss: 3.0358306398391726
    
    
    Training : 2500/6970
    Epoch:    8/10    Loss: 3.035121546268463
    
    
    Training : 3000/6970
    Epoch:    8/10    Loss: 3.028081542491913
    
    
    Training : 3500/6970
    Epoch:    8/10    Loss: 3.0578246932029725
    
    
    Training : 4000/6970
    Epoch:    8/10    Loss: 3.084838059425354
    
    
    Training : 4500/6970
    Epoch:    8/10    Loss: 3.0848916759490965
    
    
    Training : 5000/6970
    Epoch:    8/10    Loss: 3.085495857715607
    
    
    Training : 5500/6970
    Epoch:    8/10    Loss: 3.1023143072128296
    
    
    Training : 6000/6970
    Epoch:    8/10    Loss: 3.1183663454055788
    
    
    Training : 6500/6970
    Epoch:    8/10    Loss: 3.1416805458068846
    
    
    Training : 6970/6970
    Training : 500/6970
    Epoch:    9/10    Loss: 3.0185953680083726
    
    
    Training : 1000/6970
    Epoch:    9/10    Loss: 2.931344449520111
    
    
    Training : 1500/6970
    Epoch:    9/10    Loss: 2.9287690024375914
    
    
    Training : 2000/6970
    Epoch:    9/10    Loss: 2.953030924797058
    
    
    Training : 2500/6970
    Epoch:    9/10    Loss: 2.983211884021759
    
    
    Training : 3000/6970
    Epoch:    9/10    Loss: 2.9818738465309145
    
    
    Training : 3500/6970
    Epoch:    9/10    Loss: 2.97459343624115
    
    
    Training : 4000/6970
    Epoch:    9/10    Loss: 2.9968323593139647
    
    
    Training : 4500/6970
    Epoch:    9/10    Loss: 3.025308382034302
    
    
    Training : 5000/6970
    Epoch:    9/10    Loss: 3.016828173160553
    
    
    Training : 5500/6970
    Epoch:    9/10    Loss: 3.0384772391319275
    
    
    Training : 6000/6970
    Epoch:    9/10    Loss: 3.051708044528961
    
    
    Training : 6500/6970
    Epoch:    9/10    Loss: 3.0556146121025085
    
    
    Training : 6970/6970
    Training : 500/6970
    Epoch:   10/10    Loss: 2.955558274921618
    
    
    Training : 1000/6970
    Epoch:   10/10    Loss: 2.853785945415497
    
    
    Training : 1500/6970
    Epoch:   10/10    Loss: 2.8701298627853395
    
    
    Training : 2000/6970
    Epoch:   10/10    Loss: 2.88236875295639
    
    
    Training : 2500/6970
    Epoch:   10/10    Loss: 2.9075113444328307
    
    
    Training : 3000/6970
    Epoch:   10/10    Loss: 2.9034260039329527
    
    
    Training : 3500/6970
    Epoch:   10/10    Loss: 2.936406665802002
    
    
    Training : 4000/6970
    Epoch:   10/10    Loss: 2.9442231674194335
    
    
    Training : 4500/6970
    Epoch:   10/10    Loss: 2.984495931148529
    
    
    Training : 5000/6970
    Epoch:   10/10    Loss: 2.9614505667686464
    
    
    Training : 5500/6970
    Epoch:   10/10    Loss: 2.983469502925873
    
    
    Training : 6000/6970
    Epoch:   10/10    Loss: 3.0023642978668215
    
    
    Training : 6500/6970
    Epoch:   10/10    Loss: 2.9920882778167726
    
    
    Training : 6970/6970

    /usr/local/lib/python3.6/dist-packages/torch/serialization.py:251: UserWarning: Couldn't retrieve source code for container of type RNN. It won't be checked for correctness upon loading.
      "type " + obj.__name__ + ". It won't be checked "
    

    Model Trained and Saved
    

### Question: How did you decide on your model hyperparameters? 
For example, did you try different sequence_lengths and find that one size made the model converge faster? What about your hidden_dim and n_layers; how did you decide on those?

**Answer:** 

I started setting model hyperparamters from low to high.
At first, I chose the following

batch_size = 8 </br>
sequence length = 10 (average length of a sentence is about 5)</br>
num_layers = 2 </br>
hidden_dim = 32 </br>
embedding dim = 64 </br>

These setting gave a rough idea about the loss, I looked for the trend in loss and then changed the parameters as required. 


**Setting sequence length** </br>
 I found that setting larger sequence length was making model heavy and difficult to train. No doubt, it might give better result, but many times model was too huge to train. I found setting sequence_length under **20** was a better choice. Since, average number of words in a sentence is 5, so, any number under 20 could give good results with faster training.

**Setting hidden dimension**</br>
I gradually increased hidden_dims from as low as 8 to high as 512 in the power of 2. I found that setting lower number was forcing model to remaing stuck at loss around 4 to 5, hence, might be causing underfiting . So, I increased hidden_dim to 128, although, it made better loss, but model was taking lot more time to converge, it need more epochs to get better results. Then, I shifted to hidden_dim of **512**, and found promising results with specific learning rate. I think 512 is big enough to train the model.

**Setting n_layers** </br>
Decideng number of layer was not that much difficult, just gave a try of 2 and 3 layers, that's it. Model with 3 layers was taking longer, So, I deicided to chose two layers. I think choosing only one layer could not be better choice, because we have lot of words to train, so, if we deicide to decrease number of layers, we might need to increase other parameters like sequence length, hidden_dim to prevent underfitting. 

**Setting learning rate**</br>
After setting all above paramters, I ran model with larger learning rate about 0.1 with simultaneously watching the trend. I decreased learning rate in log space, I found that larger learning rate like 0.003 (in this case) or 0.01  were performing better in beginning, but soon they failed to converge, and might be overshooting the mininum. Finally, I found **0.001 , 0.0005, 0.0003** could be better.

**Setting batch_size**</br>
Choosing larger batch size as 128 or even 64 with large embedding dim of 400 and hidden_dim of 512 seems to be promising in getting better results.






---
# Checkpoint

After running the above training cell, our model will be saved by name, `trained_rnn`, and if you save your notebook progress. Wou can resume our progress by running the next cell, which will load in our word:id dictionaries _and_ load in your saved model by name!


```python

import torch
import helper
import problem_unittests as tests

_, vocab_to_int, int_to_vocab, token_dict = helper.load_preprocess()
trained_rnn = helper.load_model('./save/trained_rnn1')
trained_rnn = RNN(vocab_size, output_size, embedding_dim, hidden_dim, n_layers, dropout=0)


trained_rnn  = trained_rnn.to('cuda')
```

## Generate TV Script
With the network trained and saved, we'll use it to generate a new, "fake" Seinfeld TV script in this section.

### Generate Text
To generate the text, the network needs to start with a single word and repeat its predictions until it reaches a set length. We'll be using the `generate` function to do this. It takes a word id to start with, `prime_id`, and generates a set length of text, `predict_len`. Also note that it uses topk sampling to introduce some randomness in choosing the most likely next word, given an output set of word scores!


```python

import torch.nn.functional as F

def generate(rnn, prime_id, int_to_vocab, token_dict, pad_value, predict_len=100):
    """
    Generate text using the neural network
    :param decoder: The PyTorch Module that holds the trained neural network
    :param prime_id: The word id to start the first prediction
    :param int_to_vocab: Dict of word id keys to word values
    :param token_dict: Dict of puncuation tokens keys to puncuation values
    :param pad_value: The value used to pad a sequence
    :param predict_len: The length of text to generate
    :return: The generated text
    """
    rnn.eval()
    
    # create a sequence (batch_size=1) with the prime_id
    current_seq = np.full((1, sequence_length), pad_value)
    current_seq[-1][-1] = prime_id
    predicted = [int_to_vocab[prime_id]]
    
    for _ in range(predict_len):
        if train_on_gpu:
            current_seq = torch.LongTensor(current_seq).cuda()
        else:
            current_seq = torch.LongTensor(current_seq)
        
        # initialize the hidden state
        hidden = rnn.init_hidden(current_seq.size(0))
        
        # get the output of the rnn
        output, _ = rnn(current_seq, hidden)
        
        # get the next word probabilities
        p = F.softmax(output, dim=1).data
        if(train_on_gpu):
            p = p.cpu() # move to cpu
         
        # use top_k sampling to get the index of the next word
        top_k = 5
        p, top_i = p.topk(top_k)
        top_i = top_i.numpy().squeeze()
        
        # select the likely next word index with some element of randomness
        p = p.numpy().squeeze()
        word_i = np.random.choice(top_i, p=p/p.sum())
        
        # retrieve that word from the dictionary
        word = int_to_vocab[word_i]
        predicted.append(word)     
        
        # the generated word becomes the next "current sequence" and the cycle can continue
        current_seq = np.roll(current_seq.cpu(), -1, 1)
        current_seq[-1][-1] = word_i
    
    gen_sentences = ' '.join(predicted)
    
    # Replace punctuation tokens
    for key, token in token_dict.items():
        ending = ' ' if key in ['\n', '(', '"'] else ''
        gen_sentences = gen_sentences.replace(' ' + token.lower(), key)
    gen_sentences = gen_sentences.replace('\n ', '\n')
    gen_sentences = gen_sentences.replace('( ', '(')
    
    # return all the sentences
    return gen_sentences
```

### Generate a New Script
It's time to generate the text. Set `gen_length` to the length of TV script you want to generate and set `prime_word` to one of the following to start the prediction:
- "jerry"
- "elaine"
- "george"
- "kramer"

We can set the prime word to _any word_ in our dictionary, but it's best to start with a name for generating a TV script. (We can also start with any other names we find in the original text file!)


```python
# run the cell multiple times to get different results!
gen_length = 400 # modify the length to your preference
prime_word = 'jerry' # name for starting the script


pad_word = helper.SPECIAL_WORDS['PADDING']
generated_script = generate(trained_rnn, vocab_to_int[prime_word + ':'], int_to_vocab, token_dict, vocab_to_int[pad_word], gen_length)
print(generated_script)
```

    jerry: spreads are crashing a lot like a person who goes to a hospital. you know, you know what? i mean, what if i don't have the apartment?
    
    elaine:(confused) what do you mean?
    
    jerry:(deadpan) oh, i think that's fantastic......
    
    kramer: well, i got a message to see that.
    
    jerry: oh, you don't know?
    
    george: i don't know.
    
    jerry: you know, i was thinking. i was thinking...
    
    george:(laughs) i thought you said hi, but you were supposed to do something.
    
    jerry: well, what about my father?
    
    elaine: well, i don't know what to do.
    
    jerry: you know i don't think so.
    
    elaine: i thought he was a very nice guy.
    
    jerry: so, what are you doing?
    
    kramer: oh, come on. come on.
    
    elaine: what?
    
    kramer: what, are you kidding? you were so nice of me.
    
    jerry: you know, i was thinking if i was gonna get a picture. i was in the pool...
    
    jerry: i thought you said you'd like to talk about the baby.
    
    elaine: well, i gotta get going.(kramer goes into jerry's room) hey! hey! you can't hear the truth!
    
    elaine:(shouting) hey!
    
    jerry: hey!
    
    elaine:(to the saleswoman) hey.
    
    elaine: hey!
    
    jerry: hey.
    
    elaine: hi julie.
    
    kramer: hi, jerry.
    
    jerry: hey.
    
    jerry: hey.
    
    jerry: hey!
    
    kramer: hey.
    
    george:(to jerry) hey, i gotta go to the bathroom and talk about the drakette.
    
    jerry: what?
    
    kramer: well, i got a date for the yankees i know.
    
    elaine: oh, i can't believe this.
    
    jerry: well
    

#### Save your favorite scripts

Once we have a script that we like (or find interesting), save it to a text file!


```python
# save script to a text file
f =  open("generated_script_1.txt","w")
f.write(generated_script)
f.close()
```

# The TV Script is Not Perfect
It's ok if the TV script doesn't make perfect sense. It should look like alternating lines of dialogue, here is one such example of a few generated lines.

### Example generated script

>jerry: what about me?
>
>jerry: i don't have to wait.
>
>kramer:(to the sales table)
>
>elaine:(to jerry) hey, look at this, i'm a good doctor.
>
>newman:(to elaine) you think i have no idea of this...
>
>elaine: oh, you better take the phone, and he was a little nervous.
>
>kramer:(to the phone) hey, hey, jerry, i don't want to be a little bit.(to kramer and jerry) you can't.
>
>jerry: oh, yeah. i don't even know, i know.
>
>jerry:(to the phone) oh, i know.
>
>kramer:(laughing) you know...(to jerry) you don't know.

We can see that there are multiple characters that say (somewhat) complete sentences, but it doesn't have to be perfect! It takes quite a while to get good results, and often, we'll have to use a smaller vocabulary (and discard uncommon words), or get more data.  The Seinfeld dataset is about 3.4 MB, which is big enough for our purposes; for script generation we'll want more than 1 MB of text, generally. 

