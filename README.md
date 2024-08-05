# Transformer

Work in progress. Beginning the work of replicating the Transformer architecture introduced by Vaswani et al [] in 2017. Here we replicate the vanilla structure with both the Encoder and Decoder. In the encoder, the original paper talks of layers. This is because each encoder block is repeated N number of times with each outpout of an encoder block passed to the next layer. This is similar in the decode section.
![image](https://github.com/user-attachments/assets/17f63a7f-0ea0-4b29-97fd-b1b5ae27c999)


The encoder is responsible for encoding hidden information about all tokens in a sequence (or senence) all at once. This information is then passed to the decoder as context to inform the generation of tokens. The decoder takes one token as an input at a time and based on this context uses it to produces a singular token. This singular token, once output, is then fed back to the decoder to start the generation of thge next token. Generation is intially started by senging a begin sequence token and ended with an end of sequence token. 

The input tokens are first embedded or tokenized. In the context of natural language processing e.g analysing text , we must train the model on text inputs. However, the model cant understand text and so must be converted into a number. Here the BertTokenizer is used, accessed through Hugging Face. In this repository we use the "Healthcare NLP: LLMs, Transformers, Datasets" [2] which trains the transformer to answer medical questions about a subset of diseases. The BertTokenizer first embeds the text into numerical values. Following [1] you thrn perform positional encoding, which essentially encodes the importance of the positioning of words in a sentence. A more common method is to use Rotary encoding.

First, the encoder. The embedded tokens, are first project linearly using a simple neural network linear layer e.g. nn.Linear(). This produces three vectors called Query (Q), Keys (K) and Values (V). All together they can be thought of as being a mapping system. The query vector is the input token that is like asking a question against all other tokens in the sequence. The keys are vectors that are related to a specific value. They map directly to a specific value. The value vector representing the real token we want to output. The key vector acts as a way of comparing the queries e.g. input tokens to the real tokens, the mechanism of comparison is described when we talk about attention.
![image](https://github.com/user-attachments/assets/6758624a-83dc-4cd8-b6c1-719336c8a4fe)

Here we describe the attention mechnism. The attention mechanism is seen above. 

After all the encoder layers, the final output is QKV. The key and value vectors are passed to the decoder into the attention mechansim referred to either as encoder-decoder attention or cross-attention. This begins the decoder discussion.

The decoder 

A great resource for this topic is by Jay Alammar []



[1] [Vaswani et al](https://arxiv.org/abs/1706.03762)
[2] https://www.kaggle.com/datasets/jpmiller/layoutlm
[]  https://jalammar.github.io/
