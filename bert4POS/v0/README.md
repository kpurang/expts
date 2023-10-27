# Bert embeddings as part of speech

It is well known that Bert represents part of speech of words. In this experiment, I try to make this explicit. This is the just first version of this, a lot still needs to be done.

The aim is to have the embedding represent the part of speech. A small bert model is instantitated and trained with the target being the POS. 

Training data is from the [Universal Dependencies English dataset](https://github.com/UniversalDependencies/UD_English-EWT). The bert model and tokenizer are from  Hugging Face bert-base-uncased. The pretrained tokenizer is used, but not the pretrained model.

A very small bert instance was used:
~~~
{
    "bertDir": "train/bert",
    "hidden_size": 18,
    "num_hidden_layers": 1,
    "num_attention_heads": 2,
    "intermediate_size": 32
}
~~~

The results were not spectacular. Losses over 4 epochs:
~~~
train,val
1.048260197752998,0.7031954436591177
0.9885094598645255,0.671472317115827
0.9625852228630156,0.6873327132427332
0.9360080872263227,0.6593429960987784
~~~

The losses are relatively high and the validation loss is lower than the training loss. 

More work needs to be done.
