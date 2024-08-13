

## Summary

These are experiments in NLP, ML and LLMs

| Name              | Description       | Keywords |                                          
|-------------------|-------------------------------------------------------------|----|
| [bert4POS](https://github.com/kpurang/expts/tree/main/NL_DL/bert4POS/v0) | Training a little Bert model togenerate POS tags in its embeddings | Bert, POS tagger, Universal Dependencies |
| [dialecticChat](https://github.com/kpurang/expts/tree/main/llm/dialecticChat)     | Socratic conversation interface         | llm, chat, flask|
| [fakeGen](https://github.com/kpurang/expts/tree/main/llm/fakeGen)           | Generates plausible looking false statements                | llm prompting, TriviaQA |
| [factCheck](https://github.com/kpurang/expts/tree/main/llm/factcheck)         | Fact checking statements using web search and/or LLMs         | RAG, web search, Deberta|
| [llmConvoAnalysis](https://github.com/kpurang/expts/tree/main/llm/llmConvoAnalysis/llmConvoAnalysis_1.ipynb)  | Predicts the intent of conversational participants          | llm, langchain, conversation, ConvLab | 
| [surveyGen](https://github.com/kpurang/expts/tree/main/llm/surveyGen)     | Finds the most useful papers to read on some topic in arxiv | llm, prompting, summarization, embedding |
| [finetuneQuantizedGemma](https://github.com/kpurang/expts/blob/main/llm/fineTuneQuantizedGemma.ipynb) | Fine tuning the Gemma LLM | llm, Gemma, Lora, GCP, fine-tuning |
| [Visual anomaly detector](https://github.com/kpurang/expts/blob/main/llm/fineTuneQuantizedGemma.ipynb) | Using a convolutional autoenvoder to detect anomalies in images | vision, anomaly detection, convolutional autoencoder  |

## bert4POS

Bert represents all sorts of linguistic information in its output, but extracting this is hard. This is an expwriment to train a little instance of Bert to act as a part-of-speech tagger. The part of speech is explicitly encoded in the embeddings. 

The training data was from universal dependenies. The results of the little experiment could be better. A larger Bert instance and more data will help. 


## Dialectic chat

Here we have chatgpt questioning your thoughts and assumptions to help you gain clarity. I thought that chatgpt whould need some guidance in the form of different instructions and data processing to do that, but it knows how to do this out of the box. So this is simply a loop around chatgpt with appropriate instructions.

## FakeGen

This generates plausible sounding false statements from true ones. For example, one of these is true:
- Sunset Boulevard premiered in the US on 10th December 1993.
- Cats premiered in the US on 10th December 1993.

I use the triviaQA dataset for questions and (presumably) true answers and LLMs to generate true and false statements.

It would be interesting to see if this can be extended to generate fake but consistent and plausible stories from true ones.

## Factcheck

Given a statement like the above, can we tell whether it is true? 

LLMs are not very good at this. They make up sources to justify whatever they decide is true. Web search with inference works better. LLMs doing the inference works better than Bert based inference models.

## LLM Conversation Analysis

chat models are great at conversation, and must represent various aspects of conversation. This experiment is to try to get the intent of the speaker at each turn. LLMs make that easy

## Survey generator

Say you are interested in a topic and go to Arxiv to find more information. You are likely to find a long list of articles. Sorting through that to find what to read is not fun.

Given a topic, this script gets relevant papers from Arxiv and extracts some basic information that can help you decide quickly which of the papers you should read.

## Fine-tuning Gemma

Using LORA to fine tune the Gemma LLM. The LLM was fien tuned to improve its performance in guessing the prompt used given an input and output pair.

## Visual anomaly detection

Using a convolutional autoencoder to detect anomalies in images. In this case the images are from the MVTEC bottles daaset.




