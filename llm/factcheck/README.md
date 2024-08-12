
# Fact checking

LLMs are known for making up facts that turn out to be false. These are experiments in automatic fact checking:
- factcheck_webSearch.ipynb. This uses web search to find relevant urls and the Deberta inference model on relevant chubks of text to determine if the fact is true.
- factcheck_chatGPT.ipynb. This asks chatGPT whether a statement is true. Additionally asking for evidence and sources sometimes changes the outcome
- factcheck_palm.ipynb. This is similar to the above
- factCheck_webLLM.ipynb. Similar to the first version, this uses web search to find relevant documents but then uses chatGPT or PaLM to determine if the statement follows from the text retrieved.



