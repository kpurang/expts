

# Epistemic agents

This project aims to provide a way for agents to have persistem explitit beliefs. These beliefs are then used as the basis of the actions of the agent. This is in contrast to LLM agents that have implicit beliefs that are only manifested through their influence on the actions of the agent.

Having explicit beliefs enables:

- transparency. We can see what the agent believes and how these lead to he actions it takes
- finer control over the actions of the agent by setting constraints on the beliefs the agent can ahve
- reasoning about what beliefs the agent should adopt. Not all information built into the agent or input to the agent is to be believed. We can have procedures that assign degrees of belief based on how it as obtained
- reasoning about beliefs. The agent can draw imferences from the beliefs it has.
- controlled resolution of conflicts. Conflicts between beliefs are inevitable and the agent can adopt procedures to resolve those.


The project uses LLMs for:

- language analysis
- simple commonsense reasoning
- retrieval of information stored in the LLM

# Overview

## Classes

- Belief: one to many relation between a support and one or more statements.
- BeliefSet: set of beliefs
- BeliefStore: storage, vector store and sql
- BSContext: the context for a belief set
- Source: a source of information
- Support: graph of supports for the agent's confidence in a belief
- SInfo: source information
- SType: types of source

## Modules

- llm_utils: utilities for invoking LLMs
- reasoning: doing reasoning with LLMs and beliefs
- supportGraph: represents the graph induced by the supports

## Putting it together

- An agent has beliefs
- Beliefs are represented by one or more semantically equivalent statements.
- Beliefs are members of belief sets.
- An agent has at least one belief set.
- Belief sets have contexts 
- Beliefs have supports that justify the beliefs confidence
- Supports are linked into graphs
- Sources are a kind of support





