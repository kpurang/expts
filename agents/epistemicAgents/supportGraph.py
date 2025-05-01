import os, os.path
import json
from dataclasses import dataclass, field
from typing import ClassVar
from pymilvus import MilvusClient, DataType, FieldSchema, CollectionSchema
import ollama
import logging
import random
import pandas as pd
import params
from enum import Enum
import numpy as np

# logging
blog = logging.getLogger()
#fh = logging.FileHandler(filename=LOGFILE)
#fh.setLevel(logging.DEBUG)
#blog.addHandler(fh)

"""
This has methods that work with Supports to build a graph. this links supports 
based on which beliefs support which others. this is used to compute the 
confidence the agent has in statements
"""

"""
Basic methods.
these link a support to a bunch of others
"""

def bset_derivable(bel: Belief,
                   bset: BeliefSet,
                   bstore: BeliefStore,
                   with_llm: bool = False,
                   min_conf: float = 0.3,
                  ):
    """
    is this derivable from the bset
    :param bel:
    :param bset:
    :param min_conf:
    :return:
    """
    relevant_beliefs = bset.get_relevant_beliefs(bset, bel)
    result_sentences = llm_utils.is_derivable(bel.text_rep,
                                              [b.text_rep for b in relevant_beliefs],
                                              with_llm,
                                              min_conf)
    # result is a list of beliefs
    result = sentence_to_belief(result_sentences bset, bstore)
    confidence = compute_inference_conf(result)     # maybe take the lowest or * all
    if result is not None:
        new_support = Support(belief_id = bel.id,
                              confidence = confidence,
                              method = 'llm_utils.bset_derivable',
                              info = None,
                              supported_by = [b.id for b in result],
                              supports = [])
        merged_support = Support.by_merging(bel.id,
                                            old_support = bel.support,
                                            new_support = new_support,
                                            score = 1.0,
                                            bstore = bstore)
        new_support.supports.append(merged_support.id)
        merged_support.supported_by.append(bel.support.id, new_support.id)
        bel.support = merged_support
        return merged_support
    return None

def sentence_to_belief(result_sentences,
                       bset,
                       bstore,
                       ):
    """
    return beliefs matching senetences in bset or make new beliefs for the others
    :param result_sentences: sentences that may or may not be in bset
    :param bset: set of beliefs
    :param bstore: beliefstore
    :return:
    """
    beliefs = []
    return None
