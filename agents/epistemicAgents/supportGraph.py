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
from beliefs import Belief
from beliefSet import BeliefSet
from beliefStore import BeliefStore
from support import Support
import llm_utils
import utils.nl_utils as nl_utils
import pydotplus.graphviz as pydot

# logging
log = logging.getLogger()
#fh = logging.FileHandler(filename=LOGFILE)
#fh.setLevel(logging.DEBUG)
#blog.addHandler(fh)

"""
Supports are edges and beliefs are nodes. This module implements graph methods
over that graph
"""


def plot_derivation(bel: Belief,
                    bstore,
                     gname: str = 'Derivation',
                     pngFname: str = '/tmp/derivation.png',
                     depth: int = 20):
    """
    generates a png of the derivation of the given belief

    the nodes are support objects labeled by the belief they support
    edges are between supports in supported_by or supports
    """
    log.debug('SentenceGraps:plot_derivation')
    log.debug('Support dict\n' + Support.dump_support_dict())
    graph = pydot.Dot(gname, graph_type="digraph", simplify=True, bgcolor="white")
    node2sid = []
    sid2node = {}
    print("Support of root bel: " + str(bel.support))
    label = nl_utils.quick_compact(bel.text_rep) #+ {bel.support.confidence:.2f}
    root = pydot.Node(name=f"s_{bel.support.id}",
                      label= f"{label} @{bel.support.confidence:.2f}",
                      shape='box',
                      style='filled',
                      fillcolor = params.support2color[bel.support.info.stype.name],
                      penwidth=2,
                      peripheries=2,
                      )
    sid2node[bel.support.id] = len(node2sid)
    node2sid.append(bel.support.id)
    graph.add_node(root)
    log.debug(f"Added concludion {bel.text_rep}")
    add_premises(graph, bel.support, depth-1, bstore, [])
    log.debug("dot file\n" + graph.to_string())
    #log.debug("\ngraphviz file\n" + graph.create_dot())
    png = graph.create(prog='dot', format='png')
    with open(pngFname, 'wb') as px:
        px.write(png)
    log.info(f"written png file: {pngFname}")
    return png

def add_premises(graph, support, depth, bstore, processed_nodes):
    if depth <= 0:
        return
    premises = []
    if support.id in processed_nodes:
        log.debug(f'Alreadu processed this node {support.id}')
        return
    log.info(f'addPremise depth: {depth}')
    for sid in support.supported_by:
        log.debug(f"supported by {sid}")
        the_support = Support.by_id(sid)
        premise = Belief.by_id(the_support.belief_id, bstore)
        label = nl_utils.quick_compact(premise.text_rep) + f"@ {premise.support.confidence:.2f}"
        pnode = pydot.Node(f"s_{sid}",
                           #label = f"{premise.text_rep} @{the_support.confidence:.2f}",
                           label=label,
                           shape='box',
                           style='filled',
                           fillcolor=params.support2color[the_support.info.stype.name],
                           )
        graph.add_node(pnode)
        log.debug(f"Added node {premise.text_rep}")
        pedge = pydot.Edge(dst=f"s_{support.id}",
                           src=f"s_{sid}",
                           arrowhead='normal',
                           color='black')
        graph.add_edge(pedge)
        log.debug(f"Added edge s_{sid} -> s_{support.id}")
        processed_nodes.append(support.id)
        if depth == 1:
            log.info('generate graph reached depth limit')
            return
        add_premises(graph, the_support, depth-1, bstore, processed_nodes)



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
    result = sentence_to_belief(result_sentences, bset, bstore)
    confidence = compute_inference_conf(result)     # maybe take the lowest or * all
    if result is not None:
        new_support = Support(belief_id = bel.id,
                              confidence = confidence,
                              method = 'llm_utils.bset_derivable',
                              info = None,
                              supported_by = [b.id for b in result],
                              supports = [])
        bel.merge_supports(new_support, 1.0)
        #merged_support = Support.by_merging(bel.id,
        #                                    old_support = bel.support,
        #                                    new_support = new_support,
        #                                    score = 1.0,
        #                                    bstore = bstore)
        new_support.supports.append(merged_support.id)
        merged_support.supported_by.append(bel.support.id, new_support.id)
        bel.support = merged_support
        return merged_support
    return None

def compute_inference_conf(res):
    return 0

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
