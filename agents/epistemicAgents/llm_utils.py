
import os
import ollama
from tenacity import *
import logging
import random
import regex
import Levenshtein
import params
import json
import requests
import utils.nl_utils as nl_utils

"""
Methods:
    - make_query_no_format: sends query to ollama
    - make_query_with_format: sends query to ollama, specifying result format
    - get_degree_similarity: returns degree of similarity between 2 strings
    - get_polarity: finds if two strings have the same polarity
    - get_degree:  returns degree of similarity or opposition between 2 strings
    - backward_step: finds how a conclusion can result from some premises,
    - quick_parse: parses the response of doing backward step 
    - dereference_text: dereferences text input
    - is_derivable: determines if a conclusion is derivable from sone premises

"""

# logging
#blog = logging.getLogger()
#blog.setLevel(logging.INFO)

log = logging.getLogger()
log.setLevel(logging.DEBUG)

llog = logging.getLogger('llog')

"""
All of these need to be in files loaded as needed
"""
TASK_PROMPTS = {}   # task -> llm -> [system, prompt]
LLM_PREFS = {}      # task -> {llm -> pref}

# maps query type to supertype
QUERY_MAP = {'get_polarity': 'semantic',
            }

# query-supertype -> list of models (ordered)
MODELS = {'semantic': ['mistral'],
          }

PROMPTS = {}    # task/model -> {label, text}
SYSTEM_MESSAGES = {}    # same
# if there is no task specific data, the dict is empty

def get_llm_msg(task, model=None, msg_type_lbl='prompts', label=None):
    """
    return the prompt or system message

    :paeam msg_type: system or prompt
    :param task: the task
    :param model if None, will take the 'default'
    :param label if None will pick one at random

    the prompts (and sys-messages) are cached in a dict: task/model -> label -> text
    given a task not all models will have prompts, in which case we use default
    assume that each task has a default set of prompts
    assume that if a model either has no prompts and sys-msgs or it has at lease
    one of each
    relpath: relative path
    abspath: absolute path
    suffix tmt: task, model, type (ptompts/systems)
    """
    log.debug(f"get_llm_msg {task}, {model}, {msg_type_lbl}")
    assert (msg_type_lbl=="systems" or msg_type_lbl=="prompts")
    assert (not task is None)
    if model is None:
        model = 'default'
    global PROMPTS
    global SYSTEM_MESSAGES
    msgs = PROMPTS if msg_type_lbl == 'prompts' else SYSTEM_MESSAGES
    basedir = params.LLM_PROMPT_DIR
    relpath_t = task
    relpath_tm = os.path.join(relpath_t, model)
    relpath_tmt = os.path.join(relpath_tm, msg_type_lbl)
    lbl_text = None
    log.debug(f"relpath: {relpath_tm}")
    if relpath_tm in msgs:
        # we have seem this before and cached the llm-inputs
        if msgs[relpath_tm] == {}:
            # use default. this should exist
            lbl_text = msgs[os.path.join(relpath_t, 'default')]
        else:
            lbl_text = msgs[relpath_tm]
    else:
        log.debug('Not cached')
        # need to get the model-specific texts and default if needed
        msgs[relpath_tm] = {}
        if os.path.exists(abspath_tm := os.path.join(basedir, relpath_tm)):
            # there is a model subdir
            # assume there is at least one prompt/sys_message
            abspath_tmt = os.path.join(abspath_tm, msg_type_lbl)
            files = os.listdir(abspath_tmt)
            log.debug(f"looking in {abspath_tmt}")
            for f in files:
                if os.path.isfile(the_file := os.path.join(abspath_tmt, f)):
                    with open(the_file, 'r') as xx:
                        log.info(f"Loading {the_file}")
                        text = xx.read()
                    msgs[relpath_tm][f] = text
            lbl_text = msgs[relpath_tm]
        else:
            # there is no model specific prompt or sys-message
            # assume there is a default set
            msgs[relpath_tm] = {}   # leave empty to direct to default
            abspath_tmt = os.path.join(basedir, relpath_t, 'default', msg_type_lbl)
            files = os.listdir(abspath_tmt)
            log.debug(f"looking at files in {abspath_tmt}")
            relpath_td = os.path.join(relpath_t, 'default')
            msgs[relpath_td] = {}
            for f in files:
                print('processing file ', f, ' in ', abspath_tmt)
                if os.path.isfile(the_file := os.path.join(abspath_tmt, f)):
                    with open(the_file, 'r') as xx:
                        log.info(f"Loading {the_file}")
                        text = xx.read()
                        msgs[relpath_td][f] = text
            lbl_text = msgs[relpath_td]
    if label is None or label not in lbl_text.keys():
        chosen = random.choice(list(lbl_text.keys()))
    else:
        chosen = label
    return lbl_text[chosen]

@retry(retry=retry_if_exception(Exception), stop=stop_after_attempt(params.LLM_RETRIES),
       after=after_log(log, logging.WARN))
def make_query_no_format(model, system_msg, prompt, server=True):
    """Send prompt to ollama. Return text."""
    log.info(f'make_query_no_format {model}')
    if not server:
        response = ollama.generate(
            system=system_msg,
            prompt=prompt,
            model=model
        )
    else:
        resp = requests.post(
            params.ollama_generate_host,
            json={"model": model,
                  "system": system_msg,
                  "prompt": prompt,
                  "stream": False}
        )
        response = resp.json()
    llog.info('SYSTEM: ' + system_msg)
    llog.info('PROMPT\n' + prompt)
    llog.info('RESPONSE\n' + response['response'])
    return response

@retry(retry=retry_if_exception(Exception), stop=stop_after_attempt(params.LLM_RETRIES),
       after=after_log(log, logging.WARN))
def make_query_with_format(model, system_msg, prompt, format, server=True):
    """Send query to ollama and expect formatted output"""
    log.info(f'make_query_with_format {model}')
    if not server:
        response = ollama.generate(
            system=system_msg,
            prompt=prompt,
            model=model,
            format = format
        )
    else:
        resp = requests.post(
            params.ollama_generate_host,
            json={"model": model,
                  "system": system_msg,
                  "prompt": prompt,
                  "format": format,
                  "stream": False}
        )
        response = resp.json()
    llog.info('SYSTEM: ' + system_msg)
    llog.info('PROMPT\n' + prompt)
    llog.info('RESPONSE\n' + response['response'])
    return response


def get_degree_similarity(s1: str,
                          s2: str) -> float:
    """Find the degree of similarity or opposition between two strings."""
    log.debug(f"llm_utils.get_degree_similarity\n{s1}\n{s2}")
    pol = get_polarity(s1, s2)
    if pol == 0:
        log.debug("polarisy 0")
        return 0
    else:
        return get_degree(s1, s2, pol==1) * pol

@retry(retry=retry_if_exception(Exception), stop=stop_after_attempt(params.LLM_RETRIES),
       after=after_log(log, logging.WARN))
def get_polarity(s1:str, s2:str)->int:
    """
    Find if 2 texts are in the same or opposite directions

    :param s1: a piece of text
    :param s2: another piece of text
    :return: -1, 0, 1 as to whether the sentences are opposed, unrelated or
    similar in meaning

    retry can choose different models in case something goes wrong
    """
    log.info("get_poarity")
    model = random.choice(params.SEMANTIC_MODELS)
    sys_msg = get_llm_msg('get_polarity', model=model, msg_type_lbl='systems')
    prompt = get_llm_msg('get_polarity', model=model, msg_type_lbl='prompts').format(s1=s1, s2=s2)
    response = make_query_no_format(model, sys_msg, prompt)
    resp = response['response'].lower()
    rv: int = 0
    rv = 1 if 'similar' in resp else rv
    rv = -1 if 'opposite' in resp else rv
    log.debug(f"Polarity {rv}")
    return rv

@retry(retry=retry_if_exception(Exception), stop=stop_after_attempt(params.LLM_RETRIES),
       after=after_log(log, logging.WARN))
def get_degree(s1: str, s2: str, similarity: bool):
    """Find how similar or opposite are the 2 texts"""
    log.info("get_degree")
    model = random.choice(params.SEMANTIC_MODELS)
    sys_msg = get_llm_msg('similar_degree', model=model, msg_type_lbl='systems')
    if similarity:
        ptype = 'similar_degree'
    else:
        ptype = 'opposite_degree'
    prompt = get_llm_msg(task=ptype, model=model, msg_type_lbl='prompts').format(s1=s1, s2=s2)
    response = make_query_with_format(model, sys_msg, prompt,
                                     {"type": "number", "minimum": 1, "maximum": 10})
    val = int(response['response'])
    assert val >= 0
    assert val <= 10
    return float(val)/10

def backward_step(query: str,
                  facts: list[str],
                  with_assumptions: bool = True):
    """
    Find how the conclusion can derive from the premises.

    This uses a llm to look for evidence for a conclusion we are interested in.
    It is used as a step in a backward searh method that adds intermediate beliefs
    and continues stepping back until we get to known facts or we run out of resources.
    Find evidence for the query from the set of relevant facts, possibly incliuding
    commonsense knowledge and assumptions.
    :param query: conclusion
    :param facts: premises
    :return: response, assumptions, facts-used, conclusion
        response is the output of the llm
        assumptinos include commonsense knowledge, specific assumptions,
            intermediate steps
        concludion matches query or not
    """
    #

    # try to first infer from the belief set only
    # remove coz this is not working now
    log.info(f"backward_step")
    #model = random.choice(INFERENCE_MODELS)
    model = 'mistral'
    if with_assumptions:
        response = do_backstep_query(facts, query, model, 'any')
        tagged_response = nl_utils.tag_llm_justification(response, facts, query)
    else:
        response = do_backstep_query(facts, query, model, 'in')
        tagged_response = nl_utils.tag_llm_justification(response, facts, query)
        # checking for 'conclusion' is not the most reliable approach
        if 'conclusion' not in tagged_response:
            response = do_backstep_query(facts, query, model, 'any')
            tagged_response = nl_utils.tag_llm_justification(response, facts, query)
    return response, tagged_response


@retry(retry=retry_if_exception(Exception), stop=stop_after_attempt(params.LLM_RETRIES),
       after=after_log(log, logging.WARN))
def do_backstep_query(facts: list[str],
                      query: str,
                      model:str = 'mistral',
                      prompt_type: str = 'any'):
    #fact_str = '\n'.join(facts)
    log.info("do_backstep_query")
    fact_str = ''
    for i, f in enumerate(facts):
        fact_str += f"{i}. {f}"
    if prompt_type == 'in':
        prompt = get_llm_msg('backstep_in').format(conclusion=query, facts=fact_str)
        sys_msg = get_llm_msg('backstep_in', msg_type_lbl='systems')
    elif prompt_type == 'any':
        prompt = get_llm_msg('backstep_any', label='v3').format(conclusion=query, facts=fact_str)
        sys_msg = get_llm_msg('backstep_any', msg_type_lbl='systems')
    else:
        log.warning('Unknown prompt type')
        return None
    response = make_query_no_format(model, sys_msg, prompt)
    return response

@retry(retry=retry_if_exception(Exception), stop=stop_after_attempt(params.LLM_RETRIES),
       after=after_log(log, logging.WARN))
def get_inference_likelihood(premises, consequence):
    log.info('get-ingerence-likelihood')
    model = 'mistral'
    format = 'json'
    system_msg = get_llm_msg('inference_likelihood', model=model,
                             msg_type_lbl='systems', label='v1')
    prompt = get_llm_msg('inference_likelihood', model=model,
                             msg_type_lbl='prompts', label='v1')
    str_premises = '\n'.join(premises)
    fprompt = prompt.format(consequence=consequence, sentences=str_premises)
    log.debug(fprompt)
    response = make_query_with_format(model = model,
                                      system_msg=system_msg,
                                      prompt=fprompt,
                                      format='json')

    likelihood = json.loads(response['response'])['likelihood']
    assert likelihood >= 0.0
    assert likelihood <= 1.0
    return likelihood


@retry(retry=retry_if_exception(Exception), stop=stop_after_attempt(params.LLM_RETRIES),
       after=after_log(log, logging.WARN))
def str_compare_llm(target: str, candidate: str):
    """
    use a llm to see if the candidate directly implies the target
    :param target: string we are tyring to find in
    :param candidate: the candidate which may contain a paraphrase of the target
    :return: True or False

    Using a LLM is slower and more expensive but detect paraphrases.
    TODO: improve efficiency by filtering key terms
    """
    log.info(f"str-compare-llm")
    sys_msg = get_llm_msg(task='direct_implic', model='mistral', msg_type_lbl='systems')
    prompt = get_llm_msg(task='direct_implic', model='mistral').format(candidate=candidate,
                                                                      query=target)
    log.debug(f"str_compare\n{prompt}")
    answer = None
    response = make_query_with_format(model='mistral',
                                      system_msg=sys_msg,
                                      prompt=prompt,
                                      format='json'
                                      )
    log.debug(f"response: {response['response']}")
    implies = json.loads(response['response'])
    answer = implies['implies']
    assert (answer == 0 or answer == 1)
    if answer == 1:
        return True
    else:
        return False




def dereference_text(text):
    """Use the LLM to resolve pronominal anaphora."""
    models = ['llama3.2', 'mistral', 'phi3', 'deepseek-r1']
    sys_msgs = {'basic_1': "You are a capable linguist.",
                'map_1': """You are a language model trained to resolve pronominal references. Given a \
text, identify all pronouns and for each pronoun, decide what noun phrase it refers to. If you cannot \
find a referent for a pronoun use [UNKNOWN]. Your output should be a list of pairs of pronouns and \
referents for all pronouns in the text in order.
                """,
                'gemini': """You are a text processing agent designed to resolve pronominal references \
within given text. Your task is to identify all pronouns and replace them with the nouns or noun phrases \
they refer to. Maintain the original sentence structure and context as much as possible. \
If a pronoun's antecedent is ambiguous or cannot be determined, indicate this with "[AMBIGUOUS]". \
If a pronoun refers to something outside the provided text, mark it as "[EXTERNAL]". \
Your output should be the modified text with all resolvable pronouns replaced.""",
                'gemini_2': """You are a text processing agent designed to resolve references within given text. Your task is to identify all pronouns and replace them with the nouns or noun phrases they refer to. Maintain the original sentence structure and context as much as possible. Your output should be the modified text with all resolvable pronouns replaced.""",
                'chatgpt': """You are a highly capable language model designed to accurately resolve all pronominal references in text. Your task is to identify the antecedent of every pronoun, clarify ambiguous references, and ensure that every pronoun is linked to its appropriate noun or noun phrase. This includes resolving pronouns related to gender, number, and person. When necessary, provide explicit clarification for any pronoun that lacks a clear antecedent.""",
                'chatgpt_2': """You are a highly capable language model designed to accurately resolve all references in text. Your task is to identify the antecedent of every reference, clarify ambiguous references, and ensure that every pronoun is linked to its appropriate noun or noun phrase. This includes resolving pronouns related to gender, number, and person.""",
                }

    prompts = {'map_1': f"""Given the text below, map the pronouns to their referents.
    
    Example: 
    Text: "John was late for school. He had overslept again."
    Map: ('He', 'John')
    
    Text: {text}
    """,
               'map_2': f"""Map the pronouns to their referents for the text below.

Example:
Text: "John brought his dog to the park. There he met Jill. He gave it to her and went swimming."

    Map: [['He', 'John'], ['he', 'John'], ['it', 'his dog'], ['her', 'jill']]

    Text: "{text}"
    """,
'replacing_1': f"""Given the text below, replace the pronouns with their referents.

    Example: 
    Text: "John was late for school. He had overslept again."
    Answer: "John was late for school. John had overslept again."

    Text: {text}
    """,
               'gemini': f"""Please resolve the pronominal references in the following text:

"{text}"
               """,
               'gemini_2': f"""Please resolve the references in the following text:

               "{text}"
                              """,
               'chatgpt': f"""Please resolve all pronominal references in the following text:
               
"{text}"
               """,
               'chatgpt_2': f"""Please resolve all references in the following text:

"{text}"
                """,
                  }
    response = make_query_no_format(models[3], sys_msgs['map_1'], prompts['map_2'])
    print(response.response)

    ####
    #### Not used
    ####


def is_derivable(textoi: str,
                    context: list[str],
                    with_llm: bool,
                    ):
       """
    is textoi derivable from the context and possibly with common knowledge in the llm?
    if so returns the sentences that are used to derive textoi, otherwise retuen None
    :param textoi: sentence we want to derive
    :param context: find a derivation from these sentences
    :param with_llm: whether we can use commonsense knowledge from the llm
    :return: list of sentences or None if not derivable
    """
       pass

def make_query(query_type: str,     # query type
               model: str = None,   # model to use
               params: list[str] = [],  # query params
               struct_out: str = None,     # for structured output
               ):
    """
    Geenric llm query runner,

    TODO: eliminate that
    TODO: number of repeats with one model,
    TODO: number of models to run,
    TODO: result merging
    TODO: structured output if supported
    For not assume all models are ollama
    :param query_type:  type of query: get_similarity etc
    :param model: what model to use. if none, uses the first model dor that supertype
    :param params: parameters to the query.
    :return: response of llm
    """
    query_stype = QUERY_MAP[query_type]
    if model is None:
        model = MODELS[query_stype][0]
    if model in SYSTEM_MESSAGES[query_stype]:
        system_msg = SYSTEM_MESSAGES[query_stype][model][0]
    else:
        system_msg = SYSTEM_MESSAGES[query_stype]['default'][0]
    if model in PROMPTS[query_type]:
        prompt = PROMPTS[query_type][model][0]
    else:
        prompt = PROMPTS[query_type]['default'][0]
    pprompt = prompt.format(*params)
    try:
        if struct_out is None:
            response = make_query_no_format(model, system_msg, pprompt)
        else:
            response = make_query_with_format(model, system_msg, pprompt, struct_out)
    except Exception as e:
        log.error(f"Cannot do llm query\n{str(e)}")
        raise e
    return response

def get_degree_X(s1: str, s2: str, sim: int) -> float:
    """
    NOT USED

    :param s1: a piece of text
    :param s2: another piece of text
    :return: the degree to which the model thinks they are the same
    or different
    """
    log.debug("llm_utils.get_degree")
    if sim == -1:
        response = make_query('opposite_degree', None, [s1, s2],
                              {"type": "number", "minimum": 1, "maximum": 10})
    else:
        response = make_query('similar_degree', None, [s1, s2],
                              {"type": "number", "minimum": 1, "maximum": 10})
    val = response.response
    log.debug(f"Degree of simialrity/differnece {val}")
    return float(val)/10
