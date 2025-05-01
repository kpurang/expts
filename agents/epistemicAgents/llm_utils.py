

import ollama
from tenacity import *
import logging
import random
import regex
import Levenshtein
import params

# logging
blog = logging.getLogger()

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
# query-supretype -> model -> list of messages
SYSTEM_MESSAGES = {'semantic':
                       {'default':
                            ["You apply common sense and everyday reasoning in your judgements.",
                                ],
                        },
                   'reasoning':
                       {'default':
                        ["You are an expert in commonsense knowledge and reasoning and follow instructions carefully."],
                        }
                   }
# query-type -> model -> list of prompts
PROMPTS = {'get_polarity':
               {'default':
                ["""
Are sentence 1 and sentence 2 below similar, opposite or unrelated in meaning? \
Say "similar", "opposite" or "unrelated" without explanation.
        
Sentence 1: {s1}
    
Sentence 2: {s2}
""",
                 ]},
           'opposite_degree':
                {'default':[
"""How much is Sentence 1 opposite of Sentence 2? Respond with a number from 1 \
to 10 with 1 being somewhat opposite and 10 being completely opposite. Only say the number.

Sentence 1: {s1}

Sentence 2: {s2}
""",
                ]},
          'similar_degree':
                {'default':[
"""How much is Sentence 1 similar to Sentence 2? Respond with a number from 1 \
to 10 with 1 being somewhat similar and 10 being completely identical. Only say the number.

Sentence 1: {s1}

Sentence 2: {s2}
"""
                ]},
           'backstep_in':
               {'default':[
"""Generate a plausible chain of reasoning to conclude '{}' from the fact set below only. \
If there is no such chain of reasoning, your response should be "None".
If there is a chain, your response should be a numbered list of the facts you used and the conclusion without any comment.
                   
Fact Set: 
{}
"""
               ]},
           'backstep_any':
               {'default':[
"""Generate a plausible chain of reasoning to conclude "{conclusion}" from the fact set below and \
other assumptions or commonsense knowledge that you need.
           
Generate your response as a numbered list of the facts and assumptions you used without comment. \
If you can't find a plausible chain of resoning, say 'None'.

Fact Set: 
{facts}
           """
    ]},
           'negate_sent':
               {'default':[
""" Generate the negation of "{}"

Do not provide any explanation or comment.
"""
               ]}

}


def make_query_no_format(model, system_msg, prompt):
    blog.info('model ' + model)
    blog.info("system " + system_msg)
    blog.info('prompt ' + prompt)
    response = ollama.generate(
        system=system_msg,
        prompt=prompt,
        model=model
    )
    return response

def make_query_with_format(model, system_msg, prompt, format):
    response = ollama.generate(
        system=system_msg,
        prompt=prompt,
        model=model,
        format = format
    )
    return response

def load_task_promts(task: str, base_dir: str = params.LLM_PROMPT_DIR):
    """
    Loads the task and
    :param task:
    :param base_dir:
    :return:
    """



def get_degree_similarity(s1: str,
                          s2: str) -> float:
    blog.debug(f"llm_utils.get_degree_similarity\n{s1}\n{s2}")
    pol = get_polarity(s1, s2)
    if pol == 0:
        blog.debug("polarisy 0")
        return 0
    else:
        return get_degree(s1, s2, pol==1) * pol

@retry(retry=retry_if_exception(Exception), stop=stop_after_attempt(paras.LLM_RETRIES))
def get_polarity(s1:str, s2:str)->int:
    """
    :param s1: a piece of text
    :param s2: another piece of text
    :return: -1, 0, 1 as to whether the sentences are opposed, unrelated or
    similar in meaning
    """
    model = random.choice(params.SEMANTIC_MODELS)
    sys_msg = random.choice(SYSTEM_MESSAGES['semantic']['default'])
    if model in PROMPTS['get_polarity']:
        prompt = random.choice(PROMPTS['get_polarity'][model]).format(s1=s1, s2=s2)
    else:
        prompt = random.choice(PROMPTS['get_polarity']['default']).format(s1=s1, s2=s2)
    response = make_query_no_format(model, sys_msg, prompt)
    resp = response.response.lower()
    rv: int = 0
    rv = 1 if 'similar' in resp else rv
    rv = -1 if 'opposite' in resp else rv
    blog.debug(f"Polarity {rv}")
    return rv

@retry(retry=retry_if_exception(Exception), stop=stop_after_attempt(paras.LLM_RETRIES))
def get_degree(s1: str, s2: str, similarity: bool):
    model = random.choice(params.SEMANTIC_MODELS)
    sys_msg = random.choice(SYSTEM_MESSAGES['semantic']['default'])
    if similarity:
        ptype = 'similar_degree'
    else:
        ptype = 'opposite_degree'
    if model in PROMPTS[ptype]:
        prompt = random.choice(PROMPTS[ptype][model]).format(s1=s1, s2=s2)
    else:
        prompt = random.choice(PROMPTS[ptype]['default']).format(s1=s1, s2=s2)
    response = make_query_with_format(model, sys_msg, prompt,
                                     {"type": "number", "minimum": 1, "maximum": 10})
    val = int(response.response)
    assert val >= 0
    assert val <= 10
    return float(val)/10

def backward_step(query: str,
                  facts: list[str]):
    """
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
    response, facts, assumptions, concl = do_backward_step(query, facts, 'in')
    if concl is not None:
        return response, assumptions, facts, concl
    response, facts, assumptions, concl = do_backward_step(query, facts, 'any')
    return response, assumptions, facts, concl

@retry(retry=retry_if_exception(Exception), stop=stop_after_attempt(params.LLM_RETRIES))
def do_backward_step(query: str,
                     fact_lst: list[str],
                     prompt_type: str,
                     ) -> (str, list[str], list[str], str):
    print(f'back_do {query}, {fact_lst}, {prompt_type}')
    response, facts, assumptions, concl = None, None, None, None
    model = random.choice(INFERENCE_MODELS)
    sys_msg = random.choice(SYSTEM_MESSAGES['reasoning']['default'])
    fact_str = '\n'.join(fact_lst)
    if prompt_type == 'in':
        prompt = random.choice(PROMPTS['backstep_in']['default']).format(query, fact_str)
    elif prompt_type == 'any':
        prompt = random.choice(PROMPTS['backstep_any']['default']).format(query, fact_str)
    else:
        blog.warning('Unknown prompt type')
        return response, facts, assumptions, concl
    response = make_query_no_format(model, sys_msg, prompt)
    print('response\n', response)
    #facts, assumptions, concl = parse_backstep(response.response, model)
    facts, assumptions, concl = quick_parse(response.response, query, fact_lst)
    assert len(facts) + len(concl) > 0      # most likely parsing failed, try again
    #real_concl = []
    #for c in concl:
        # TODO: also compare substrings and vector distance
    #    if Levenshtein.ratio(c, query) >= params.LEVENSHTEIN_LB:
    #        real_concl = [c]
    #        break
    #assert len(real_concl) == 1
    return response, facts, assumptions, concl

def quick_parse(response:str, query:str, bset:list[str]):  # was _2
    """
    simple parse of the response.
    :param response: llm response
    :param query: the query
    :param bset: list of facts provided
    :return: list of candidates for facts, assumption and conclusion
    """
    extracted = []
    _facts, facts = [], []
    _conclusion, conclusion = [], []
    _assumptions, assumptions = [], []
    lines = response.split('\n')
    xit = None
    for line in lines:
        line = line.strip()
        # maybe should eliminate this
        """
        if xit is not None:
            if len(line.strip()) > 0:
                cline, number, prop_type = try_identify(line)
                if prop_type is None:
                    extracted.append(cline)
                elif prop_type == 'fact':
                    facts.append(cline)
                elif prop_type == 'assumption':
                    assumptions.append(cline)
                elif prop_type == 'conclusion':
                    conclusion.append(cline)
                else:
                    blog.warning('Should not be here while parsing ' + line)
                xit = None   # ignores case of Facts: 1. ...\n 2. ...\n etc
        """
        cline, number, prop_type = try_identify(line)
        print('identified: ', cline, '\n', number, '\n', prop_type)
        if cline is not None and len(cline) > 0:
            if prop_type is None:
                extracted.append(cline)
            elif prop_type == 'fact':
                _facts.append(cline)
            elif prop_type == 'assumption':
                _assumptions.append(cline)
            elif prop_type == 'conclusion':
                _conclusion.append(cline)
            else:
                blog.warning('Should not be here while parsing ' + line)
        else:
            xit = prop_type
    #print('Extracted: ', extracted)
    #assert len(extracted) > 0
    print('Before str compare')
    print('Facts: ', _facts)
    print('Assumptions: ', _assumptions)
    print('Conclusion: ', _conclusion)
    print('Extracted: ', extracted)
    found_it = False
    # brute force greedy matching.
    facts_found = set()     # so we dont duplicate facts
    conclusion_found = False
    for i in range(len(_conclusion)):
        if str_compare(query, _conclusion[i]):
            conclusion.append((query, _conclusion[i]))
            conclusion_found = True
            extracted.extend(_conclusion[i+1:])
            break
        else:
            extracted.append(_conclusion[i])
    for fx in _facts:
        matched = False
        for i, f in enumerate(bset):
            if str_compare(f, fx):
                if f not in facts_found:
                    facts.append((f, fx))
                    facts_found.add(f)
                matched = True
                break
        if not matched:
            extracted.append(fx)
    bad_ones = []
    print('assumptions + extracted processing\n', extracted, '\n', assumptions)
    for aex in _assumptions + extracted:
        print('considering ', aex)
        if str_compare(query, aex):
            if not conclusion_found:
                conclusion.append((query, aex))
                conclusion_found = True
            print('is conclision')
            break
        matched = False
        for f in bset:
            if str_compare(f, aex):
                if not f in facts_found:
                    facts.append((f, aex))
                    facts_found.add(f)
                print('is fact')
                matched = True
                break
        if not matched:
            print('adding to assumptions')
            assumptions.append(aex)
    print('Facts: ', facts)
    print('Assumptions: ', assumptions)
    print('Conclusion: ', conclusion)
    return facts, assumptions, conclusion


    """
    for txt in extracted:
        print('Processing ', txt)
        is_fact = False
        print(query, ' ', txt, ' ', Levenshtein.ratio(query, txt))
        if Levenshtein.ratio(query, txt) >= params.LEVENSHTEIN_LB:
            conclusion.append(txt)
            continue
        else:
            for f in bset:
                print(f, ' ', txt, ' ', Levenshtein.ratio(f, txt))
                if Levenshtein.ratio(f, txt) >= params.LEVENSHTEIN_LB:
                    facts.append(txt)
                    is_fact = True
                    break
        if not is_fact:
            assumptions.append(txt)
    """

def str_compare(the_str:str, candidate:str):
    """
    quick way to approximately find a string in a longer string. Gvien a fact,
    the llm might modify and add more text to it in the output.
    TODO: loop over matches in case there are multiple
    :param the_str: string to find
    :param candidate: where to find it in
    :return: True or False
    """
    margin = 2
    lstr = len(the_str)
    sidx = random.randint(0, lstr//2)
    eidx = random.randint(0, lstr-sidx) + sidx
    loc = candidate.find(the_str[sidx:eidx])
    if loc == -1:
        return False
    lb = max(loc - sidx - margin, 0)
    ub = min(lb + lstr + margin + 1, len(candidate))
    if Levenshtein.ratio(the_str, candidate[lb:ub]) >= params.LEVENSHTEIN_LB:
        return True
    else:
        return str_compare(the_str, candidate[eidx+1:])

numRE = regex.compile('([^d])*(\d+)\.(.*)')
preRE = regex.compile('([^:\(]*):(.*)')
parRE = regex.compile('(.*)\((.*)\)(.*)')


def try_identify(line):
    """ try to identify the line by the predix or the parens, and return the clean string

    """
    print('try_identify ', line)
    num_str = None
    pre_str = None
    par_str = None
    if line.startswith('('):
        return '', None, None
    m = parRE.match(line)
    if m is not None:
        line = (m[1] or '') + (m[3] or '')
        par_str = m[2]
    m = numRE.match(line)
    if m is not None:
        line = (m[1] or '') + ' ' + (m[3] or '')
        num_str = m[2]
    m = preRE.match(line)
    if m is not None:
        line = m[2]
        pre_str = m[1]
    prop_type = None
    num = None
    if num_str is not None:
        num = int(num_str)
    if pre_str is not None:
        prop_type = id_type(pre_str.lower())
    if prop_type is None and par_str is not None:
        prop_type = id_type(par_str.lower())
    return line.strip(), num, prop_type

def id_type(id_str):
    prop_type = None
    if 'fact' in id_str:
        prop_type = 'fact'
    elif 'assump' in id_str or 'common' in id_str:
        prop_type = 'assumption'
    elif 'conc' in id_str:
        prop_type = 'conclusion'
    else:
        print('unknown type: ', id_str)
    return prop_type

def rm_prefix(text):
    rv = regex.sub('.*:', '', text)
    if rv != text:
        return rv
    rv = regex.sub('\s*\d+\.', '', text)
    return rv

def rm_parens(text):
    rv = regex.sub('\(.*\)', '', text)
    return rv

def dereference_text(text):
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
def make_query(query_type: str,     # query type
               model: str = None,   # model to use
               params: list[str] = [],  # query params
               struct_out: str = None,     # for structured output
               ):
    """
    Geenric llm query runner,
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
        blog.error(f"Cannot do llm query\n{str(e)}")
        raise e
    return response

def get_degree_X(s1: str, s2: str, sim: int) -> float:
    """
    :param s1: a piece of text
    :param s2: another piece of text
    :return: the degree to which the model thinks they are the same
    or different
    """
    blog.debug("llm_utils.get_degree")
    if sim == -1:
        response = make_query('opposite_degree', None, [s1, s2],
                              {"type": "number", "minimum": 1, "maximum": 10})
    else:
        response = make_query('similar_degree', None, [s1, s2],
                              {"type": "number", "minimum": 1, "maximum": 10})
    val = response.response
    blog.debug(f"Degree of simialrity/differnece {val}")
    return float(val)/10
