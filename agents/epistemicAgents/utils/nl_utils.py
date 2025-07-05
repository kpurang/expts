"""
nl_utils has various string manipulation and parsing utilities

Assume there is a ollma server running
"""

import re

import Levenshtein
import logging
import params
import numpy as np
import scipy.spatial.distance as distance
import json
import requests
import random
import llm_utils
import regex

EMBEDDING_MODEL = 'all-minilm'

# logging
log = logging.getLogger()
llog = logging.getLogger('llog')

paren_re = re.compile('([^\(]*)\((.*)\)([^\)]*)')
# was '(\w+)[: ](.*)'
prefix_re = re.compile('([\w\s]+):(.*)')

def tag_llm_justification(response_dict,
                          facts,
                          conclusion,
                          prefix_list=params.justification_prefixes,
                          ):
    """
    cleans up and standardises the types of sentence the llm generates in its
    justification

    :param response: response from llm justifying a conclusion = [justification]
    :param facts: facts used as basis for the conclusion
    :param conclusion: the conclusion to justify
    :param prefix_list: the prefixes to map to
    :return: a list of [prefix, justification line]
    """
    log.info("tag-llm-justification")
    response = response_dict['response']
    kwd_vectors = get_kwd_vectors(prefix_list)      # should save and reuse
    lines = [x.strip() for x in response.split('\n') if len(x) > 0]
    lines = [re.sub('^\W*', '', line) for line in lines]  # soetiems line starts with - or something
    for line in lines:
        log.debug(line)
    lines = strip_enum(lines)
    lines = strip_parens(lines)     # should be smarter about this
    tagged_lines = []
    num_conclusions = 0
    for line in lines:
        log.debug(f"Processing line: {line}")
        pre_m = prefix_re.match(line)
        labeled = False
        if pre_m is not None:
            log.debug('matching prefix')
            # we can identify a prefix
            prefix_dist = str_compare(pre_m[1], prefix_list, kwd_vectors)
            if prefix_dist is not None:
                tagged_lines.append([prefix_dist[0], pre_m[2].strip()])
                if prefix_dist[0] == 'conclusion':
                    num_conclusions += 1
                labeled = True
        if not labeled:
            # cannot identify a prefix, so we take all words
            log.debug('matching all words for prefix')
            words = line.split(' ')
            for word in words:
                prefix_dist = str_compare(word, prefix_list, kwd_vectors)
                if prefix_dist is not None:
                    tagged_lines.append([prefix_dist[0], line.strip()])
                    if prefix_dist[0] == 'conclusion':
                        num_conclusions += 1
                    labeled = True
                    break  # we should perhaps look at all the words iso being greedy
        if not labeled:
            tagged_lines.append('None', line.strip())
        log.debug(f"labeled as {tagged_lines[-1][0]}")
            # maybe throw exception here
            # or match with facts
    # fail if there are no conclusions
    if num_conclusions == 0:
        log.error("no conclusions")
        assert num_conclusions > 0
    # verify facts and match None. replace with the given fact.
    fact_vectors = get_kwd_vectors(facts)
    log.debug('matching facts')
    for t in tagged_lines:
        if t[0] in ['fact', 'None']:
            f_match = str_compare(t[1], facts, fact_vectors)
            if f_match is not None:
                log.debug(f"matched {t[1]} to {f_match[0]}")
                t[0] = 'fact'
                t[1] = f_match[0]
            else:
                log.debug(f"{t[1]} not matched to fact")
                t[0] = 'None'   # change fact to none is no match
    # if there are many labeled conclusion, pick the closest
    # assume at least one is a decent match
    if num_conclusions > 1:
        c_dists = []
        for i, tl in enumerate(tagged_lines):
            if tl[0] == 'conclusion':
                log.debug('Matching conclusion')
                #c_match = get_distance(conclusion, tl[1])
                #log.debug(f"{i}: {conclusion} - {tl[1]}: {c_match}")
                #c_dists.append([i, c_match])
                match_degree = str_compare(conclusion, [tl[1]], None)
                if type(match_degree) is not list or len(match_degree) != 2 or match_degree is None:
                    log.warning(f'getting bad matchdegree for conclusion {type(match_degree)} {str(match_degree)}')
                    # if match_degree is None:
                    match_degree = [tl[1], 99]
                c_dists.append(match_degree)
        log.debug("c_dists " + str(c_dists))
        c_dists.sort(key=lambda x: x[1])
        log.debug(str(c_dists))
        # ERROR: c-dists is list of [string, score] not tagged-line-index, string, score
        #for x in c_dists[1:]:
        #    tagged_lines[x[0]][0] = 'None'
        for i, tl in enumerate(tagged_lines):
            if tl[0] == 'conclusion':
                if tl[1] != c_dists[0][0]:
                    tagged_lines[i][0] = 'None'
                else:
                    tagged_lines[i][1] = conclusion
        # change to the expected conclusion
        #if c_dists[0][1] > 0:   # need a threshold here
        #    tagged_lines[c_dists[0][0]][1] = conclusion
        #else:
        #    tagged_lines[c_dists[0][0]][1] = 'None'
    # change all None to assumption
    llog.info('\nTAGGED:')
    llog.info('\n'.join([f"{tl[0]}: {tl[1]}" for tl in tagged_lines]) + '\n')
    log.info('\nTAGGED:')
    log.info('\n'.join([f"{tl[0]}: {tl[1]}" for tl in tagged_lines]) + '\n')
    return tagged_lines

def strip_enum(lines):
    return [re.sub('^\d+\.?', '', line).strip() for line in lines]

def strip_parens(lines):
    """
    strips parens and sometimes their contents from lines
    :param lines: lines to strip parens from
    :return: lines without parens

    Assumes only one pair of parens per line. **not true** fix that
    TODO: Sometiems the parens contain important info. needs fixing
    """
    rv = []
    for line in lines:
        m = paren_re.match(line)
        if m is not None:
            if len(m[1]) > 0 and len(m[2].strip()) <= params.p_prop * len(line):
                line = re.sub('\(.*\)', ' ', line)
            else:
                line = line.replace('(', ' ').replace(')', ' ').strip()
        rv.append(line)
    return rv

def strip_paren_line(line):
    # assume parens are balanced, and in the right order
    np_line = ''
    lb = 0
    while lb < len(line) - 1:
        open_ix = line[lb:].find('(')
        if open_ix < 0:
            np_line += line[lb:]
            break
        else:
            np_line += line[lb:lb+open_ix]
            lb = lb + open_ix + 1
            close_ix = line[lb:].find(')')
            #assert close_ix > 0
            if close_ix < 0:
                # if lone (, skip it. could also delete to end of line..
                lb += 1
            else:
                lb = close_ix + 1
    return np_line


def get_kwd_vectors(kwds: list[str]):
    """
    Converts a list of strings to a list of embeddings
    :param kwds: list of strings to convert
    :returns: list of embedding nparrays

    kwds is likely to be short. do this to avoid the ovehead of a vector db
    """
    vectors = []
    for k in kwds:  # could do all at once.
        resp = requests.post(
            params.ollama_embedding_host,
            json={"model": params.EMBEDDING_MODEL,
                  "input": k
                  })

        embeddings = resp.json()['embeddings'][0]
        vectors.append(np.array(embeddings))
    return vectors


def get_best_match(tomatch: str, embeddings, texts):
    # if texts is small enough, maybe can try str compare first
    sims = []
    cand = get_kwd_vectors([tomatch])[0]
    for i, e in enumerate(embeddings):
        sims.append([texts[i], distance.cosine(cand, e)])
    sims.sort(key=lambda x: x[1])
    return sims[0]


def get_distance(s1, s2):
    resp = requests.post(
        params.ollama_embedding_host,
        json={"model": params.EMBEDDING_MODEL,
              "input": [s1, s2],
              }
    )
    embs = resp.json()['embeddings']
    dist = distance.cosine(embs[0], embs[1])
    return dist

def str_compare(target: str, candidates: list[str] = [], embeddings = None):
    """
    want to see if the strings are the same or have similar meanings
    :param target: what we are looking for
    :param candidate: potential match
    :return: [match, match-degree]

    Assume the strings input are stripped
    use more expensive methods later
    """
    candidate_str = '\n'.join(candidates)
    log.debug(f"str_compare {target}, {candidate_str}")
    log.debug('equality check')
    ltarget = re.sub('[^\w\s]', '', target.lower()).strip()
    lcandidates = [re.sub('[^\w\s]', '', x.lower()).strip() for x in candidates]
    for i, candidate in enumerate(lcandidates):
        if ltarget == candidate:
            return [candidates[i], 0]
    log.debug('substring check')
    for i, candidate in enumerate(lcandidates):
        print(f"{i} {ltarget} -- {candidate}")
        if str_compare_sstring(ltarget, candidate):
            return [candidates[i], 1]
    log.debug('levenshtein dist')
    for i, candidate in enumerate(lcandidates):
        print(f"{i} {ltarget} -- {candidate}")
        if str_compare_levenshtein(ltarget, candidate):
            return [candidates[i], 2]
    if embeddings is None or len(candidates) < 5 :
        log.debug('looking ar distance')
        for i, candidate in enumerate(lcandidates):
            print(f"{i} {ltarget} -- {candidate}")
            if get_distance(ltarget, candidate) <= params.max_dist:
                return [candidates[i], 3]
    else:
        log.debug('using vector db')
        mrv =  get_best_match(ltarget, embeddings, lcandidates)
        print(mrv)
        if mrv[1] <=params.max_dist:
            return mrv
    # TODO: process all with one llm query
    log.debug('using llm compare')
    for candidate in candidates:
        if llm_utils.str_compare_llm(target, candidate):
            return [candidate, 4]
    log.debug('no match')
    return None

def str_compare_sstring(the_str:str, candidate:str):
    if the_str in candidate:
        return True
    elif candidate in the_str:
        return True
    return False

def str_compare_levenshtein(the_str:str, candidate:str):
    """
    quick way to approximately find a string in a longer string.

    Gvien a fact,the llm might modify and add more text to it in the output.
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
        return str_compare_levenshtein(the_str, candidate[eidx+1:])


def quick_compact(label, maxlen=params.max_graph_label_len):
    """
    split a belief text rep into multiple lines to look better in the graph
    :param label: the label
    :param maxlen: max length of a line
    :return: fixed label

    Quick approach to this.
    """
    words = label.split(' ')
    new_label = ''
    line = ''
    lb = 0
    ub = 0
    while ub < len(words):
        if len(line) > maxlen:
            new_label += ' '.join(words[lb:ub - 1]) + '\n'
            line = words[ub - 1] + ' '
            lb = ub - 1
        else:
            line += ' ' + words[ub]
            ub += 1
    new_label += ' '.join(words[lb:])
    return new_label


## below not used

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
    """Identify the type of line from the LLM response."""
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
    """ Cleaning LLM response."""
    rv = regex.sub('.*:', '', text)
    if rv != text:
        return rv
    rv = regex.sub('\s*\d+\.', '', text)
    return rv

def rm_parens(text):
    """ Cleaning LLM response."""
    rv = regex.sub('\(.*\)', '', text)
    return rv

