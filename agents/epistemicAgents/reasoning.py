import logging
import Levenshtein
import params
import llm_utils
from beliefs import Belief
from beliefStore import BeliefStore
from beliefSet import BeliefSet
from support import Support

"""
This is for reasoning.
Instead of using llms to do all the work, we 
Uses llm_utils to do the work. 
"""

blog = logging.getLogger()

def verify(p: Belief,
           bset: BeliefSet,
           bstore: BeliefStore,
           depth: int = params.BS_DEPTH,
           propagation_min_change = params.PROP_MIN_CHANGE
           ):
    """
    TODO:
        verify the negation of the query also
        implement other means of veridication
    THis is to verify a belief. It will update the confidence in the belief and
    add new ones possibly
    :param p:  the beleif to verify
    :param bset: the belief set this is from
    :param bstore: the store
    :param depth: max steps of recursion in the verification
    :param propagation_min_change: minimum change in confidence to propagate the
    change to downstream beliefs
    :return: True or False according to success. can be False and partly successful
    """
    pos_result =  verify_by_bs(p, bset, bstore, depth, propagation_min_change)
    # we should also negate p and verify it
    # by contradiction
    # by web search
    # by dbpedia or others
    # by asking user

def verify_by_bs(p: Belief,
                 bset: BeliefSet,
                 bstore: BeliefStore,
                 depth: int = params.BS_DEPTH,
                 propagation_min_change = params.PROP_MIN_CHANGE,
                ):
    """
    Given a belief and a beliefset it is from, this verifies the belief
    the result is that the spport set and the confidence in the belief are modified.
    This is done by backward chaining for one step with the help of a llm. If the llm
    uses statements not in the beliefset, these are also verified to the max depth
    specified.
    THe result can be the addition of new statements and beliefs and an upodate
    to the support and confidence of existing beliefs.
    :param p: belief to be verified
    :param bset: the set where it is from
    :param bstore: the store
    :param depth: how much work to do to try to verify
    :param propagation_min_change: min change in confidence that will be propagated
    :return: true if all recusrsive verifications succeeded, else false
        sideeffet: updates the support of p

    - select relevant beliefs from the bset
    - llm_utils.backward_step (for now just one step) with sentences of p and relevant bset
    - if necessary, add support, new beliefs

    - do same with the negation of p and merge (llms not so smart about negations,
    so we need to do separately)
    """
    # list of bid, dist, text
    query = p.text_rep
    blog.info(f"verify_bs {query}")
    matches, _, _ = bstore.get_similar_beliefs(txt=query, beliefset_id=bset.id,
                                               max_dist=params.RB_THRESHOLD,
                                               max_match=20,
                                               mult_match=True,)
    blog.debug('similar beliefs Matches:\n' + '\n'.join([m[2] for m in matches]))
    match_texts = [m[2] for m in matches if Levenshtein.ratio(m[2], query) < params.LEVENSHTEIN_LB]
    blog.debug("relevant facts: " + '\n'.join(match_texts))
    try:
        response, assumptions, facts, concl = llm_utils.backward_step(query, match_texts)
    except Exception as e:
        blog.warning('Cannot find backward step for ' + query)
        return False
    blog.debug('llm response\n' + response.response)
    blog.debug("facts used: " + '\n'.join(str(facts)))
    blog.debug("assumptions: " + '\n'.join(assumptions))
    if concl is not None and len(concl) > 0:
        assumption_bels = []
        supported_by = []
        fact_ids = []
        # TODO:
        #if len(concl) > 1:
        #    best_concl, others = pick_best_concl(concl, query)
        #    assumptions.extend(others)
        for fp in facts:
            # facts used should be among the matches above. identify these here,
            f = fp[0]
            for m in matches:
                blog.debug('considering match ' + str(m))
                if Levenshtein.ratio(m[2], f) >= params.LEVENSHTEIN_LB:
                    fact_ids.append(m[0])
                    b = Belief.by_id(m[0], bstore)
                    blog.info(f"Belief for fact\n{str(b)}")
                    supported_by.append(b.support)
                    blog.info('matches')
                    break
            #if not is_fact:
            #    assumptions.append(f)
        if len(fact_ids) < len(facts):
            blog.warning("missing beliefs for facts")
        # assumptinos are not among the facts provided
        for a in assumptions:
            blog.debug('processing assumption ' + a)
            support = Support.from_source(p.id,{'source': 'llm'}, bstore)
            supported_by.append(support)
            bel = bset.add_bel_from_support(a, support)
            if bel is not None:
                assumption_bels.append(bel)
            else:
                blog.warning('Cannot get beleif for assumption ' + a)
        support = Support.from_reasoning(bstore, p.id, [s.id for s in supported_by],
                                         {'method': 'backward_search'})
        # update the support for the belief to verify
        p.support = Support.by_merging(p.id, p.support, support, 1, bstore)
        results = True
        # try to verify assumptions
        for a in assumption_bels:
            blog.info('Recursing on ' + a.text_rep)
            s = verify(a, bset, bstore, depth = depth - 1, propagation_min_change=params.PROP_MIN_CHANGE )
            results = results and s
        return results
    else:
        blog.warning("backward search failure for " + query)
        return False

def pick_best_concl(candidates, query):
    """
    If the llm output parsing ends up with multiple conclusiongs, pick the one
    closest to what we want
    :param candidates: multiple conclusions from llm query
    :param query: the query
    :return: the conclusion, other statements
    """
    bad_ones = []
    good_ones = []
    for c in candidates:
        dist = Levenshtein.ratio(c, query)
        if dist < params.LEVENSHTEIN_LB:
            bad_ones.append(c)
        else:
            good_ones.append((c, dist))
    if len(good_ones) > 0:
        good_ones.sort(key=lambda x: x[1], reverse=True)
        return good_ones[0], bad_ones
    else:
        return None, candidates