import logging
import Levenshtein
import params
import llm_utils
from beliefs import Belief
from beliefStore import BeliefStore
from beliefSet import BeliefSet
from support import Support
from reasoner import Reasoner
import traceback
import json
import utils.nl_utils as nl_utils
from tenacity import retry, retry_if_exception, stop_after_attempt, after_log


"""
Methods:
    - verify: 		Given a belief, finds support for it and updates the confidence.
    - verify_by_bs    Verifies a belief by doing backward search
    - pick_bext_concl:	Given a belief, pick one from the KB closest to it.


"""

log = logging.getLogger()
just_log = logging.getLogger('just_log')

class SentenceReasoner(Reasoner):

    def __init__(self,
                 bstore: BeliefStore):
        super().__init__(bstore)

    def verify(self,
               p: Belief,
               bset: BeliefSet,
               max_depth: int = params.BS_DEPTH,
               propagation_min_change = params.PROP_MIN_CHANGE,
               fail_on_conclusion = True,
               ):
        """
        Updates the confidence in the belief given.

        TODO:
            verify the negation of the query also
            implement other means of veridication
        :param p:  the beleif to verify
        :param max_depth: max steps of recursion in the verification
        :param propagation_min_change: minimum change in confidence to propagate the
        change to downstream beliefs
        :return: True or False according to success. can be False and partly successful
        """
        conclusion =  self.verify_by_llm(p, bset, max_depth, propagation_min_change,
                                         fail_on_conclusion,)
        # we should also negate p and verify it
        # by contradiction
        # by web search
        # by dbpedia or others
        # by asking user
        return conclusion

    @retry(retry=retry_if_exception(Exception), stop=stop_after_attempt(params.LLM_RETRIES),
           after=after_log(log, logging.WARN))
    def verify_by_llm(self,
                      p: Belief,
                      bset: BeliefSet,
                      premises:list[Belief] = [],
                      max_depth: int = params.BS_DEPTH,
                      propagation_min_change = params.PROP_MIN_CHANGE,
                      add_matches:bool = False,
                      only_premises:bool = False,
                      fail_on_conclusion: bool = True,
                     ):
        """
        Verifies a belief by asking a llm

        :param p: belief to be verified
        :param premises: premises to use to get to the result
        :param max_depth: how many layers deep to look
        :param propagation_min_change: min change in confidence that will be propagated
        :param add_matches: whether to add matching statements from the beliefset to premises
        :param only_premises: whether to try to get p using only the premises, without llm assumptions
        :param fail_on_conclusion: raise exception if the conclusion cannot be verified
        :return: the updated conclusion (support will change)

       Given a belief and a beliefset it is from, this verifies the belief
        the result is that the spport set and the confidence in the belief are modified.
        This is done by backward chaining for one step with the help of a llm. If the llm
        uses statements not in the beliefset, these are also verified to the max depth
        specified.
        THe result can be the addition of new statements and beliefs and an upodate
        to the support and confidence of existing beliefs.
        - select relevant beliefs from the bset
        - llm_utils.backward_step (for now just one step) with sentences of p and relevant bset
        - if necessary, add support, new beliefs

        - do same with the negation of p and merge (llms not so smart about negations,
        so we need to do separately)

        This can fail and be redone if the text the llm generates is opposite to
        existing beliefs and the old belief wins

        """
        # list of bid, dist, text
        query = p.text_rep
        premise_texts = [p.text_rep for p in premises]
        log.info(f"SentenceReasoner.verify_by_llm {query}")
        if premises == [] or add_matches:
            # TODO: ensure we dont use consequences of the query to try to prove it
            #   implement Bel.is_consequence_of(bel)
            matches, _, _ = self.bstore.get_similar_beliefs(txt=query,
                                                            beliefset_id=bset.id,
                                                   max_dist=params.RB_THRESHOLD,
                                                   max_match=20,
                                                   mult_match=True,)
            log.debug('similar beliefs Matches:\n' + '\n'.join([m[2] for m in matches]))
            premise_texts += [m[2] for m in matches if m[2] not in premise_texts] # if Levenshtein.ratio(m[2], query) < params.LEVENSHTEIN_LB]
        log.debug("relevant facts: " + '\n'.join(premise_texts))
        try:
            response, tagged_response = llm_utils.backward_step(query,
                                                                premise_texts,
                                                                with_assumptions=not(only_premises))
        except Exception as e:
            log.error('Exception ' + str(e))
            print(traceback.format_exc())
            log.warning('Cannot find backward step for ' + query)
            return p
        log.debug('TAGGED RESPONSE\n')
        for t in tagged_response:
            log.debug(str(t))
        links = self.link_argument_steps(tagged_response)
        log.debug(str(links))
        for link in links:
            log.debug("Premises:")
            for ix in link[0]:
                log.debug(f"\t{tagged_response[ix][1]}")
            log.debug(f"Consequene: {tagged_response[link[1]][1]}: {link[2]}\n")
        # add to beliefs and supports
        upd_conclusion, beliefs_by_tag = self.gen_conclusion_support(bset, links, tagged_response)
        if max_depth > 1:
            # try to verify assumptions
            # TODO: make sure no circularity
            for a in beliefs_by_tag['assumption']:
                log.info('Recursing on ' + a.text_rep)
                descendants = a.support.supports
                before_support = a.support
                # check the parms
                upd_a = self.verify(a,
                                    bset,
                                    max_depth =max_depth - 1,
                                    propagation_min_change=params.PROP_MIN_CHANGE,
                                    fail_on_conclusion = False,)

        return upd_conclusion


        """
        sfacts = [str(x) for x in facts]
        log.debug("facts used: " + '\n'.join(sfacts))
        log.debug("assumptions: " + '\n'.join(assumptions))
        log.debug("infer_steps\n" + json.dumps(infer_steps))
        if concl is not None and len(concl) > 0:
            assumption_bels = []
            supported_by = []
            fact_ids = []
            # this is the default approach
            # find beleif ids for facts used. These should be in the table
            for fp in facts:
                # facts used should be among the matches above. identify these here,
                f = fp[0]
                for m in matches:
                    log.debug('considering match ' + str(m))
                    if Levenshtein.ratio(m[2], f) >= params.LEVENSHTEIN_LB:
                        fact_ids.append(m[0])
                        b = Belief.by_id(m[0], self.bstore)
                        log.info(f"Belief for fact\n{str(b)}")
                        supported_by.append(b.support)
                        log.info('matches')
                        break
            if len(fact_ids) < len(facts):
                log.warning("missing beliefs for facts")
            # add assumptions as new beliefs
            # assumptinos are not among the facts provided
            for a in assumptions:
                log.debug('processing assumption ' + a)
                support = Support.from_llm(p.id,{'source': 'llm'}, self.bstore)
                supported_by.append(support)
                bel = self.bset.add_bel_from_support(a, support)
                if bel is not None:
                    assumption_bels.append(bel)
                else:
                    log.warning('Cannot get beleif for assumption ' + a)
            # add all assumptions asd facts together as supports for the conclusion
            support = Support.from_reasoning(self.bstore, p.id, [s.id for s in supported_by],
                                             {'method': 'backward_search'})
            # update the support for the belief to verify
            p.support = Support.by_merging(p.id, p.support, support, 1, self.bstore)
            results = True
            # try to verify assumptions
            for a in assumption_bels:
                log.info('Recursing on ' + a.text_rep)
                s = self.verify(a, max_depth =max_depth - 1, propagation_min_change=params.PROP_MIN_CHANGE)
                results = results and s
            return results
        else:
            log.warning("backward search failure for " + query)
            return False
        """


    def Xidentify_fact_belief(self, fact_text, matches):
        """
        #param fact_text: the text that is supposed to be a fact
        @param matches: the facts input to the reasoner, fact_text should match one
        of these
        @returns: None if the fact cannot be found, else the belief for that fact
        """
        b = None
        for m in matches:
            log.debug('considering match ' + str(m))
            if Levenshtein.ratio(m[2], fact_text) >= params.LEVENSHTEIN_LB:
                b = Belief.by_id(m[0], self.bstore)
                log.info(f"Belief for fact\n{str(b)}")
                log.info('matches')
                break
        return b

    def Xpick_best_concl(self, candidates, query):
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

    def link_argument_steps(self, argument: list[list[str]]):
        log.debug(f"SentenceREasoner.link_argument_steps")
        links = []
        supported_list = []
        ix = 0
        just_str = ""
        while ix < len(argument):
            #print("supported: ", '\n'.join(str(x) for x in supported_list))
            if argument[ix][0] in ['fact', 'assumption']:
                supported_list.append(argument[ix])
            if argument[ix][0] in ['consequence', 'conclusion', 'None']:
                consequence = argument[ix][1]
                cons_type = argument[ix][0]
                cons_idx = ix
                log.debug(f"processing consequence: {consequence}")
                premises = []
                premise_idx = []
                jx = len(supported_list) - 1
                is_supported = False
                # assume one premise not sufficient for a consequence. so preload
                # if jx > 0:
                #    premises.append(supported_list[jx][1])
                #    premise_idx.append(jx)
                #    jx -= 1
                while jx >= 0:
                    premises.append(supported_list[jx][1])
                    premise_idx.append(jx)
                    log.debug(f"Adding premise {jx}: {premises[-1]}\nNum premises: {len(premises)}")
                    if len(premises) >= 2: #2: maybe one premise can be sufficient sometimes
                        redocnt= 0
                        success = False
                        # try that multiple times - and take the best answer
                        while redocnt < params.LLM_RETRIES:
                            likelihood = llm_utils.get_inference_likelihood(premises, consequence)
                            redocnt += 1
                            if likelihood >= params.min_inference_likslihood:
                                links.append((premise_idx, cons_idx, likelihood))
                                is_supported = True
                                supported_list.append(argument[ix])
                                just_p = ['- ' + x + '\n' for x in premises]
                                just_str += f"{just_p}\n\t->{consequence}\n"
                                success = True
                                break
                        if success:
                            break
                    jx -= 1
                if not is_supported:
                    log.warning(f"FAILED TO SUPPORT {consequence}")
                    if cons_type in ['None', 'consequence']:
                        log.info('Adding as assumption')
                        supported_list.append(['assumption', consequence])
                    elif cons_type == 'conclusion':
                        log.error("Cannot get justification for conclusion")
                        raise ValueError("Cannot get justification for conclusion")
            ix += 1
            just_log.info('Justification\n' + just_str + '\n\n')
        return links

    def gen_conclusion_support(self, bset, links, tagged_response):
        """
        Converts the links indexed into the response to beliefs and support objs
        :param bset:
        :param links: list of [list of premise_idx, conclusion_idx, confidence in inference]
        :param tagged_response: list of [tag, sentence]
        :return: the conclusion belief, {tag -> [beliefs]}

        Assumes the consequences and assumptions are generated before they are used
        """
        log.info(f"SentenceReasoner.gen_conclusion_support")
        log.info(f"Links: {links}\nTaggedresponse:")
        for i, t in enumerate(tagged_response):
            log.info(f"{i}: {t}")
        belief_by_tag = {}    # tag -> [bel..] maybe add index?
        for t in params.justification_prefixes:
            belief_by_tag[t] = []
        response_beliefs = [None] * len(tagged_response)    # map argument texts to beliefs
        for link in links:
            log.debug("Processing link: " + str(link))
            for premise in link[0]:
                log.debug(f"processing premise {premise}:  {tagged_response[premise]}")
                if response_beliefs[premise] is not None:
                    pb = response_beliefs[premise]
                else:
                    log.debug('Premise not seen before')
                    pb, score, _, _ = bset.get_matching_existing_belief(tagged_response[premise][1])
                    sp = Support.from_llm(0, {'source': 'mistral'}, self.bstore)
                    if pb is None:
                        # the premise is not in the db. call it an assumption
                        log.debug(f"No belief for this premise - llm: {tagged_response[premise][1]}")
                        try:
                            pb = bset.add_bel_from_support(tagged_response[premise][1],
                                                           sp,
                                                           do_match=False)
                        except Exception as e:
                            log.error('Cannot get belief\n' + str(e))
                            raise e
                        belief_by_tag['assumption'].append(pb)
                    else:
                        #_, status = pb.merge_supports(sp, score)
                        #sp.belief_id = pb.id
                        #sp.update()
                        #if status not in (0, 2):    # not (merged or sp wins)
                        if score < 0:
                            log.error(f"Premise is opposite to existing belief and lost. Redo reasoning")
                            raise ValueError ("LLM generated fact opposite to existing belief and eliminated.")
                        # can be a consequence or a fact
                        if tagged_response[premise][0] == 'fact':
                            belief_by_tag['fact'].append(pb)
                        else:
                            belief_by_tag['consequence'].append(pb)
                    response_beliefs[premise] = pb
                    sp.belief_id = pb.id
                    sp.update()
            concl, score, _, _ = bset.get_matching_existing_belief(tagged_response[link[1]][1])
            if score is None:
                log.error(f"No match for conclusion: {tagged_response[link[1]][1]} ")
                # should not be here

            elif score < 0:
                log.error("Conclusion is opposite to existing belief:")

            log.debug(f"premises: {str(link[0])}")
            for px in link[0]:
                log.debug(f"response-beliefs for {px}: {response_beliefs[px]}")
            try:
                pbids = [response_beliefs[p].id for p in link[0]]
                support_ids = [response_beliefs[p].support.id for p in link[0]]
            except Exception as e:
                log.error(f"Do not have the premise belief\n" + str(e))
                raise e
            csp = Support.from_reasoning(self.bstore,
                                         0,
                                         support_ids,
                                         {'method': 'llm', 'confidence': link[2]})
            log.debug(f"reasoning support: {csp}")
            if concl is None:
                log.info(f"conclusion {tagged_response[link[1]][1]} not in db")
                concl: Belief = bset.add_bel_from_support(tagged_response[link[1]][1],
                                                       csp)
                belief_by_tag['consequence'].append(concl)
            else:
                csp.belief_id = concl.id
                upd_bel, status = concl.merge_supports(csp, score)
                if status not in (0, 2):  # not (merged or sp wins)
                    log.error(f"conclusion is opposite to existing belief and lost. Redo reasoning")
                    raise ValueError("LLM generated fact opposite to existing belief and eliminated.")
                #msupport = Support.by_merging(concl.id, concl.support, csp, 1.0,
                #                              self.bstore)
                #concl.support = msupport
                belief_by_tag['conclusion'].append(concl)
            if tagged_response[link[1]][0] == 'conclusion':
                the_conclusion = concl
        return the_conclusion, belief_by_tag
