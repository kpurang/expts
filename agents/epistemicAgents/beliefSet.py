
import logging
import params
from beliefs import Belief
import llm_utils
from beliefStore import BeliefStore
from support import Support

blog = logging.getLogger()

class Source:
    """
    Source represents a source of information.
    TODO:
        - move to bstore
        - remoce credibility fro ehre had add bset-source-credibility table
    sources hsould be common to all bsets, but diff bsets can have dif cred
    """
    sources = []            # list {source_type, label, description, credibility}
    label_dict = {}         # dict label -> above + is
    maxid = 0

    @classmethod
    def load_sources(cls, bstore):
        res = bstore.conn.execute("select * from sources")
        sources = res.fetchall()
        maxid = 0
        lst_sources = []
        for s in sources:
            lst_sources.append({'id': s[0], 'source_type': s[1], 'url': s[2],
                                'label': s[3], 'description': s[4],
                                'credibility': s[5]})
            cls.label_dict[s[2]] = lst_sources[-1]
            maxid = s[0] if s[0] > maxid else maxid
        cls.sources = [None] * maxid + params.SOURCE_LEN_ADD
        for s in lst_sources:
            cls.sources[s['id']] = s
        cls.maxid = maxid
        return maxid

    @classmethod
    def add_source(cls,
                   bstore,
                   source_type: int= 0,
                   url: str = '',
                   label: str = '',
                   description: str = '',
                   credibility: float = 0.0,):
        cur = bstore.conn.cursor()
        sql = """insert into sources(source_type, url, label, description, credibility)
        values  (?, ?, ?, ?, ?)
        """
        cur.execute(sql, (source_type, url, label, description, credibility))
        this_id = cur.lastrowid
        cur.close()
        bstore.conn.commit()
        if len(cls.sources) <= this_id:
            cls.sources.extend([None] * params.SOURCE_LEN_ADD)
        sobj = {'id': this_id, 'source_type': source_type, 'label': label,
                'url': url, 'description': description, 'credibility': credibility}
        cls.sources[this_id] = sobj
        cls.label_dict[label] = sobj
        return this_id

    @classmethod
    def get_source_by_Label(cls, label:str):
        if label in Source.label_dict:
            return Source.label_dict[label]
        else:
            blog.warning('No source with this label: ' + label)
            return None


class BeliefSet:
    """
    A beliefSet is a set od beleifs an agent holds. An agent has many belief sets
    which are connected in a graph. Differnet beleifsets can represent the agent's
    view of other agents' beliefs, different assumptions the agent makes, different
    situations it considers etc.

    TODO:
        - develop this
    """

    def __init__(self,
                 bstore: BeliefStore,  # the belief store that includes the bset
                 path: str,  # the string identifying this bset.
                 description: str = None):
        """ if this path exists in the belief store, retrieve its info else
        create a new one"""
        blog.info(f"BeliefSet.init for {path}")
        self.bstore = bstore
        self.id = None
        self.path = path
        self.description = description
        self.beliefs = {}   # id -> belief
        try:
            sql_str = f"select * from bsets where path='{path}';"
            res = bstore.conn.execute(sql_str)
            cands = res.fetchall()
        except Exception as e:
            blog.error(f"Cannt get bset\n{e}")
            raise e
        if len(cands) > 0:
            self.id = cands[0][0]
        else:
            try:
                cur = bstore.conn.cursor()
                cur.execute(f"""insert into bsets (path, description) \
values('{path}', '{description}')""")
                self.id = cur.lastrowid
                cur.close()
                bstore.conn.commit()
            except Exception as e:
                blog.error(f"Cannot get new bset\n{str(e)}")
                raise e
        self.bstore.bset_set.add(self)

    def add_bel_from_source(self, text, source_dict):
        """
        Add a beleif that comnes from a source
        :param text:  the text representing the belief
        :param source_dict:  the source
        :return: None
        """
        support = Support.from_source(0, source_dict, self.bstore)
        return self.add_bel_from_support(text, support)

    def add_bel_from_support(self,
                             text: str,  # the text to set into a belief
                             support: Support,  # information about the source
                             ):
        """ add a belief that has a support
        1. verify whether this or its negation is in the bset
        2. if so merge with the existing belief
        3. else add as a new belief
        """
        blog.debug(f"BeliefSet.add_stmt_from_source: {text}")
        closest_bel, score, stmt_id, stmt_is_new = self.get_matching_existing_belief(text)
        if score is not None and abs(score) > params.LLM_SIM_THRESHOLD:
            # need to merge the new support with the existing one
            closest_bel.add_support(support, score)
            # the text is a differnet representation of an existing belief.
            if stmt_is_new:
                try:
                    sql = f"insert into stmt2bel (stmt_id, belief_id) values({closest_bel.id}, {stmt_id})"
                    self.bstore.conn.execute(sql)
                except Exception as e:
                    blog.error(f"Cannot add stmt2id\n{e}")
                    raise e
            blog.debug(f"updated belief {closest_bel.id} ")
        else:
            # this is a new beleif
            new_bel = Belief.from_support(self.bstore,
                                          bset = self,
                                          text_rep = text,
                                          support = support,
                                        )
            blog.debug(f"BeliefSet added belief {new_bel.id}: {new_bel.text_rep}")

    def get_matching_existing_belief(self,
                                     text: str,
                                     ) -> (Belief, float, int, bool):
        """
        Given a text, find if there is any existing belief that has the same or
        opposite meaning to this. Assume all existing beliefs have been processed
        in the same way and the process is somewhat linear and the beliefs have
        distinct meanings.
        :param text: text representing the belief
        :return: the closest bel, how similar they are, stmt_id, is_new
        """
        blog.debug(f"BeliefSet.get_matchi8ng_existing_belief for {text}")
        matches, stmt_id, is_new = self.bstore.get_similar_beliefs(txt=text, beliefset_id=self.id,
                                                           max_dist=params.SB_THRESHOLD)
        # matches: list of [belief-id, distance, text]
        matches.sort(key=lambda x: x[1], reverse=False)
        closest_bel, closest_text, best_score = None, None, None
        # use llms to more precisely look for same/diff meanings. just want the
        # best one
        blog.debug("Matches")
        for m in matches:
            blog.debug(f"{m[0]}: {m[1]} - {m[2]}")
        for bd in matches:
            blog.debug(f"Considering belief {bd[0]} at {bd[1]}")
            #
            last_belief = None
            score = llm_utils.get_degree_similarity(bd[2], text)
            if (best_score is None) or abs(score) > best_score:
                closest_bel_id = bd[0]
                closest_text = bd[2]
                closest_bel = last_belief
                best_score = abs(score)
        if closest_bel is not None:
            closest_bel = self.get_belief_by_id(closest_bel_id)
            return closest_bel, best_score, stmt_id, is_new
        else:
            return None, None, None, None


    def get_belief_by_id(self, bid):
        """
        gets the belief from the dict or gets from bstore
        :param bid: id of belief
        :return: the belief obj
        """
        blog.debug(f"BeliefSet.get_belief_by_id {bid}")
        if bid not in self.beliefs:
            belief_roe = self.bstore.get_belief_row_by_id(bid)
            belief = Belief.from_sql_row(self.bstore)
            self.beliefs[bid] = belief
        return self.beliefs[bid]


    def get_common(self,
                   bset_2,
                   dist: float,
                   delta_conf: float,  # max distance between confidence to count as same
                   effort: int,
                  ):
        """ return the beliefs common between this set and bset_2
        """
    pass


    def get_inconsistencies(self,
                            bset_2,
                            dist: float,
                            delta_conf: float,
                            effort: int,
                            ):
        """ counterpart of get_common
        """
        pass



    def pprint(self, verb=0):
        rstr = ''
        if self.parent is None:
            rstr += 'ROOT'
        else:
            rstr += self.parent.pprint()
        rstr += ' / '
        if self.agent is not None:
            rstr += self.agent
        if len(self.assumptions) > 0:
            rstr += ' | ' + ', '.join(s for s in self.assumptions)
        return rstr

    # ==== inference ====

    def follows_from_bset(self,
                          p: str,
                          ):
        """
        Does p follow from this belief set? assume it is not in the set.
        Use llms to do a small inference step. llm may use some information
        it knows, but don't expect long derivations here
        :param p:
        :return:
        """
        pass

    def missing_info(self,
                     p: str,
                     ):
        """
        MAYBE
        What is a beleif that would make it easy to infer P?
        :param p:
        :return:
        """

    def get_inconsistent(self,
                         bel: Belief,
                         dist: float,
                         add_int: bool = True,  # are intermediate beliefs added?
                         effort: float = 0,
                         ):
        """ find if there are any beliefs inconsistent with the one given
        maybe will be eliminated
        """
        pass

    def add_source_and_merge(self, closest_bel, score, source_info):
        """
        This will eventually involve statements to track diff versions of the same meaning
        situation: we have a belief and we get something else that is similar
        need to incorporate the new source into the beleif and update the confidence
        NOTE: if we already had this source, the confidence should not change.
        :param closest_bel:
        :param score:
        :param source_info:
        :return:
        """
        pass

