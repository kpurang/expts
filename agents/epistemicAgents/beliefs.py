
import json
from dataclasses import dataclass
from typing import ClassVar
import ollama
import logging
import params
from support import Support
import beliefStore

"""
Classes:
    - Belief. Represents a belief of the agent.
"""

log = logging.getLogger()


@dataclass
class Belief:
    """
    TODO:
    - merge beleifs. 'Bill is happy'. 'Mr smith is happy' are separate beliefs
        but we later find Bill = Mr smith. now need to merge these beliefs.

    A belief:
        - maps to one or more statements
        - has a support object which shows why this belief has been adopted
        - belongs to a beliefSet
    This class is generally used by other classes.

    Attributes:
        - id
        - bset_id   the beliefSet it belongs to
        - text_rep  one of the texts that expresses the belief
        - support   why is it adopted
        - bstore    link to serialization and storage

    Class methods:
        - from_support: create a belief from a support
        - from_sql_row: create belief from sql
        - by_id: get belief given id
        - update_table: update belief sql table

    Instance methods:
        - add_support: add a support to a belief
        - srep_for_plot: concise string representation
    """
    id: int = 0
    bset_id: int = 0        # what set it is in
    text_rep: str = ''      # a representative text for the belieg
    support: Support = None         # support object for this
    bstore: beliefStore.BeliefStore = None

    # cache
    belief_dict: ClassVar[dict] = {}    # id -> [belief, list of updates]

    def srep_for_plot(self):
        """
        returns a string to represent the belief in a plot
        :return: str
        """
        srep = self.text_rep
        if self.support is not None:
            srepp = f"{self.support.info.stype.name}: {srep} | {self.support.confidence:.2f}"
        else:
            srepp = "UNK: {srep} | NaN"
        return srepp

    @classmethod
    def from_support(cls,
                    bstore: beliefStore.BeliefStore,
                    bset,
                    text_rep: str,
                    support: Support,
                     ):
        """
        Generating a belief from the support

        :param bstore: store for beliefs
        :param bset:  which set this belongs to
        :param text_rep: a representative text. there may be many statements that
        map to that belief.
        :param support: what supports adding this belief
        :return: the belief

        **NOTE** this assumes the belief does not already exist. if it does we will
        get duplicate beliefs.
        TODO: possibly check that the beleif does not exist
        All beliefs need to have a support.
        This creates a belieg based on the support. It also sets the belief-id
        of the support.
        """

        log.debug("Belief.from_support")
        id_dist, isNew = bstore.get_similar_stmts(text_rep,
                                                  wide_net=True,    # SHOULD BE TRUE
                                                  max_match=1)
        stmt_id = id_dist[0][0]
        log.debug(f"adding to store stmt {stmt_id}")
        bel = Belief()
        bel.bstore = bstore
        bel.bset_id = bset.id
        bel.text_rep = text_rep
        bel.support = support
        b_id = bel._add_to_store(bstore, stmt_id)
        if b_id is None:
            log.error("Cannot add belief " + text_rep)
            raise ValueError("cannot add belief to store")
        bel.id = b_id
        bel.support.belief_id = bel.id
        cls.belief_dict[bel.id] = [bel]
        bset.beliefs.append(bel.id)
        log.info(f"Created belief {bel.id}: {bel.text_rep}")
        log.info(f"Support: {str(bel.support)}")
        #blog.info(f"in support dict: {Support.support_dict[support.id][0]}")
        #blog.debug(f"len of belief_list = {len(cls.belief_dict)}")
        #blog.debug(f"In beliefs : {str(cls.belief_dict[b_id])}")
        return bel

    @classmethod
    def from_sql_row(cls, row, bstore):
        """
        Instantiate a belief from a sql row.

        :param row: row in the df to deserialize
        :param bstore:
        :return: the belief
        """
        log.debug("Beleif.from_sql_row")
        bel = cls(id=row[0],
                  bset_id=row[2],
                  text_rep=row[1],
                  support = Support.by_id(row[3])
               )
        bel.bstore = bstore
        cls.belief_dict[bel.id] = [bel]
        log.debug(f"Belied from sql: {bel.id}: {bel.text_rep}")
        return bel

    @classmethod
    def by_id(cls, id:int, bstore):
        """ Given an id, return the belief. """
        log.debug("Beleif.by_id")
        if id in cls.belief_dict:
            return cls.belief_dict[id][0]
        else:
            log.debug("Belief not in dict")
            # will raise error if not found. Let error propagate
            row = bstore.get_belief_row_by_id(id)
            bel = cls.from_sql_row(row, bstore)
        return bel


    @classmethod
    def update_table(cls, bstore):
        """
        Update the sql table with the beliefs in memory

        TODO: only update modified ones
        :param bstore:
        :return:
        """
        log.debug("Belief.update_table")
        values = []
        for bid in cls.belief_dict.keys():
            if True: #len(cls.belief_dict[bid][1]) > 1:
                bel = cls.belief_dict[bid][0]
                aval = [bel.text_rep, bel.bset_id,
                        bel.support.id, bel.id]
                values.append(aval)
        if len(values) > 0:
            try:
                sql = """update beliefs set text_rep=?, bset_id=?, support_id=? where id=? """
                print(sql, '\n', values)
                bstore.conn.executemany(sql, values)
                bstore.conn.commit()
            except Exception as e:
                log.error(f"Cannot update beliefs\n{str(e)}")
                raise e
        return True

    def _add_to_store(self, bstore, stmt_id):
        """ add the belief to the bstore.  """
        log.debug("Belief._add_to_store")
        bel_id = None
        try:
            cur = bstore.conn.cursor()
            cur.execute("insert into beliefs(text_rep, bset_id, support_id) values(?, ?, ?)",
                        (self.text_rep, self.bset_id, self.support.id))
            bel_id = cur.lastrowid
            cur.close()
            sql = f"insert into stmt2bel(stmt_id, belief_id, score) values({stmt_id}, {bel_id}, 1.0)"
            log.debug(sql)
            bstore.conn.execute(sql)
            bstore.conn.commit()
        except Exception as e:
            log.error(f"Cannot insert belief in db\n{str(e)}")
            raise e
        return bel_id

    def add_support(self,
                    support: Support,
                    score: float):
        """
        Given a new sypport for a belief, create a new one by merging

        :param source_info: info about the surce
        :param score: how close the original statment is to this one.
        :return:
        """
        if self.bstore is None:
            raise ValueError("self.bstore is None")
        #new_support = Support.from_source(self.id, source_dict, self.bstore)
        merged_support = Support.by_merging(self.id,
                                            self.support,
                                            support,
                                            score,
                                            self.bstore)
        self.support = merged_support
        return True

# =============================

