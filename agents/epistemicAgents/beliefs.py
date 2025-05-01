
import json
from dataclasses import dataclass
from typing import ClassVar
import ollama
import logging
import params
from support import Support
import beliefStore

blog = logging.getLogger()


@dataclass
class Belief:
    """
    A belief maps onto one or more statements and has a support object that
    shows the origins of the belief and the agent's degree of confidence in it.
    Each belief is in some beliefset
    """
    id: int = 0
    bset_id: int = 0        # what set it is in
    text_rep: str = ''      # a representative text for the belieg
    support: Support = None         # support object for this
    bstore: beliefStore.BeliefStore = None

    # cache
    belief_dict: ClassVar[dict] = {}    # id -> [belief, list of updates]


    @classmethod
    def from_support(cls,
                    bstore: beliefStore.BeliefStore,
                    bset,
                    text_rep: str,
                    support: Support,
                     ):
        """
        construct a new belief given the support etc
        """
        blog.debug("Belief.from_source")
        id_dist, isNew = bstore.get_similar_stmts(text_rep,
                                                  wide_net=True,    # SHOULD BE TRUE
                                                  max_match=1)
        stmt_id = id_dist[0][0]
        bel = Belief()
        bel.bstore = bstore
        bel.bset_id = bset.id
        bel.text_rep = text_rep
        bel.support = support
        b_id = bel._add_to_store(bstore, stmt_id)
        if b_id is None:
            blog.error("Cannot add belief " + text_rep)
            raise ValueError("cannot add belief to store")
        bel.id = b_id
        bel.support.belief_id = bel.id
        bel.bstore = bstore
        cls.belief_dict[bel.id] = [bel, []]
        blog.info(f"Created belief {bel.id}: {bel.text_rep}")
        blog.debug(f"len of belief_list = {len(cls.belief_dict)}")
        return bel

    @classmethod
    def from_sql_row(cls, row, bstore):
        """
        THis is a belief that exists in the db. instantiate it for use
        :param row: row in the df to deserialize
        :param bstore:
        :return: the belief
        """
        blog.debug("Beleif.from_sql_row")
        bel = cls(id=row[0],
                  bset_id=row[2],
                  text_rep=row[1],
                  support = Support.from_sql_row(bstore.get_support_row_by_id(row[3])),
               )
        bel.bstore = bstore
        cls.belief_dict[bel.id] = [bel, []]
        blog.debug(f"Belied from sql: {bel.id}: {bel.text_rep}")
        return bel

    @classmethod
    def by_id(cls, id:int, bstore):
        if id in cls.belief_dict:
            return cls.belief_dict[id][0]
        else:
            row = bstore.get_belief_row_by_id(id)
            bel = cls.from_sql_row(cls, row, bstore)
        return bel


    @classmethod
    def update_table(cls, bstore):
        """
        is that useful?
        :param bstore:
        :return:
        """
        blog.debug("Belief.update_table")
        values = []
        for bid in cls.belief_dict.keys():
            if len(cls.belief_dict[bid][1]) > 1:
                bel = cls.belief_dict[bid][0]
                aval = [bel.text_rep, bel.bset_id,
                        bel.support.id, bel.id]
                values.append(aval)
        if len(values) > 0:
            try:
                sql = """update beliefs set text_rep=?, bset_id=?, support_id=?, where id=? """
                print(sql, '\n', values)
                bstore.conn.executemany(sql, values)
                bstore.conn.commit()
            except Exception as e:
                blog.error(f"Cannot update beliefs\n{str(e)}")
                raise e
                return False
        return True

    def _add_to_store(self, bstore, stmt_id):
        """ add the belief to the bstore:
        """
        blog.debug("Belief._add_to_store")
        bel_id = None
        try:
            cur = bstore.conn.cursor()
            insert_sql = f"""insert into beliefs(text_rep, bset_id, \
            support_id) values('{self.text_rep}', {self.bset_id}, \
            {self.support.id})
            """
            blog.debug(insert_sql)
            cur.execute(insert_sql)
            bel_id = cur.lastrowid
            cur.close()
            sql = f"insert into stmt2bel(stmt_id, belief_id) values({stmt_id}, {bel_id})"
            blog.debug(sql)
            bstore.conn.execute(sql)
            bstore.conn.commit()
        except Exception as e:
            blog.error(f"Cannot insert belief in db\n{str(e)}")
            raise e
        return bel_id

    def add_support(self,
                    support,
                    score):
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
                                            [support, self.support],
                                            score,
                                            self.bstore)
        self.support = merged_support


# =============================

