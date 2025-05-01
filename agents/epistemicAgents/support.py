

import json
from dataclasses import dataclass, field
from typing import ClassVar
import logging
import params
from enum import Enum
import numpy as np
import globals

# logging
blog = logging.getLogger()


# =======================
class BeliefStore:
    pass

class Belief:
    pass

class Support:
    pass

def limiter(x):
    # keeps things in range +1 -1
    s = 1 / (1 + np.exp(-x*2))
    return (s * 2) - 1

class SType(Enum):
    from_source = 1
    from_inference = 2
    from_merge = 3
    from_axiom = 4
    from_query = 5
    unknown = 999

@dataclass
class SInfo:
    """
    This provides inforation about the source of the associated belief
    TODO: perhaps subclass this according to stype and each having expected info slots.
    for now leave the info as a dict
    if stype is from_source, info has to have a slot: label
    """
    stype: SType = SType.unknown
    info: dict = field(default_factory=dict)

    @classmethod
    def deserialize(cls, ser):
        deser = json.loads(ser)
        si = cls()
        si.stype = SType(deser['stype'])
        si.info = deser['info']
        return si

    def serialize(self):
        flat = {'stype': self.stype.value, 'info': self.info}
        return json.dumps(flat)


@dataclass
class Support:
    """
    Supports track the reasons a belief has the confidence it has. these are
    immutable. If new info comes in, a new support is generated that depends on
    the previous one and the new info.
    Supports form a tree which is used for computing the confidence and for
    propagating changes
    """
    id: int = 0
    belief_id: int = 0      # the belief this support is about
    confidence: float = 0   # [-1, 1]
    method:str = ''         # easier to search
    info: SInfo = None      # info is it based on is any. should have type
    supported_by: list[int] = field(default_factory=list)   # ids of supports who support this
    supports: list[int] = field(default_factory=list)       # ids of supports this supports.

    support_dict: ClassVar[dict] = {}   # id -> [sp_obj, list upd fields]

    def __str__(self):
        sb_str = ', '.join(str(s) for s in self.supported_by)
        ss_str = ', '.join(str(s) for s in self.supports)
        rv = f"""id: {self.id}, belief_id: {self.belief_id}, confidence: {self.confidence}\
 method: {self.method}, info: {str(self.info)}
supported_by: {sb_str}, 
supports: {ss_str}"""
        return rv

    @classmethod
    def generic(cls,
                bstore,
                belief_id:int,
                stype: SType,
                info: dict={},
                supported_by: list=[],
                supports: list=[],
                ):
        sp = cls()
        sp.belief_id = belief_id
        sp.info = SInfo(stype=stype, info=info)
        sp.supported_by = supported_by
        sp.supports = supports
        sp.compute_confidence()
        sp_id = sp._add_to_store(bstore)
        if sp_id is None:
            blog.error("Cannot create suport.")
            raise ValueError("not found id")
        sp.id = sp_id
        cls.support_dict[sp.id] = [sp, []]
        blog.debug(f"support from source: {sp.id} for {sp.belief_id}, confidence: {sp.confidence}")
        return sp

    @classmethod
    def from_source(cls, belief_id, source_dict, bstore):
        stype = SType.from_source
        return cls.generic(bstore, belief_id, stype, source_dict, [], [])

    @classmethod
    def from_axiom(cls, belief_id, source_dict, bstore):
        stype = SType.from_axiom
        return cls.generic(bstore, belief_id, stype, source_dict, [], [])

    @classmethod
    def from_query(cls, belief_id, source_dict, bstore):
        stype = SType.from_query
        return cls.generic(bstore, belief_id, stype, source_dict, [], [])

    @classmethod
    def from_reasoning(cls, bstore, belief_id, supported_by, info):
        stype = SType.from_inference
        return cls.generic(bstore, belief_id, stype, info, supported_by, [])

    @classmethod
    def from_sql_row(cls, row):
        blog.debug("Support.from_sql_row")
        sp = cls()
        sp.id = row[0]
        sp.belief_id = row[1]
        sp.confidence = row[2]
        sp.method = row[3]
        sp.info = SInfo.deserialize(row[4])
        sp.supported_by = json.loads(row[5])
        sp.supports = json.loads(row[6])
        cls.support_dict[sp.id] = [sp, []]
        blog.debug(f"support from db: {sp.id} for {sp.belief_id}, confidence: {sp.confidence}")
        return sp

    @classmethod
    def by_id(cls, sid):
        if sid in cls.support_dict:
            return cls.support_dict[sid][0]
        else:
            row = globals.bstore.get_support_row_by_id(sid)
            return cls.from_sql_row(row)

    @classmethod
    def by_merging(cls,
                   belief_id: int,
                   old_support: Support,
                   new_support: Support,
                   score: float,
                   bstore: BeliefStore):
        """
        TODO: add merge method ordering as parameter
        generatees a new support by merging the new to the old
        assume the support has a type field
        There will eventually be multiple cases of supports to merge
        :param old_support: supports to merge
        :param new_support: new one
        :param score: how well the new text matches the belief. polarity is important
        :param bstore: to store the new support
        :return: nothing
        """
        blog.debug("Support.by_merging")
        sp = cls()
        sp.belief_id = belief_id
        sp.info = SInfo(stype=SType.from_merge, info={'similarity_score': score})
        prev_conf = old_support.confidence
        sp.supported_by = [old_support.id, new_support.id]
        sp.supports = []
        sp.method = 'by_merging'
        if old_support.confidence * new_support.confidence < 0:
            old_str = Belief.by_id(old_support.belief_id).text_rep
            new_str = Belief.by_id(new_support.belief_id).text_rep
            blog.warning(f"Contradiction between\n{old_str}\n{new_str}")
            sp.resolve_contradiction()
        else:
            sp.compute_confidence()    # confidence is already assigned
        sp_id = sp._add_to_store(bstore)
        if sp_id is None:
            return
        sp.id = sp_id
        if sp.confidence != prev_conf:
            # propagate the change to the beliefs supported
            for s in old_support.supports:
                print('TODO: propagate support')
                # TODO: update
        cls.support_dict[sp.id] = [sp, []]
        old_support.supports.append(sp.id)
        new_support.supports.append(sp.id)
        blog.debug(f"support by merging: {sp.id} for {sp.belief_id}, confidence: {sp.confidence}")
        return sp

    # methods to compute resulting support.

    def compute_confidence(self, parms = {}):
        # TODO: an array of methods
        if self.info.stype == SType.from_source:
            self.confidence = self.get_source_confidence(self.info.info)
        elif self.info.stype == SType.from_merge:
            support_0 = Support.support_dict[self.supported_by[0]][0]
            support_1 = Support.support_dict[self.supported_by[1]][0]
            if support_0.info.stype == SType.from_source and support_1.info.stype == SType.from_source:
                self.combine_cred(support_0, support_1, self.info.info['similarity_score'])
                self.method += '.combine_cred'
            else:  # default case
                self.sum_confidence(support_0, support_1, self.info.info['similarity_score'])
                self.method += '.sum_confidence'
        elif self.info.stype == SType.from_inference:
            if len(self.supported_by) > 0:
                inst_sby = [Support.by_id(sid) for sid in self.supported_by]
                print('supported by ', self.supported_by)
                for s in inst_sby:
                    print(s)
                self.confidence = min([abs(s.confidence) for s in inst_sby])
            else:
                blog.warning('compute confidence, reasoning case: No supported_by')
                self.confidence = 0.0
        elif self.info.stype == SType.from_axiom:
            self.confidence = limiter(1)
        elif self.info.stype == SType.from_query:
            self.confidence = 0
        else:
            # there can be more than one here. need to modify sum
            # should not be here. pick lowest absolute
            blog.warning(f"compute-confidence, unknown type: {self.info.stype}")
            inst_sby = [Support.by_id(sid) for sid in self.supported_by]
            self.confidence = min([abs(s.confidence) for s in inst_sby])
        return self.confidence

    def combine_cred(self,
                      old_support,
                      new_support,
                      score,
                      ):
        # pick whichever source has higher credibility.
        # assume all from_source have info that has a label field
        blog.debug("Merge_by_cred")
        blog.debug('old ' + str(old_support))
        blog.debug('new ' + str(new_support))
        assert old_support.info.stype == SType.from_source
        assert new_support.info.stype == SType.from_source
        score_pol = -1 if score < 0 else 1
        try:
            s0 = old_support.info.info['label']
            s0_c = old_support.confidence
        except:
            blog.warning(f"belief {old_support.belief_id} has no confidence or label")
            s0_c = params.source_credibility['default']
        try:
            s1 = new_support.info.info['label']
            s1_c = new_support.confidence
        except:
            blog.warning(f"belief {new_support.belief_id} has no confidence")
            s1_c = params.source_credibility['default']
        blog.info(f"combining {s0_c}, {s1_c}")
        if abs(s0_c) > abs(s1_c):
            self.confidence = s0_c
        elif abs(s1_c) > abs(s0_c):
            self.confidence = s1_c * score_pol
        else:
            self.confidence = new_support.confidence
        self.info.info['merge_method'] = 'more_credible_source'
        print('result support: ', str(self))
        return True

    def sum_confidence(self,
                      old_support,
                      new_support,
                      score,
                    ):
        blog.debug("Default merge: sum and limit")
        score_pol = -1 if score < 0 else 1
        self.confidence = limiter(old_support.confidence + new_support.confidence * score_pol)
        self.info.info['method'] = 'sum_confidence'  # already have stype
        return True

    def resolve_contradiction(self):
        """
        this new support depends on 2 supports that are opposite polarity
        cases:
        by-source: if both from source, then pick the more credibel ont
        by-specificity: if one support uses more specific info than the other, pick it
        by-add: just add is all up
        :return: None. confidence is updated
        """
        return self.compute_confidence()

    @classmethod
    def update_table(cls, bstore):
        blog.debug("Support.update_table")
        upds = []
        for sp_id in cls.support_dict.keys():
            if len(cls.support_dict[sp_id][1]) >= 0:
                # keep it simple for noe
                sp = cls.support_dict[sp_id][0]
                upds.append([sp.belief_id, sp.confidence, sp.method,
                             sp.info.serialize(), json.dumps(sp.supported_by),
                             json.dumps(sp.supports), sp.id])
        try:
            bstore.conn.executemany("""update supports set belief_id=?, \
                                    confidence=?, method=?, info=?, \
                                    supported_by=?, supports=? where id=?""", upds)
            bstore.conn.commit()
        except Exception as e:
            blog.error(f"Cannot update supports\n{str(e)}")
            raise e
        return False


    def get_source_confidence(self, info):
        if 'source_id' in info and info['source_id'] in params.source_credibility:
            return limiter(params.source_credibility[info['source_id']])
        else:
            return limiter(params.source_credibility['default'])

    def _add_to_store(self, bstore):
        """ add the belief to the bstore: the dataframe and the vector db
        SUPPORT_COLS = ['id', 'belief_id', 'confidence', 'method', 'info', 'support_ids']
        """
        blog.debug("Support._add_to_store")
        sp_id = None
        try:
            cur = bstore.conn.cursor()
            insert_sql= f"""insert into supports (belief_id, confidence, method, info, \
              supported_by, supports) \
              values({self.belief_id}, {self.confidence}, '{self.method}', '{self.info.serialize()}', \
              '{json.dumps(self.supported_by)}', '{json.dumps(self.supports)}')   """
            blog.info(insert_sql)
            print(insert_sql)
            cur.execute(insert_sql)
            sp_id = cur.lastrowid
            cur.close()
            bstore.conn.commit()
        except Exception as e:
            blog.error(f"Cannot write to support table.\n{str(e)}")
            raise e
        return sp_id
