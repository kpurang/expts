import os
import params
from beliefs import Belief
from beliefStore import BeliefStore
from beliefSet import BeliefSet
from Belief_Support import Belief_Support
from sentenceReasoner import SentenceReasoner
from supportGraph import plot_derivation
from datetime import datetime


class BeliefSupport:

    def __init__(self, name:str='test'):
        """

        :param name: a tag for this run
        """
        print(name)
        print(params.BASEDIR)
        self.basedir = os.path.join(params.BASEDIR, name)
        self.vector_db_file = os.path.join(self.basedir, 'dbs/vector.db')
        self.rel_db_file = os.path.join(self.basedir, 'dbs/rel.db')
        self.img_dir = os.path.join(self.basedir, 'imgs')
        if not os.path.exists(self.basedir):
            os.makedirs(self.basedir)
            os.makedirs(os.path.join(self.basedir, 'dbs'))
        self.bstore = BeliefStore(self.vector_db_file, self.rel_db_file)
        self.bset = BeliefSet(self.bstore, 'root')
        self.reasoner = SentenceReasoner(self.bstore)


    def list_bsets(self):
        return self.bstore.list_bsets()

    def assert_belief(self,
                      text: str,
                      confidence: float = 1.0):
        bel, _ = Belief_Support.from_axiom(self.bstore,
                                           self.bset,
                                           text,
                                           {'source': 'user'}
                                           )
        bel.support.confidence = confidence
        return bel

    def generate_query(self,
                       text):
        query, _ = Belief_Support.from_query(self.bstore,
                                             self.bset,
                                             text,
                                             {'source': 'user'})
        return query

    def get_belief_confidence(self,
                              text_rep: str,
                              belief_id: int,
                              premise_ids: list[int],
                              verify_even_if_exists: bool = False):
        query = Belief.by_id(belief_id, self.bstore)
        premises = [Belief.by_id(x, self.bstore) for x in premise_ids]
        bel = self.reasoner.verify_by_llm(query,
                                          self.bset,
                                          premises,
                                          max_depth=1,
                                          add_matches=True,
                                          only_premises=False,
                                          fail_on_conclusion=True,
                                          )
        print(bel.support.confidence)
        return bel

    def write_derivation_graph(self,
                               bel: Belief,
                               idstr: str):
        dt = datetime.now().strftime('%m%d_%H%M')
        pngFname = os.path.join(self.img_dir, f"{idstr}_{dt}.png")
        plot_derivation(bel,
                        self.bstore,
                        'Reasoning',
                        pngFname,
                        depth=20)



    def exit(self):
        self.bstore.exit()


class BeliefSupportContext:
    def __init__(self, dir):
        self.dir = dir

    def __enter__(self):
        BeliefSupportContext.bs = BeliefSupport(self.dir)
        return BeliefSupportContext.bs

    def __exit__(self, type, value, traceback):
        BeliefSupportContext.bs.exit()


