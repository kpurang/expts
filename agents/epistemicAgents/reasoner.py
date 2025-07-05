from beliefs import Belief
from beliefSet import BeliefSet
from beliefStore import BeliefStore

class Reasoner:
    def __init__(self,
                 bstore: BeliefStore):
        self.bstore = bstore


    def verify(self,
               bel: Belief,
			   bset: BeliefSet,
               max_depth:int):
        pass

    def get_consequences(self,
                         bel: Belief,
						 bset: BeliefSet,
                         max_depth: int):
        pass
