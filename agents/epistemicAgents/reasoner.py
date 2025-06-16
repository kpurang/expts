from beliefs import Belief
from beliefSet import BeliefSet
from beliefStore import BeliefStore

class Reasoner:
    def __init__(self,
                 bset: BeliefSet,
                 bstore: BeliefStore):
        self.bset = bset
        self.bstore = bstore


    def verify(self,
               bel: Belief,
               max_depth:int):
        pass

    def get_consequences(self,
                         bel: Belief,
                         max_depth: int):
        pass