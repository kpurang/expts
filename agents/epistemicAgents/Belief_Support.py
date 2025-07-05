import logging
from support import Support, SType
from beliefs import Belief
import beliefStore

log = logging.getLogger()

class Belief_Support:
    """
    A factory class to create a Belief and its initial Support object together.
    This simplifies the process of creating new beliefs from various sources.
    """

    @classmethod
    def _create_belief_from_support(cls, bstore, bset, text_rep, support):
        """A helper to create the belief once the support is ready."""
        if not support:
            log.error("Failed to create support object.")
            return None, "Failed to create support"

        belief, status = Belief.from_support(
            bstore=bstore,
            bset=bset,
            text_rep=text_rep,
            support=support
        )
        support.belief_id = belief.id
        support.update()
        return belief, status

    @classmethod
    def from_source(cls, bstore: beliefStore.BeliefStore, bset, text_rep: str, source_dict: dict):
        """Creates a belief and support from a source."""
        log.info(f"Creating belief from source for: '{text_rep}'")
        support = Support.from_source(
            belief_id=0,  # Placeholder
            source_dict=source_dict,
            bstore=bstore
        )
        return cls._create_belief_from_support(bstore, bset, text_rep, support)

    @classmethod
    def from_reasoning(cls, bstore: beliefStore.BeliefStore, bset, text_rep: str, supported_by: list, info: dict):
        """Creates a belief and support from a reasoning step."""
        log.info(f"Creating belief from reasoning for: '{text_rep}'")
        support = Support.from_reasoning(
            bstore=bstore,
            belief_id=0, # Placeholder
            supported_by=supported_by,
            info=info
        )
        return cls._create_belief_from_support(bstore, bset, text_rep, support)

    @classmethod
    def from_axiom(cls, bstore: beliefStore.BeliefStore, bset, text_rep: str, source_dict: dict):
        """Creates a belief and support from an axiom."""
        log.info(f"Creating belief from axiom for: '{text_rep}'")
        support = Support.from_axiom(
            belief_id=0, # Placeholder
            source_dict=source_dict,
            bstore=bstore
        )
        return cls._create_belief_from_support(bstore, bset, text_rep, support)

    @classmethod
    def from_query(cls, bstore: beliefStore.BeliefStore, bset, text_rep: str, source_dict: dict):
        """Creates a belief and support from a user query."""
        log.info(f"Creating belief from query for: '{text_rep}'")
        support = Support.from_query(
            belief_id=0, # Placeholder
            source_dict=source_dict,
            bstore=bstore
        )
        return cls._create_belief_from_support(bstore, bset, text_rep, support)

    @classmethod
    def from_llm(cls, bstore: beliefStore.BeliefStore, bset, text_rep: str, source_dict: dict):
        """Creates a belief and support from an LLM."""
        log.info(f"Creating belief from LLM for: '{text_rep}'")
        support = Support.from_llm(
            belief_id=0, # Placeholder
            source_dict=source_dict,
            bstore=bstore
        )
        return cls._create_belief_from_support(bstore, bset, text_rep, support)

    @classmethod
    def from_support(cls, bstore: beliefStore.BeliefStore, bset, text_rep: str, support: Support):
        """Creates a belief from an existing support object."""
        log.info(f"Creating belief from existing support for: '{text_rep}'")
        return cls._create_belief_from_support(bstore, bset, text_rep, support)
