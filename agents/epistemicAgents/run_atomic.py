"""
This loads atomic examples and finds justifications for the conclusions.

Atomic examples are processed from the atomic dataset and consist of a subset
of the events with corresponding human generated conclusions from the events.

The events are added to the graph and a LLM is used to derive the conclusions
from the event using commonsense information. That reasoning is added to the
graph and are linked according to which beliefs support which others.

The result is a derivtion graph generated for each conclusion.

"""
import os
import unittest

import params
from support import Support, SType, SInfo
import beliefs
from beliefStore import BeliefStore, BSContext
from beliefSet import BeliefSet
from beliefs import Belief, Support
import logging
import sys
import random
import llm_utils
from sentenceReasoner import SentenceReasoner
import supportGraph
from datetime import datetime
import pandas as pd
import json
import argparse
import regex

# from SO
# https://stackoverflow.com/questions/384076/how-can-i-color-python-logging-output
class CustomFormatter(logging.Formatter):

    grey = "\x1b[38;20m"
    yellow = "\x1b[33;20m"
    red = "\x1b[31;20m"
    bold_red = "\x1b[31;1m"
    reset = "\x1b[0m"
    format = "%(asctime)s - %(name)s - %(levelname)s - %(message)s (%(filename)s:%(lineno)d)"

    FORMATS = {
        logging.DEBUG: grey + format + reset,
        logging.INFO: grey + format + reset,
        logging.WARNING: yellow + format + reset,
        logging.ERROR: red + format + reset,
        logging.CRITICAL: bold_red + format + reset
    }

    def format(self, record):
        log_fmt = self.FORMATS.get(record.levelno)
        formatter = logging.Formatter(log_fmt)
        return formatter.format(record)



log = logging.getLogger()
#fh = logging.FileHandler(filename=LOGFILE)
#fh.setLevel(logging.DEBUG)
cw = logging.StreamHandler(sys.stdout)
cw.setLevel(logging.DEBUG)
formatter = CustomFormatter('%(asctime)s - %(levelname)s - %(message)s',
                              "%m/%d %H:%M:%S")
cw.setFormatter(formatter)
log.addHandler(cw)
log.setLevel(logging.DEBUG)

llog = logging.getLogger('llog')
fh = logging.FileHandler(filename=params.llm_log_fname)
fh.setLevel(logging.INFO)
fh.setFormatter(formatter)
llog.addHandler(fh)
llog.setLevel(logging.INFO)


jlog = logging.getLogger('just_log')
fg = logging.FileHandler(filename=params.just_log_fname)
fg.setLevel(logging.INFO)
fg.setFormatter(formatter)
jlog.addHandler(fg)
jlog.setLevel(logging.INFO)

shi = logging.StreamHandler(sys.stdout)
shi.setLevel(logging.INFO)
shi.setFormatter(formatter)

ilog = logging.getLogger('infoLogger')
ilog.addHandler(shi)
ilog.setLevel(logging.INFO)

logging.getLogger("requests").setLevel(logging.INFO)
logging.getLogger("urllib3").setLevel(logging.INFO)
logging.getLogger("httpcore").setLevel(logging.INFO)


#MILVUS_FILE = '/tmp/s_test_milvus.db'
#SQLITE_FILE = '/tmp/s_test_sqlite.db'

example_file = "/Users/kp/projects/python/projects/agents/data/ex_50_5_0524_16:43.csv"
out_basename = "/Users/kp/projects/python/projects/agents/data/imgs/"
out_responses = "/Users/kp/projects/python/projects/agents/data/working/"

def justify_atomic(example_file = example_file,
                   nevents:int =0,
                   nconcls:int = 0,
                   eclist:list = [],
                   ):  # plot graph for whuch conclusion.
    """
    Justify conclusions from events in the atomic dataset

    :param example_file: processed subset of atomic data
    :param event_idx:   which event to process
    :param add_event:   whether the event is to be added (to be eliminated)
    :param num_conclusions_to_add:  how many conclusions to process
    :return: None

    The atomic dataset consists of conclusions people draw from events. This
    uses a LLM to find a commonsense reasoning process that generates the
    conclusions from the event and generates a support graph fro each conclusion.

    """
    date = datetime.now().strftime('%m%d_%H%M')

    ilog.warning("=================" + date + "========================================")
    jlog.warning("=================" + date + "========================================")
    with BSContext() as bstore:
        bset = BeliefSet(bstore, '/', 'root')
        reasoner = SentenceReasoner(bset, bstore)
        event_concl = get_event_conclusion(example_file, nevents, nconcls,
                                           eclist)
        print(event_concl)
        for ec in event_concl:
            upd_conclusion = justify_concl(bstore,
                                           bset,
                                           reasoner,
                                           ec[0][1],
                                           ec[1][1])
            fname = f"e{ec[0][0]}_c{ec[1][0]}_{date}.png"
            pathname = os.path.join(out_basename, fname)
            supportGraph.plot_derivation(upd_conclusion, bstore, 'graph 1',
                                     pathname, 10)


def get_event_conclusion(example_file:str, nevent:int=0, nconcl:int=0,
                         eclist = []):
    dfa = pd.read_csv(example_file, header=0, index_col=None)
    events = dfa['event'].unique()
    event_concl = []    # list of [[event_idx, event_str], [concl_idx, concl_str]]
    if len(eclist) > 0:
        eclist.sort(key=lambda x: x[0])
        i = 0
        cdf = None
        last_event = None
        while i < len(eclist):
            print(f"processing eclist {eclist[i]}")
            if eclist[i][0] >= len(events):
                print(f"Dont have event {eclist[i][0]}")
                i += 1
                continue
            if eclist[i][0] != last_event:
                last_event = eclist[i][0]
                event = events[eclist[i][0]]
                cdf = dfa[dfa['event'] == event]
            if eclist[i][1] >= len(cdf):
                print(f"DOnt have conclusion {eclist[i][1]}")
                i += 1
                continue
            event_concl.append([[eclist[i][0], event],
                                [eclist[i][1], cdf.iloc[eclist[i][1]]['conclusion']]])
            i += 1
    else:
        if nevent >= len(events):
            print('Dont have that many events')
            event_list = list(range(len(events)))
        else:
            event_list = random.sample(list(range(len(events))), nevent)
        for e in event_list:
            event = events[e]
            cdf = dfa[dfa['event'] == event]
            if nconcl >= len(cdf):
                print('Dont have that many concl')
                concl_list =  list(range(len(cdf)))
            else:
                concl_list = random.sample(list(range(len(cdf))), nconcl)
            for c in concl_list:
                event_concl.append([[e, event],
                                    [c, cdf.iloc[c]['conclusion']]])
    return event_concl


def add_event(bstore: BeliefStore, bset: BeliefSet, event: str):
    support = Support.from_axiom(0, {'source': 'atomicDB'}, bstore)
    event_0 = Belief.from_support(bstore,
                                  bset,
                                  event,
                                  support)
    print('Support ', support)
    print('Belief ', event_0)
    return event_0

def justify_concl(bstore: BeliefStore, bset:BeliefSet, reasoner:SentenceReasoner,
                  event: Belief, conclusion: str):
    support = Support.from_query(0, {'source': 'atomicDB'}, bstore)
    conclusion_bel, _, _, _ = bset.get_matching_existing_belief(conclusion)
    if conclusion_bel is None:
        conclusion_bel = Belief.from_support(bstore,
                                         bset,
                                         conclusion,
                                         support, )
    event_bel, _, _, _ = bset.get_matching_existing_belief(event)
    if event_bel is None:
        log.info('Event does not exist, addint it')
        e_support = Support.from_axiom(0, {'source': 'atomidDb'}, bstore)
        event_bel = Belief.from_support(bstore, bset, event, e_support, )
    #print('Support ', support)
    #print('Belief ', conclusion)
    # p = reasoner.verify(conclusion)
    #upd_conclusion = reasoner.justify_with_llm([event], conclusion)
    upd_conclusion = reasoner.verify_by_llm(conclusion_bel,
                                            [event_bel],
                                            max_depth=1,
                                            add_matches = False,
                                            only_premises = False,
                                            )
    return upd_conclusion

    #os.remove(MILVUS_FILE)
    #os.remove(SQLITE_FILE)

def get_llm_justifications(efname=example_file,
                           nevent=10,
                           nconcl = 5,
                           outfname=None):
    """
    NOT USED NO MORE

    runs the llm to find reasom=n something is true. just does strings
    :param efname: file name for events
    :param nevent: number of events to process
    :param nconcl: number of conclusions per event
    :param outfname: file for output
    :return: none

    THis just runs the query on the llm (mistral) to get response. use that
    to develop the parser and the support generator
    """
    date = datetime.now().strftime('%m%d_%H%M')
    results = []
    dfa = pd.read_csv(efname, header=0, index_col=None)
    events = dfa['event'].unique()
    e2p = random.sample(range(len(events)), nevent)
    for e in e2p:
        event = events[e]
        print(f"processing event {e}: {event}")
        df = dfa[dfa['event'] == event]
        if len(df) > nconcl:
            concls = random.sample(range(len(df)), nconcl)
        else:
            concls = list(range(len(concls)))
        for c in concls:
            conclusion = df.iloc[c]['conclusion']
            response, facts, assumptions, concl, infer_steps = llm_utils.do_backward_step(
                query=conclusion,
                fact_lst=[event],
                prompt_type='any'
            )
            results.append([[event], conclusion, response['response']])
    if outfname is None:
        outfname = f"reponses_{nevent}_{nconcl}_{date}.jsonl"
        outpath = os.path.join(out_responses, outfname)
        with open(outpath, 'w') as ox:
            json.dump(results, ox, indent=4)

def parse_eclist(str_eclist:str = ''):
    """
    parses these into list os index pairs
    :param str_eclist: list of form e1-c1 e2-c2 etc
    :return: list of pairs of ints
    """
    rv = []
    ecs = [x.strip() for x in str_eclist]
    ec_re = regex.compile('(\d+)-(\d+)')
    for ec in ecs:
        print(ec)
        m = ec_re.match(ec)
        if m is None:
            print('cannot parse ' + ec)
            continue
        else:
            rv.append([int(m[1]), int(m[2])])
    return rv

if __name__ == '__main__':
    # handle args
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--example_file', type=str, default=example_file,
                    help='File containing processed atomic examples')
    ap.add_argument('--nevents', type=int, default=0,
                    help="Number of randomly picked events to process.")
    ap.add_argument('--nconclusions', type=int, default=0,
                    help="Max number of randomly picked conclusions per event.")
    ap.add_argument('--eclist', nargs='*', type=str, default='',
                    help="A list of event-concl to process. Eg '0-3 0-6 1-2'")
    args = ap.parse_args()
    if args.nevents == 0 and args.eclist == '':
        print("Nothing to do")
        exit()
    eclist = []
    print('args.eclist ', (a_ecl := args.eclist))
    print(a_ecl)
    if a_ecl != []:
        eclist = parse_eclist(a_ecl)
    justify_atomic(example_file=example_file,
                   nevents = args.nevents,
                   nconcls = args.nconclusions,
                   eclist = eclist,
                   )

