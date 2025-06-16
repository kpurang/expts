"""
Generates events and conclusions from these from the atomic dataset

Output is a csv:[event, conclusion]
This is meant to be used to test the reasoning and support structures with
commonsense examples that are more realistic than the ones up to now.
"""
import os
import pandas as pd
import re
from datetime import datetime
import random
import json
import Levenshtein


BASEDIR = '/Users/kp/projects/datasets/NL/atomic'
# INFILE has randomly picked 100 events among those without a blank placeholder
# from the atomic dataset
INFILE = os.path.join(BASEDIR, 'atomic_100_1.csv')
OUTDIR = '/Users/kp/projects/python/projects/agents/data'
# NAMEFILE has 3 cols: fname, gender, number
NAMEFILE = '/Users/kp/projects/datasets/NL/ssa_names/yob1880.csv'

DIM_SENTENCES = {
    'xIntent': lambda x, y, concl: f"{x} did it {concl}.",
    'xNeed': lambda x, y, concl: f"{x} needed {concl} beforehand.",
    'xAttr': lambda x, y, concl: f"{x} is {concl}.",
    'xEffect': lambda x, y, concl: f"{x} {concl}.",
    'xReact': lambda x, y, concl: f"{x} felt {concl}.",
    'xWant': lambda x, y, concl: f"{x} now wants {concl}.",
    'oReact': lambda x, y, concl: f"{y} fely {concl}.",
    'oEffect': lambda x, y, concl: f"{y} {concl}.",
    'oWant': lambda x, y, concl: f"{y} now wants {concl}.",
}

min_levenshtein_ratio = 0.8

def gen_examples(num_events: int = 10,
                 max_dim_conclusions: int = 5,
                 dims_excluded: list[str] = []):
    """
    :params num_events: number of events to process
    :params max_dim_conclusions: max concludions used for each dimension
    :params dims_excluded list of dimensions excluded
    """
    event_df = pd.read_csv(INFILE, index_col=None, header=0, dtype='string')
    names_df = pd.read_csv(NAMEFILE, index_col=None, header=None,
                           names=['name', 'gender', 'number'])
    event_df = event_df.sample(n=num_events)
    for col in event_df.columns:
        if col in dims_excluded:
            event_df = event_df.drop(col, axis=1)
    results = []
    cols = event_df.columns
    for i, row in event_df.iterrows():
        results.extend(process_row(row, cols, names_df, max_dim_conclusions))
    now_str = datetime.now().strftime('%m%d_%H:%M')
    outfname = os.path.join(OUTDIR, f"ex_{num_events}_{max_dim_conclusions}_{now_str}.csv")
    result_df = pd.DataFrame(results, columns=['event', 'ctype', 'conclusion'])
    result_df.to_csv(outfname, header=True, index=False)
    print(f"Written {outfname}")

def process_row(row, cols, names_df, max_dim_conclusions):
    results = []
    num_names = len(names_df)
    event = row['event']
    x, y = None, None
    if 'PersonX' in event:
        x = names_df.iloc[random.randint(0, num_names)]['name']
        event = event.replace('PersonX', x)
    if 'PersonY' in event:
        # assume we will not pick the same name again
        y = names_df.iloc[random.randint(0, num_names)]['name']
        event = event.replace('PersonY', y)
    print('========\nEvent: ', event)
    for col in cols[1:]:
        concls = get_for_dim(row[col], max_dim_conclusions)
        for concl in concls:
            print('raw concl: ', concl)
            if x is not None:
                concl = concl.replace('PersonX', x)
            if y is not None:
                concl = concl.replace('PersonY', y)
            else:
                y = 'others'
            if col in ['xNeed', 'xWant', 'oWant']:
                if not concl.startswith('to '):
                    concl = 'to ' + concl
            #print(DIM_SENTENCES[col](x, y, concl))
            #print(f := DIM_SENTENCES[col])
            #print(f('foo', 'bar', 'baz'))
            #print(x, ' ', y, ' ', concl)
            results.append([event, col, DIM_SENTENCES[col](x, y, concl)])
            print(results[-1])
    return results



def get_for_dim(examples_k, max_num):
    clean = []
    examples = json.loads(examples_k)
    for x in examples:
        is_in = False
        for y in clean:
            if Levenshtein.ratio(x, y) >= min_levenshtein_ratio:
                is_in = True
                break
        if not(x == 'none' or is_in ):
            clean.append(x)
    print('clean: ', clean)
    if len(clean) <= max_num:
        return clean
    else:
        return random.sample(clean, max_num)


gen_examples(50, 5, [])

