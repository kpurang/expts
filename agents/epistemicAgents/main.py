import unittest
import os
from beliefs import Belief
from beliefSet import BeliefSet
from beliefStore import BeliefStore #, blog
import logging
import sys

blog = logging.getLogger()
#fh = logging.FileHandler(filename=LOGFILE)
#fh.setLevel(logging.DEBUG)
cw = logging.StreamHandler(sys.stdout)
cw.setLevel(logging.INFO)
formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s',
                              "%m/%d %H:%M:%S")
cw.setFormatter(formatter)
blog.addHandler(cw)
blog.setLevel(logging.INFO)

def main():
    bstore = setup()
    try:
        #test_10_makeBelief(bstore)
        test_20_add_stmt_from_source(bstore)
    except Exception as e:
        blog.error('Dailed test\n', e)
    teardown(bstore)

def test_10_makeBelief(bstore):
    bset = BeliefSet(bstore, 'ROOT', 'top level context')
    print('Bset: ', bset.id)
    support = Support(0, {'source_type': 0}, self.bstore)
    b1 = Belief.from_support(bstore = bstore, bset_id=bset.id,
                             text_rep='Ten rockets were launched today.',
                             support)
    print('MAIN: the belief:\n', b1)
    print(bstore.get_belief_by_id(b1.id))
    bstore.dump_vector_store()
    bstore.dump_table('beliefs')
    bstore.dump_table('bsets')
    bstore.dump_table('supports')

    # print(bstore.get_belief_by_id(b1.id))  Is not in bstore yet
    # bstore.dump_vector_store()  # nothing there

def test_20_add_stmt_from_source(bstore):
    bset = BeliefSet(bstore, 'ROOT', 'top level context')
    blog.info('Adding stmt 1: Bird flu will be the next pandemic')
    bset.add_bel_from_source(text="Bird flu will be the next pandemic.", source_dict=)
    blog.info('Adding stmt 2: Cats like chasing mice')
    bset.add_bel_from_source(text="Cats like chasing mice.", source_dict=)
    blog.info('Adding stmt 3: Felines enjoy chasing rodents.')
    bset.add_bel_from_source(text="Felines enjoy chasing rodents.", source_dict=)
    blog.info('Adding stmt 4: Kitties don\'t like chasing mice.')
    bset.add_bel_from_source(text="Kitties don't like chasing mice.", source_dict=)
    blog.info('Adding stmt 5: Cats like chasing mice')
    bset.add_bel_from_source(text="Cats like chasing mice.", source_dict=)

def setup():
    blog.info('setup')
    bstore = BeliefStore(milvus_file='/tmp/kp/test_mv.db',
                         sqlite_file='/tmp/kp/test_mysql.db')
    print('done setup')
    return bstore

def teardown(bstore):
    bstore.exit()
    #os.remove('/tmp/kp/test_mv.db')
    #os.remove('/tmp/kp/test_mysql.db')
    #os.remove('/tmp/kp/test_mv.db')
    #os.remove('/tmp/kp/test_bel.csv')
    #os.remove('/tmp/kp/test_bset.csv')
    #os.remove('/tmp/kp/test_support.csv')
    print('Done teardown')

# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    main()

# See PyCharm help at https://www.jetbrains.com/help/pycharm/
