import unittest
import os
from beliefs import Belief, Support
from beliefSet import BeliefSet
from beliefStore import BeliefStore, blog



class TestBelief(unittest.TestCase):
    def setUp(self):
        blog.info('setup')
        self.bstore = BeliefStore(milvus_file='/tmp/test_mv.db',
                                  sqlite_file='/tmp/test_mysql.db')
        print('done setup')

    def tearDown(self):
        self.bstore.exit()
        os.remove('/tmp/test_mv.db')
        os.remove('/tmp/test_mysql.db')
        print('Done teardown')

    def test_10_makeBelief(self):
        bset = BeliefSet(self.bstore, 'ROOT', 'top level context')
        print('Bset: ', bset.id)
        b1 = Belief.from_source(bstore = self.bstore, bset_id=bset.id,
                                text_rep='test text one',
                                source={'source_type': 0})
        print('the beief:\n', b1)
        print(self.bstore.get_belief_by_id(b1.id))
        self.bstore.dump_vector_store()
        self.bstore.dump_table('beliefs')
        self.bstore.dump_table('bsets')
        self.bstore.dump_table('supports')


if __name__ == '__main__':
    unittest.main()
