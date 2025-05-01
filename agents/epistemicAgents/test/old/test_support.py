import os
import unittest
from support import Support, SType, SInfo
import beliefs
import beliefStore
#from beliefs import Belief

MILVUS_FILE = '/tmp/s_test_milvus.db'
SQLITE_FILE = '/tmp/s_test_sqlite.db'

class TestSupport(unittest.TestCase):

    def setUp(self) -> None:
        self.bstore = beliefStore.BeliefStore(milvus_file=MILVUS_FILE,
                                              sqlite_file=SQLITE_FILE)

    def tearDown(self) -> None:
        self.bstore.exit()
        os.remove(MILVUS_FILE)
        os.remove(SQLITE_FILE)
        print('----all done----')


    def test_create(self):
        self.s_dict = {}
        s1 = Support.from_source(0, {'source_id': 0}, self.bstore)
        print('s1: ', s1)
        #self.assertEqual(True, False)  # add assertion here


if __name__ == '__main__':
    unittest.main()
