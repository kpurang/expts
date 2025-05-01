import os
import unittest
from support import Support, SType, SInfo
import beliefs
from beliefStore import BeliefStore
from beliefSet import BeliefSet
from beliefs import Belief

MILVUS_FILE = '/tmp/s_test_milvus.db'
SQLITE_FILE = '/tmp/s_test_sqlite.db'

class TestBeliefStore(unittest.TestCase):


    def setUp(self):
        self.bstore = BeliefStore(MILVUS_FILE, SQLITE_FILE)
        self.bset = BeliefSet(self.bstore, '/', 'root')
        self.sentences = [
        'The cat chased the rat',
        'The cat chased the rat',
        'The cart toppled over',
        'Olive trees produce olives',
        'Sheherazade is next',
        'The cat chased the mouse',
        'The cat chased the mouse',
    ]


    def tearDown(self) -> None:
        #self.bstore.exit()
        os.remove(MILVUS_FILE)
        os.remove(SQLITE_FILE)
        print('Done teardown')


    def test_10_get_similar_stmts(self):
        # max-dist = 1 by default so should get all the inputs to max-match=10
        for s in self.sentences:
            print('Test ', s)
            id_dists = self.bstore.get_similar_stmts(s)
            print('result:')
            self.print_id_dists(id_dists)
            self.bstore.dump_vector_store()
        sql = """select a.stmt_id_1, b.text, a.stmt_id_2, c.text, a.value 
        from stmtdists as a
        join stmts as b on a.stmt_id_1=b.id
        join stmts as c on a.stmt_id_2=c.id
                """
        res = self.bstore.conn.execute(sql)
        for sd in res.fetchall():
            print(sd)

    def test_get_similar_beliefs(self):
        beliefs = []
        for s in self.sentences:
            print('Test ', s)
            bid_dist_txt = self.bstore.get_similar_beliefs(s, self.bset.id)
            if len(bid_dist_txt) == 0:
                print('No matches, adding belief for ', s)
                b = Belief.from_source(self.bstore, self.bset, s, {'label': 'foo'})
                beliefs.append(b)
            else:
                print('has matches')
                for r in bid_dist_txt:
                    print(r)
                if abs(bid_dist_txt[0][0]) < 1e-3:
                    print('adding belief for ', s)
                    b = Belief.from_source(self.bstore, self.bset, s, {'label': 'foo'})


    def print_id_dists(self, id_dists):
            for x in id_dists:
                print(x)

b_tests = unittest.TestSuite()
b_tests.addTests([TestBeliefStore('test_get_similar_beliefs')])


if __name__ == '__main__':
    #unittest.main()
    unittest.TextTestRunner().run(b_tests)