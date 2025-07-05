import os
import unittest
from support import Support, SType, SInfo
import beliefs
from beliefStore import BeliefStore
from beliefSet import BeliefSet
from beliefs import Belief, Support
from BeliefSupport import Belief_Support
import logging
import sys
import random
import llm_utils
from sentenceReasoner import SentenceReasoner
import supportGraph
import utils.nl_utils  as nl_utils

blog = logging.getLogger()
#fh = logging.FileHandler(filename=LOGFILE)
#fh.setLevel(logging.DEBUG)
cw = logging.StreamHandler(sys.stdout)
cw.setLevel(logging.DEBUG)
formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s',
                              "%m/%d %H:%M:%S")
cw.setFormatter(formatter)
blog.addHandler(cw)
blog.setLevel(logging.DEBUG)

MILVUS_FILE = '/tmp/s_test_milvus.db'
SQLITE_FILE = '/tmp/s_test_sqlite.db'


class TestBelief(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        print('Belief test setup')
        #blog.info('setup')
        cls.bstore = BeliefStore(milvus_file=MILVUS_FILE,
                                  sqlite_file=SQLITE_FILE)
        print('done setup')

    @classmethod
    def tearDownClass(cls):
        cls.bstore.exit()
        os.remove(MILVUS_FILE)
        os.remove(SQLITE_FILE)
        print('Done teardown')

    def runTest(self):
        self.test_10_makeBelief()

    def test_10_makeBelief(self):
        bset = BeliefSet(self.bstore, 'ROOT', 'top level context')
        print('Bset: ', bset.id)
        the_text = 'test text one'
        stmt_id = self.bstore.get_stmt_id(the_text)
        print('stmt id ', stmt_id)
        b1, _ = Belief_Support.from_source(bstore = self.bstore, bset=bset,
                                text_rep=the_text,
                                source_dict={'source_type': 0})
        print('the beief:\n', b1)
        print(Belief.from_sql_row(self.bstore.get_belief_row_by_id(b1.id), self.bstore))
        self.bstore.dump_vector_store()
        self.bstore.dump_table('beliefs')
        self.bstore.dump_table('bsets')
        self.bstore.dump_table('supports')

class TestSupport(unittest.TestCase):

    def runTest(self):
        print('running 10')
        self.test_10_from_source()
        print('running 20')
        self.test_20_by_merging()

    @classmethod
    def setUpClass(cls) -> None:
        print('test support setup')
        cls.num_s = 6
        cls.bstore = BeliefStore(milvus_file=MILVUS_FILE,
                                              sqlite_file=SQLITE_FILE)
        cls.supports = []

    @classmethod
    def tearDownClass(cls) -> None:
        print('---teardown---')
        info = cls.bstore.exit(dump_tables=['supports'])
        print(info)
        #print('---supports---')
        #print(cls.bstore.dump_table('supports'))
        os.remove(MILVUS_FILE)
        os.remove(SQLITE_FILE)
        print('----all done----')


    def test_10_from_source(self):
        for i in range(1, self.num_s+1):    # to get ids from 1
            s = Support.from_source(i, {'source_id': i}, self.bstore)
            self.supports.append(s)
            print('new support ', str(s))
        #self.assertEqual(True, False)  # add assertion here

    def test_20_by_merging(self):
        print('start test-20')
        for i in range(0, self.num_s, 2):
            print('merging\n', self.supports[i], '\n', self.supports[i+1])
            s = Support.by_merging(self.supports[i].belief_id,
                                   self.supports[i+1],
                                   self.supports[i],
                                   random.random(),
                                   self.bstore
                                   )
            print("new support ", str(s))
        # verify num of new supports, supported_by, supports


class TestBeliefStore(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        print('test beliefstore setup')
        cls.bstore = BeliefStore(MILVUS_FILE, SQLITE_FILE)
        cls.bset = BeliefSet(cls.bstore, '/', 'root')
        cls.sentences = [
        'The cat chased the rat',
        'The cat chased the rat',
        'The cart toppled over',
        'Olive trees produce olives',
        'Sheherazade is next',
        'The cat chased the mouse',
        'The cat chased the mouse',
    ]

    @classmethod
    def tearDown(cls) -> None:
        #self.bstore.exit()
        print('===vector store===')
        cls.bstore.dump_vector_store()
        print('---Beliefs---')
        print(cls.bstore.dump_table('beliefs'))
        print('---stmts---')
        print(cls.bstore.dump_table('stmts'))
        print('---stmtdists---')
        print(cls.bstore.dump_table('stmtdists'))
        print('---stmt2bel---')
        print(cls.bstore.dump_table('stmt2bel'))
        print('---supports---')
        print(cls.bstore.dump_table('supports'))
        os.remove(MILVUS_FILE)
        os.remove(SQLITE_FILE)
        print('Done teardown')

    def runTest(self):
        self.test_similar_milvus()

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
            bid_dist_txt, the_stmt_id, is_new = self.bstore.get_similar_beliefs(s, self.bset.id)
            if len(bid_dist_txt) == 0:
                print('No matches, adding belief for ', s)
                b, _ = Belief_Support.from_source(self.bstore, self.bset, s, {'label': 'foo'})
                beliefs.append(b)
            else:
                print('has matches')
                for r in bid_dist_txt:
                    print(r)
                if abs(bid_dist_txt[0][1]) > 1e-2:
                    print('adding belief for ', s)
                    b, _ = Belief_Support.from_source(self.bstore, self.bset, s, {'label': 'foo'})
                else:
                    print('Existing belief is close enough')

    def test_similar_milvus(self):
        _, _ = self.bstore.get_similar_stmts('Strawberries are red.')
        _, _ = self.bstore.get_similar_stmts('Dogs like digging.')
        _, _ = self.bstore.get_similar_stmts('Alpha Centauri is very far away.')
        _, _ = self.bstore.get_similar_stmts('JWST is at a Lagrange point.')
        idd, isnew = self.bstore.get_similar_stmts('Jack is a bird.', wide_net=True)
        print('is new ', isnew)
        for x in idd:
            print(x)
        idd, isnew = self.bstore.get_similar_stmts('Birds fly.', wide_net=True)
        print('is new ', isnew)
        for x in idd:
            print(x)
        idd, isnew = self.bstore.get_similar_stmts('Jack flies.', wide_net=True)
        print('is new ', isnew)
        for x in idd:
            print(x)
        idd, isnew = self.bstore.get_similar_stmts('Cherries are red.', wide_net=True)
        print('is new ', isnew)
        for x in idd:
            print(x)


    def print_id_dists(self, id_dists):
            for x in id_dists:
                print(x)

class Test_LLMUtils(unittest.TestCase):

    def runTest(self):
        #print('Get degree similarity')
        #self.test_get_degree_similarity()
        #print('\n\nbackward_step')
        #self.test_backward_step_0()
        #self.test_quick_parse()
        #self.test_dereference()
        self.test_str_compare()
        #self.test_get_llm_msg()

    def test_get_degree_similarity(self):
        s1 = 'It is raining.'
        s2 = 'It is not raining.'
        s3 = 'It is drizzling.'
        sim = llm_utils.get_degree_similarity(s1, s1)
        print(f"sim: {sim}: {s1} | {s1}")
        self.assertGreater(sim, 0.9)
        sim = llm_utils.get_degree_similarity(s1, s2)
        print(f"sim: {sim}: {s1} | {s2}")
        self.assertLess(sim, -0.9)
        sim = llm_utils.get_degree_similarity(s1, s3)
        print(f"sim: {sim}: {s1} | {s3}")
        self.assertGreater(sim, 0.5)
        sim = llm_utils.get_degree_similarity(s2, s3)
        print(f"sim: {sim}: {s2} | {s3}")
        self.assertLess(sim, -0.5)

    def test_backward_step_0(self):
        query = "Jack flies."
        facts = ["Jack is a bird.", "Birds fly."]
        print('Query: ', query)
        print('Facts: ', facts)
        response, facts, assumptions, concl = llm_utils.backward_step(query, facts)
        print('response\n', response)
        print('facts\n', facts)
        print('assumptions\n', assumptions)
        print('concl\n', concl)
        return True

    def test_quick_parse(self):
        responses = ['1. Jack is a bird\n    (definition of Jack)\n\n   2. Birds fly.\n    (general knowledge fact)\n\n   Conclusion: Jack flies.' ,
                     ]

        facts, assumptions, conclusion = nl_utils.quick_parse(responses[0], 'Jack flies',
                                                               ['Jack is a bird.', 'Birds fly'])
        print('Facts: ', facts)
        print('Assumptions: ', assumptions)
        print('Conclusions: ', conclusion)

    def test_str_compare(self):
        pairs = [['the cat', 'the cat', True],
                 ['Aron starts to sleep better.',
                  "Therefore, Aron starting to sleep better is plausible if we assume that Venie continues to improve Aron's lot .", True],
                 ['Jack is a bird', 'He said that Jack is not a bird, did he not?', False]
        ]
        for p in pairs:
            matches = llm_utils.str_compare(p[0], p[1])
            print(p, ' ', matches)
            assert matches == p[2]

    def test_dereference(self):
        text =  "Persian (/ˈpɜːrʒən, -ʃən/), also known by its endonym Farsi (فارسی fārsi (fɒːɾˈsiː) ( listen)), is one of the Western Iranian languages within the Indo-Iranian branch of the Indo-European language family. It is primarily spoken in Iran, Afghanistan (officially known as Dari since 1958), and Tajikistan (officially known as Tajiki since the Soviet era), and some other regions which historically were Persianate societies and considered part of Greater Iran. It is written in the Persian alphabet, a modified variant of the Arabic script, which itself evolved from the Aramaic alphabet."
        llm_utils.dereference_text(text)

    def test_get_llm_msg(self):
        msg = llm_utils.get_llm_msg(task='negate_sent', model=None,
                                    msg_type_lbl='prompts', label=None )
        print('msg\n', msg)
        msg = llm_utils.get_llm_msg(task='backstep_in', model='deepseek-r1',
                                    msg_type_lbl='prompts', label=None)
        print('msg\n', msg)
        msg = llm_utils.get_llm_msg(task='backstep_in', model='deepseek-r1',
                                    msg_type_lbl='systems', label=None)
        print('msg\n', msg)


class Test_reasoning(unittest.TestCase):


    @classmethod
    def setUpClass(cls) -> None:
        print('test reasoning')
        cls.bstore = BeliefStore(MILVUS_FILE, SQLITE_FILE)
        #cls.bset = BeliefSet(cls.bstore, '/', 'root')

    @classmethod
    def tearDownClass(cls) -> None:
        print('---teardown---')
        info = cls.bstore.exit(dump_tables=['stmts', 'beliefs', 'supports',
                                            'stmtdists'])
        print(info)
        os.remove(MILVUS_FILE)
        os.remove(SQLITE_FILE)
        print('----all done----')

    def runTest(self):
        print('test_10_bs_0')
        self.test_20_bs_0()

    def test_10_bs_0(self):
        bset = BeliefSet(self.bstore, path='/test_10', description='test_10')
        sentenceReasoner = SentenceReasoner(bset, self.bstore)
        bird_jack, _ = Belief_Support.from_axiom(self.bstore,
                                        bset,
                                        'Jack is a bird',
                                        {'source': 'axiom'})
        print(bird_jack)
        birds_fly, _ = Belief_Support.from_axiom(self.bstore,
                                        bset,
                                        'Birds fly.',
                                        {'source': 'axiom'})
        jack_flies, _ = Belief_Support.from_query(self.bstore,
                                        bset,
                                        'Jack flies.',
                                        {'source': 'query'})
        p = sentenceReasoner.verify(jack_flies, bset, self.bstore)

    def test_20_bs_0(self):
        bset = BeliefSet(self.bstore, path='/test_10', description='test_10')
        sentenceReasoner = SentenceReasoner(bset, self.bstore)
        bird_jack, _ = Belief_Support.from_axiom(self.bstore,
                                        bset,
                                        'Jack is a bird',
                                        {'source': 'axiom'})
        print(bird_jack)
        jack_flies, _ = Belief_Support.from_query(self.bstore,
                                        bset,
                                        'Jack flies.',
                                        {'source': 'query'})
        p = sentenceReasoner.verify(jack_flies, bset, self.bstore)

class Test_supportGraph(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.bstore = BeliefStore(MILVUS_FILE, SQLITE_FILE)
        cls.bset = BeliefSet(cls.bstore, '/', 'root')
        cls.sentenceReasoner = SentenceReasoner(cls.bset, cls.bstore)
        cls.premises = []
        cls.inferred = []
        b1, _ = Belief_Support.from_source(cls.bstore, cls.bset, 'Jack is a bird', {'source_id': 0})
        cls.premises.append(b1)
        #s2 = Support.from_source(0, {'source_id': 0}, cls.bstore)
        #b2 = Belief.from_support(cls.bstore, cls.bset, 'Birds fly', s2)
        #cls.premises.append(b2)
        #s3 = Support.from_reasoning(cls.bstore, 0, [b1.id, b2.id], {})
        #b3 = Belief.from_support(cls.bstore, cls.bset, 'Jack flies', s3)
        #cls.inferred.append(b3)
        jack_flies, _ = Belief_Support.from_query(cls.bstore,
                                        cls.bset,
                                        'Jack flies.',
                                        {'source': 'query'})
        cls.inferred.append(jack_flies)

    @classmethod
    def tearDownClass(cls) -> None:
        info = cls.bstore.exit(dump_tables=['beliefs', 'stmts', 'stmtdists',
                                            'stmt2bel', 'supports'])
        print(info)
        os.remove(MILVUS_FILE)
        os.remove(SQLITE_FILE)

    def runTest(self):
        p = self.sentenceReasoner.verify(self.inferred[0], self.bset, self.bstore)
        self.test_plot_derivation()

    def test_plot_derivation(self):
        supportGraph.plot_derivation(self.inferred[0], self.bstore, 'g1', '/tmp/g1.png', 10)


if __name__ == '__main__':
    #unittest.main()
    testsuite = unittest.TestSuite()
    #testsuite.addTests([TestBelief("test_10_makeBelief")])
    #testsuite.addTests([TestBeliefStore('test_get_similar_beliefs')])
    #testsuite.addTests([TestSupport('test_10_from_source'),
    #                  TestSupport('test_20_by_merging')])
    #testsuite.addTest(TestSupport())
    #testsuite.addTest(TestBelief())
    testsuite.addTest(Test_LLMUtils())
    #testsuite.addTest(Test_reasoning())
    #testsuite.addTest(TestBeliefStore())
    #testsuite.addTest(Test_supportGraph())
    unittest.TextTestRunner().run(testsuite)



