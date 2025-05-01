import unittest
import llm_utils

class TestLlmUtils(unittest.TestCase):
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

if __name__ == '__main__':
    #unittest.main()
    suite = unittest.TestSuite()
    suite.addTest(TestLlmUtils("test_backward_step_0"))
    runner = unittest.TextTestRunner()
    #runner.run(TestLlmUtils.test_backward_step_0())
    runner.run(suite)