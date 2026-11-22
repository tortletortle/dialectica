import unittest
from dialectica.engine import AgentProfile, BetaBelief, Task, ThompsonScheduler

class SchedulerTests(unittest.TestCase):
    def agents(self):
        return [AgentProfile("critic"), AgentProfile("builder"), AgentProfile("user")]

    def test_beta_update(self):
        b = BetaBelief()
        b.update(True)
        b.update(False)
        self.assertEqual((b.alpha, b.beta), (2.0, 2.0))
        self.assertAlmostEqual(b.mean, .5)

    def test_select_is_reproducible(self):
        t = Task("t1", "architecture")
        s1 = ThompsonScheduler(self.agents(), seed=17)
        s2 = ThompsonScheduler(self.agents(), seed=17)
        self.assertEqual(s1.select(t), s2.select(t))

    def test_feedback_changes_posterior(self):
        s = ThompsonScheduler(self.agents(), seed=1)
        s.observe("critic", "logic", True)
        self.assertGreater(s.snapshot()["critic"]["logic"]["mean"], .5)

    def test_exploration_floor_one(self):
        s = ThompsonScheduler(self.agents(), seed=2, exploration_floor=1)
        self.assertTrue(s.select(Task("t", "logic")).exploratory)

    def test_invalid_feedback_agent(self):
        s = ThompsonScheduler(self.agents())
        with self.assertRaises(ValueError):
            s.observe("missing", "logic", True)

    def test_invalid_task_cost(self):
        with self.assertRaises(ValueError):
            Task("bad", "logic", estimated_cost=0).priority()

    def test_empty_agents(self):
        with self.assertRaises(ValueError):
            ThompsonScheduler([])

    def test_non_boolean_feedback(self):
        s = ThompsonScheduler(self.agents())
        with self.assertRaises(TypeError):
            s.observe("critic", "logic", 1)

if __name__ == "__main__":
    unittest.main()
