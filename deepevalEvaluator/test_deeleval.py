import unittest
from deepeval_module import Deepeval_evaluator

class TestDeepeval_evaluator(unittest.TestCase):
    def setUp(self):
        self.evaluator = Deepeval_evaluator()

    def test_evaluate_all(self):
        input = "What are the causes of climate change?"
        actual_output = "Climate change is caused by human activities."
        retrieval_context = "Human activities such as burning fossil fuels cause climate change."
        metrics = self.evaluator.evaluate_deep_all(input,actual_output,retrieval_context)
        self.assertIsInstance(metrics, dict)
        #self.assertIn("g_eval", metrics)
        #self.assertIn("Summarization_metric", metrics)
        self.assertIn("answer_relevancy_metric", metrics)
        self.assertIn("faithfulness_metric", metrics)
        self.assertIn("contextual_relevancy_metric", metrics)
        self.assertIn("hallucination_metric", metrics)
        self.assertIn("bias_etric", metrics)
        self.assertIn("hallucination_metric", metrics)
        self.assertIn("toxicity_metric", metrics)


if __name__ == "__main__":
    unittest.main()
