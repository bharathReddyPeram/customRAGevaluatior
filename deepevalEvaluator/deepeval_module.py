
from deepeval.metrics import GEval
from deepeval.test_case import LLMTestCaseParams
from deepeval.metrics import SummarizationMetric
from deepeval.test_case import LLMTestCase
from deepeval.metrics import AnswerRelevancyMetric
from deepeval.metrics import FaithfulnessMetric
from deepeval.metrics import ContextualRelevancyMetric
from deepeval.metrics import HallucinationMetric
from deepeval.metrics import BiasMetric
from deepeval.metrics import ToxicityMetric
import os


class Deepeval_evaluator:
  def __init__(self):

    openai_api_key = os.environ["OPENAI_API_KEY"]



  def g_eval(self,input, actual_output,retrieval_context):

    correctness_metric = GEval(
      name="Correctness",
      criteria="Determine whether the actual output is factually correct based on the expected output.",
      evaluation_steps=[
          "Check whether the facts in 'actual output' contradicts any facts in 'expected output'",
          "You should also heavily penalize omission of detail",
          "Vague language, or contradicting OPINIONS, are OK"
      ],
      evaluation_params=[LLMTestCaseParams.INPUT, LLMTestCaseParams.ACTUAL_OUTPUT],
    )

    test_case = self.llm_testcase_cont(
    input,
    actual_output,
    retrieval_context
    )
    correctness_metric.measure(test_case)
    return  {"correctness_metric.score":correctness_metric.score,
                       "correctness_metric.reason":correctness_metric.reason}
  

  def Summarization_metric(self,input,actual_output):

    test_case = self.llm_testcase(input,actual_output)
    metric = SummarizationMetric(
    threshold=0.5,
    model="gpt-4",
    assessment_questions=[
        "Is the coverage score based on a percentage of 'yes' answers?",
        "Does the score ensure the summary's accuracy with the source?",
        "Does a higher score mean a more comprehensive summary?"
    ])
    metric.measure(test_case)
    return {"metric.score":metric.score,
                       "metric.reason":metric.reason,
                       "metric.score_breakdown":metric.score_breakdown}
  
  
  def answer_relevancy_metric(self,input,actual_output):

    metric = AnswerRelevancyMetric(
        threshold=0.7,
        model="gpt-4",
        include_reason=True
    )
    test_case = self.llm_testcase(input,actual_output)
    metric.measure(test_case)
    return {"metric.score":metric.score,
                       "metric.reason":metric.reason}
  

  def faithfulness_metric(self,input,actual_output,retrieval_context):

    metric = FaithfulnessMetric(
        threshold=0.7,
        model="gpt-4",
        include_reason=True
    )
    test_case = self.llm_testcase_cont( input, actual_output, retrieval_context)
    metric.measure(test_case)
    return {"metric.score":metric.score,
                       "metric.reason":metric.reason}
  

  def contextual_relevancy_metric(self,input,actual_output,retrieval_context):

    metric = ContextualRelevancyMetric(
        threshold=0.7,
        model="gpt-4",
        include_reason=True
    )
    test_case = self.llm_testcase_cont(input,actual_output,retrieval_context)

    metric.measure(test_case)

    return {"metric.score":metric.score,
                       "metric.reason":metric.reason}
  

  def hallucination_metric(self,input,actual_output,retrieval_context):

    test_case = self.llm_testcase_cont(input,actual_output,retrieval_context)
        
    metric = HallucinationMetric(threshold=0.5)

    metric.measure(test_case)
    return {"metric.score":metric.score,
                       "metric.reason":metric.reason}
    
  

  def bias_etric(self,input,actual_output):

    metric = BiasMetric(threshold=0.5)
    test_case = self.llm_testcase(input,actual_output)

    metric.measure(test_case)
    return {"metric.score":metric.score,
                       "metric.reason":metric.reason}
    
  

  def toxicity_metric(self,input,actual_output):

    metric = ToxicityMetric(threshold=0.5)
    test_case = self.llm_testcase(input,actual_output)

    metric.measure(test_case)
    return {"metric.score":metric.score,
                       "metric.reason":metric.reason}

  

  def llm_testcase(self,input,actual_output):

    return LLMTestCase(
        input=input,
        actual_output=actual_output
    )
  

  def llm_testcase_cont(self,input,actual_output,retrieval_context):
    
    return LLMTestCase(
        input=input,
        actual_output=actual_output,
        retrieval_context=[retrieval_context]
    )
  
  def evaluate_deep_all(self,input,actual_output,retrieval_context):
        deepeval_results = {}
        deepeval_results['g_eval'] = self.g_eval(input,actual_output,retrieval_context)
        deepeval_results['Summarization_metric'] = self.Summarization_metric(input,actual_output)
        deepeval_results['answer_relevancy_metric']= self.answer_relevancy_metric(input,actual_output)
        deepeval_results['faithfulness_metric']= self.faithfulness_metric(input,actual_output,retrieval_context)
        deepeval_results['contextual_relevancy_metric']= self.contextual_relevancy_metric(input,actual_output,retrieval_context)
        deepeval_results['hallucination_metric']= self.hallucination_metric(input,actual_output,retrieval_context)
        deepeval_results['bias_etric']= self.bias_etric(input,actual_output)
        deepeval_results['toxicity_metric']= self.toxicity_metric(input,actual_output)
        return deepeval_results




