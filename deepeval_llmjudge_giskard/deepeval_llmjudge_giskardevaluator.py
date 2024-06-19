from giskard import Suite, test
import deepeval as de
#from llm_judge import LLMJudge

# Define a test case for text summarization
test_case = {
    "input": "The quick brown fox jumps over the lazy dog. This is a simple sentence. It is meant to be short and easy to understand.",
    "expected_output": "A quick brown fox jumps over a lazy dog."
}

# Mock LLM for demonstration
def my_llm(text):
    return "A brown fox jumps over a lazy dog."

@test("LLM Summarization Test")
def test_llm_summarization(test_case):
    output = my_llm(test_case["input"])
    
    # DeepEval for various metrics
    de_results = de.evaluate(
        my_llm, 
        [test_case["input"]], 
        [test_case["expected_output"]], 
        metrics=["bleu", "rouge1", "rouge2", "rougeL"]
    )
    print("DeepEval BLEU Score:", de_results["bleu"])
    print("DeepEval ROUGE-1 Score:", de_results["rouge1"])
    print("DeepEval ROUGE-2 Score:", de_results["rouge2"])
    print("DeepEval ROUGE-L Score:", de_results["rougeL"])

    LLM Judge for fluency, relevance, and other metrics
    judge = LLMJudge(
        model_name="gpt-3.5-turbo",  # Choose your LLM for judging
        verbose=True
    )
    judge_results = judge.judge_summaries(
        [output],
        [test_case["input"]]
    )
    print("LLM Judge Fluency:", judge_results["fluency"])
    print("LLM Judge Relevance:", judge_results["relevance"])
    print("LLM Judge Coherence:", judge_results["coherence"])  # Additional LLM Judge metric
    print("LLM Judge Grammaticality:", judge_results["grammaticality"]) # Another additional metric

    # Giskard Metrics: Pass/Fail
    assert de_results["bleu"] > 0.5, "BLEU score too low"  # Custom assertion for Giskard
    #assert judge_results["relevance"] > 0.8, "Relevance score too low"  # Another assertion
    
# Create a test suite
suite = Suite()
suite.add_test(test_llm_summarization)
suite.run()
