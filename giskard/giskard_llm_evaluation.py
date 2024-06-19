import giskard
import os
from giskard.llm.client.openai import OpenAIClient
import pandas as pd
from giskard.rag.question_generators import complex_questions, double_questions
from giskard.rag import generate_testset, KnowledgeBase
from giskard.rag import QATestset

openai_api_key = os.environ["OPENAI_API_KEY"] 

giskard.llm.set_llm_api("openai")
oc = OpenAIClient(model="gpt-4-turbo-preview")
giskard.llm.set_default_client(oc)


# Load your data and initialize the KnowledgeBase
df = pd.read_csv("streamlit_app\prompts-pt_BR.csv")

knowledge_base = KnowledgeBase.from_pandas(df, columns=["act", "prompt"])

# Generate a testset with 10 questions & answers for each question types (this will take a while)
testset = generate_testset(
    knowledge_base, 
    num_questions=60,
    language='en',  # optional, we'll auto detect if not provided
    agent_description="A customer support chatbot for company X", # helps generating better questions
)

# Save the generated testset
testset.save("my_testset.jsonl")

# You can easily load it back
loaded_testset = QATestset.load("my_testset.jsonl")

# Convert it to a pandas dataframe
df = loaded_testset.to_pandas()



testset = generate_testset(
    knowledge_base, 
    question_generators=[complex_questions, double_questions],
)
