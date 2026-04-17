#! 8.1.2 数据集与评估
import os
from langsmith import Client, wrappers
from openevals.llm import create_llm_as_judge
from openevals.prompts import CORRECTNESS_PROMPT
from openai import OpenAI

from dotenv import load_dotenv

# 加载环境变量
load_dotenv()

# 从环境变量中获取 API key
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

if not OPENAI_API_KEY:
    raise ValueError("OpenAI API key is required. Set OPENAI_API_KEY env var or input it when prompted.")
client = Client()

dataset = client.create_dataset(
    dataset_name="ds-wooden-length-39", description="A sample dataset in LangSmith."
)
examples = [
    {
        "inputs": {"question": "Which country is Mount Kilimanjaro located in?"},
        "outputs": {"answer": "Mount Kilimanjaro is located in Tanzania."},
    },
    {
        "inputs": {"question": "What is Earth's lowest point?"},
        "outputs": {"answer": "Earth's lowest point is The Dead Sea."},
    },
]
client.create_examples(dataset_id=dataset.id, examples=examples)

# Wrap the OpenAI client for LangSmith tracing
openai_client = wrappers.wrap_openai(OpenAI(api_key=OPENAI_API_KEY,
                                            base_url="https://api.lingyaai.cn/v1/"))


# Define the application logic to evaluate.
# Dataset inputs are automatically sent to this target function.
def target(inputs: dict) -> dict:
    response = openai_client.chat.completions.create(
        model="gpt-4.1-mini",
        messages=[
            {"role": "system", "content": "Answer the following question accurately"},
            {"role": "user", "content": inputs["question"]},
        ],
    )
    return {"answer": response.choices[0].message.content}


# @SampleCode: using an LLM to judge output or another LLM
# Define an LLM-as-a-judge evaluator to evaluate correctness of the output
def correctness_evaluator(inputs: dict, outputs: dict, reference_outputs: dict):
    evaluator = create_llm_as_judge(
        prompt=CORRECTNESS_PROMPT,
        model="openai:o3-mini",
        feedback_key="correctness",
    )
    eval_result = evaluator(
        inputs=inputs, outputs=outputs, reference_outputs=reference_outputs
    )
    return eval_result


# Run experiment
experiment_results = client.evaluate(
    target,
    data="ds-wooden-length-39",
    evaluators=[correctness_evaluator],
    experiment_prefix="experiment-quickstart-wilted-density-36",
    max_concurrency=2,
)
