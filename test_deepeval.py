import pytest
import deepeval
from deepeval import assert_test
from deepeval.dataset import EvaluationDataset
from deepeval.test_case import LLMTestCase, LLMTestCaseParams
from deepeval.metrics import AnswerRelevancyMetric, GEval
from console_bot import get_chain
from dotenv import load_dotenv
load_dotenv()

# To run this file: deepeval test run <file_name>.py

def get_answer(question):
    store = {}
    conversation_chain = get_chain(store)
    response = conversation_chain.invoke(
        {'input': question},
        config={"configurable": {"session_id": "test123"}},
    )
    return str(response['answer'])

dataset = EvaluationDataset(
    test_cases=[
        LLMTestCase(
            input="What if these shoes don't fit?",
            # Replace this with the actual output of your LLM application
            actual_output=get_answer(input),
            expected_output="You're eligible for a free full refund within 30 days of purchase.",
         )
    ]
)


@pytest.mark.parametrize(
    "test_case",
    dataset,
)
def test_everything(test_case: LLMTestCase):

    answer_relevancy_metric = AnswerRelevancyMetric(threshold=0.7)
    correctness_metric = GEval(
        name="Correctness",
        criteria="Correctness - determine if the actual output is correct according to the expected output.",
        evaluation_params=[
            LLMTestCaseParams.ACTUAL_OUTPUT,
            LLMTestCaseParams.EXPECTED_OUTPUT,
        ],
        # strict=True,
    )
    assert_test(test_case, [answer_relevancy_metric, correctness_metric])


@deepeval.log_hyperparameters(model="gpt-4", prompt_template="...")
def hyperparameters():
    return {"temperature": 1, "chunk size": 500}