from ragas.llms import LangchainLLMWrapper
from ragas.embeddings import LangchainEmbeddingsWrapper
from langchain_openai import ChatOpenAI
from langchain_openai import OpenAIEmbeddings
from typing import Dict, List

try:
    from ragas import SingleTurnSample
    from ragas.metrics import (
        BleuScore,
        NonLLMContextPrecisionWithReference,
        ResponseRelevancy,
        Faithfulness,
        RougeScore
    )
    from ragas import evaluate

    RAGAS_AVAILABLE = True

except ImportError:
    RAGAS_AVAILABLE = False


def evaluate_response_quality(
    question: str,
    answer: str,
    contexts: List[str]
) -> Dict[str, float]:

    if not RAGAS_AVAILABLE:
        return {"error": "RAGAS not available"}

    evaluator_llm = LangchainLLMWrapper(
        ChatOpenAI(
            model="gpt-3.5-turbo",
            temperature=0
        )
    )

    evaluator_embeddings = LangchainEmbeddingsWrapper(
        OpenAIEmbeddings(
            model="text-embedding-3-small"
        )
    )

    sample = SingleTurnSample(
        user_input=question,
        response=answer,
        retrieved_contexts=contexts,
        reference=contexts[0] if contexts else ""
    )

    metrics = [
        BleuScore(),
        RougeScore(),
        ResponseRelevancy(
            llm=evaluator_llm,
            embeddings=evaluator_embeddings
        ),
        Faithfulness(
            llm=evaluator_llm
        ),
        NonLLMContextPrecisionWithReference()
    ]

    results = evaluate(
        dataset=[sample],
        metrics=metrics
    )

    return results.to_pandas().iloc[0].to_dict()