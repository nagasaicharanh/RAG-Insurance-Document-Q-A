import json
import os
import config
from rag_chain import get_rag_chain_with_sources

# For RAGAS
from datasets import Dataset
from ragas import evaluate
from ragas.metrics import faithfulness, context_precision
from langchain_groq import ChatGroq
from langchain_huggingface import HuggingFaceEmbeddings

# For DeepEval
from deepeval.metrics import HallucinationMetric
from deepeval.test_case import LLMTestCase

def load_test_data():
    if not os.path.exists(config.TEST_QA_PATH):
        print(f"Test data not found at {config.TEST_QA_PATH}")
        return []
    with open(config.TEST_QA_PATH, 'r') as f:
        return json.load(f)

def run_evaluation():
    test_data = load_test_data()
    if not test_data:
        return
        
    print(f"Loaded {len(test_data)} test cases.")
    
    questions = []
    ground_truths = []
    answers = []
    contexts = []
    
    chain = get_rag_chain_with_sources()
    
    # Run pipeline for each test case
    for item in test_data:
        q = item["question"]
        gt = item["answer"]
        questions.append(q)
        ground_truths.append([gt])
        
        print(f"Processing question: {q}")
        try:
            result = chain.invoke(q)
            ans = result["answer"]
            docs = result["context"]
            
            answers.append(ans)
            contexts.append([doc.page_content for doc in docs])
        except Exception as e:
            print(f"Error evaluating question '{q}': {e}")
            answers.append("Error")
            contexts.append([])

    # 1. RAGAS Evaluation
    print("\n--- Running RAGAS Evaluation ---")
    data = {
        "question": questions,
        "answer": answers,
        "contexts": contexts,
        "ground_truth": ground_truths
    }
    
    dataset = Dataset.from_dict(data)
    
    llm = ChatGroq(model_name=config.MODEL_NAME, temperature=0.0)
    embeddings = HuggingFaceEmbeddings(model_name=config.EMBEDDING_MODEL)
    
    try:
        ragas_result = evaluate(
            dataset=dataset,
            metrics=[faithfulness, context_precision],
            llm=llm,
            embeddings=embeddings,
        )
        print("\nRAGAS Results:")
        print(ragas_result)
    except Exception as e:
        print("RAGAS evaluation failed:", e)

    # 2. DeepEval Hallucination Check (Offline)
    print("\n--- Running DeepEval Hallucination Check ---")
    try:
        hallucination_metric = HallucinationMetric(threshold=0.5)
        for i in range(len(questions)):
            test_case = LLMTestCase(
                input=questions[i],
                actual_output=answers[i],
                context=contexts[i]
            )
            hallucination_metric.measure(test_case)
            print(f"Q: {questions[i]}")
            print(f"Hallucination Score: {hallucination_metric.score}")
            print(f"Reason: {hallucination_metric.reason}")
    except Exception as e:
        print(f"DeepEval hallucination check failed: {e}")

if __name__ == "__main__":
    if not config.GROQ_API_KEY:
        print("Please set GROQ_API_KEY in your .env file")
        exit(1)
        
    run_evaluation()