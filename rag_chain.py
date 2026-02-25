import os
from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough, RunnableParallel
from langchain_core.output_parsers import StrOutputParser
import config
from vector_store import get_vector_store

def get_retriever():
    vector_store = get_vector_store()
    return vector_store.as_retriever(
        search_type="mmr",
        search_kwargs={"k": config.TOP_K}
    )

def get_llm():
    return ChatGroq(
        model_name=config.MODEL_NAME,
        temperature=0.0,
        api_key=config.GROQ_API_KEY
    )

def get_rag_chain_with_sources():
    retriever = get_retriever()
    llm = get_llm()
    
    template = """You are a helpful and expert assistant for answering questions about insurance and reinsurance documents.
Use the following pieces of retrieved context to answer the question. 
If you don't know the answer, just say that you don't know. 
Answer purely based on the context provided.

Context:
{context}

Question: {question}

Answer:"""
    
    prompt = ChatPromptTemplate.from_template(template)
    
    def format_docs(docs):
        return "\n\n".join(doc.page_content for doc in docs)
        
    rag_chain_from_docs = (
        RunnablePassthrough.assign(context=(lambda x: format_docs(x["context"])))
        | prompt
        | llm
        | StrOutputParser()
    )
    
    rag_chain_with_source = RunnableParallel(
        {"context": retriever, "question": RunnablePassthrough()}
    ).assign(answer=rag_chain_from_docs)
    
    return rag_chain_with_source

def ask_question(question: str):
    chain = get_rag_chain_with_sources()
    result = chain.invoke(question)
    
    response = result["answer"]
    docs = result["context"]
    
    sources = []
    for doc in docs:
        sources.append({
            "source": doc.metadata.get("source", "Unknown"),
            "page": doc.metadata.get("page", "Unknown")
        })
        
    # Deduplicate sources based on source name and page
    unique_sources = []
    seen = set()
    for s in sources:
        identifier = f"{s['source']}_page_{s['page']}"
        if identifier not in seen:
            seen.add(identifier)
            unique_sources.append(s)
            
    return response, unique_sources

if __name__ == "__main__":
    if not config.GROQ_API_KEY:
        print("Please set GROQ_API_KEY in your .env file")
        exit(1)
        
    test_queries = [
        "What is the maximum coverage for natural disasters?",
        "Are there any exclusions for cyber attacks?"
    ]
    
    for query in test_queries:
        print(f"\nQ: {query}")
        try:
            ans, sources = ask_question(query)
            print(f"A: {ans}")
            print("Sources:", sources)
        except Exception as e:
            print("Error during execution:", e)