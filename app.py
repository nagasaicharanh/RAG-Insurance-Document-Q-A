import streamlit as st
import config
from rag_chain import get_rag_chain_with_sources

# For DeepEval inline check
from deepeval.metrics import HallucinationMetric
from deepeval.test_case import LLMTestCase
import asyncio

st.set_page_config(page_title="RAG Insurance Q&A", layout="wide")

st.title("📄 RAG Insurance Document Q&A")
st.markdown("Ask questions about reinsurance policies. The system retrieves answers from 50+ ingested PDFs using LLaMA 3.3 and ChromaDB.")

if "messages" not in st.session_state:
    st.session_state.messages = []

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])
        if "sources" in message and message["sources"]:
            with st.expander("View Sources"):
                for s in message["sources"]:
                    st.write(f"- {s['source']} (Page {s['page']})")
        if "hallucination_score" in message and message["hallucination_score"] is not None:
            score = message["hallucination_score"]
            if score > 0.5:
                st.error(f"⚠️ Hallucination Alert (Score: {score:.2f})")
            else:
                st.success(f"✅ Factually Grounded (Score: {score:.2f})")

if prompt := st.chat_input("Ask a question about your insurance documents..."):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)
        
    with st.chat_message("assistant"):
        message_placeholder = st.empty()
        message_placeholder.markdown("Thinking...")
        
        try:
            chain = get_rag_chain_with_sources()
            result = chain.invoke(prompt)
            
            response = result["answer"]
            docs = result["context"]
            
            sources = []
            seen = set()
            for doc in docs:
                src = doc.metadata.get("source", "Unknown")
                page = doc.metadata.get("page", "Unknown")
                identifier = f"{src}_page_{page}"
                if identifier not in seen:
                    seen.add(identifier)
                    sources.append({"source": src, "page": page})
            
            message_placeholder.markdown(response)
            
            if sources:
                with st.expander("View Sources"):
                    for s in sources:
                        st.write(f"- {s['source']} (Page {s['page']})")
                        
            status_placeholder = st.empty()
            status_placeholder.info("Running hallucination check...")
            
            score = None
            try:
                loop = asyncio.new_event_loop()
                asyncio.set_event_loop(loop)
                
                hallucination_metric = HallucinationMetric(threshold=0.5)
                test_case = LLMTestCase(
                    input=prompt,
                    actual_output=response,
                    context=[doc.page_content for doc in docs]
                )
                hallucination_metric.measure(test_case)
                score = hallucination_metric.score
                
                status_placeholder.empty()
                if score > 0.5:
                    st.error(f"⚠️ Hallucination Alert (Score: {score:.2f}) - {hallucination_metric.reason}")
                else:
                    st.success(f"✅ Factually Grounded (Score: {score:.2f})")
                    
            except Exception as e:
                status_placeholder.warning(f"Could not run hallucination check: {e}")

            st.session_state.messages.append({
                "role": "assistant", 
                "content": response,
                "sources": sources,
                "hallucination_score": score
            })
            
        except Exception as e:
            st.error(f"An error occurred: {e}")