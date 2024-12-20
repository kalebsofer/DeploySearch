import streamlit as st
import requests
import pandas as pd
from config.settings import get_settings
from logs import log_manager

settings = get_settings()

st.set_page_config(page_title="Simple Search Engine", layout="wide")

def truncate_text(text, max_length=50):
    return text[:max_length] + "..." if len(text) > max_length else text

def display_document(docs, selected_index, doc_type):
    st.write(f"{doc_type} Document:")
    st.write(docs[selected_index])


def search_documents(query: str):
    try:
        response = requests.post(
            f"{settings.BACKEND_URL}/search",
            json={"query": query},
            timeout=settings.TIMEOUT_SECONDS,
        )
        response.raise_for_status()
        return response.json()
    except requests.exceptions.RequestException as e:
        st.error(f"Error connecting to backend service: {str(e)}")
        return None

st.title("Simple Search Engine")

st.write(
    """
    The following model was trained on 80,000 pairs of queries and human-generated answers from Bing's <a href='https://microsoft.github.io/msmarco/' target='_blank'>MS MARCO dataset</a>. 
    Training took a few hours on an [RTX A6000](https://www.runpod.io/pricing).
""",
    unsafe_allow_html=True,
)

with st.expander("How to Use", expanded=False):
    st.write(
        """
    - Type a search query and hit search.
    - Select an article from the dropdown.
    - If you think it's relevant, say thanks!
    - These logs will be passed back to the model as <a href='https://huggingface.co/blog/rlhf' target='_blank'>reinforcement learning from human feedback (RLHF)</a>, improving the model performance with usage.
    """,
        unsafe_allow_html=True,
    )

with st.expander("What is Cosine Similarity?", expanded=False):
    st.write(
        """
    A measure used to determine how similar two vectors are, regardless of their size. It calculates the cosine of the angle between two vectors in an inner product space, which is often used in text analysis to measure document similarity. In the context of search engines, embeddings are vector representations of documents or queries, and cosine similarity helps in ranking documents based on their relevance to a given query.
    """,
        unsafe_allow_html=True,
    )

st.info(
    """
    Google indexes over 50 billion web pages, this database contains 100,000 static documents so keep expectations relative.

    However, it will improve over time through reinforcement learning, getting better the more it's used!
    """,
    icon=":material/info:",
)


col1, col2 = st.columns(2)

with col1:
    query = st.text_input("Enter your search query", max_chars=200)

search_button = st.button("Search")

if "search_performed" not in st.session_state:
    st.session_state.search_performed = False

if search_button and query:
    results = search_documents(query)
    if results:
        st.session_state.search_performed = True
        st.session_state.rel_docs = results["rel_docs"]
        st.session_state.rel_docs_sim = results["rel_docs_sim"]

if st.session_state.search_performed:
    df_similar = pd.DataFrame(
        {
            "Document": [truncate_text(doc) for doc in st.session_state.rel_docs],
            "Cosine Similarity": st.session_state.rel_docs_sim,
        }
    ).reset_index(drop=True)

    with col1:
        st.subheader("Results:")
        if len(df_similar) > 0:
            selected_similar = st.selectbox(
                "Select a similar document to view full text:",
                options=list(range(len(df_similar))),
                format_func=lambda x: df_similar.loc[x, "Document"],
                key="similar_select",
            )
            st.table(df_similar.style.format({"Cosine Similarity": "{:.4f}"}))
        else:
            st.write("No similar documents found.")

    with col2:
        st.subheader("Selected")
        if len(df_similar) > 0 and selected_similar is not None:
            display_document(st.session_state.rel_docs, selected_similar, "")

            log_manager.log_search(
                query=query,
                selected_document=st.session_state.rel_docs[selected_similar],
                similarity_score=st.session_state.rel_docs_sim[selected_similar],
            )

            if st.button("Thanks!", key="thanks_button"):
                log_manager.update_feedback(
                    query=query,
                    selected_document=st.session_state.rel_docs[selected_similar],
                )
                st.success("Thank you for your feedback!")