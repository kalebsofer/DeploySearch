import streamlit as st
import requests
import pandas as pd
from config.settings import get_settings

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

col1, col2 = st.columns(2)

with col1:
    query = st.text_input("Enter your search query", max_chars=200)

search_button = st.button("Search")

# Initialize session state
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
        st.subheader("Most Similar Results:")
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
        st.subheader("Selected Document:")
        if len(df_similar) > 0 and selected_similar is not None:
            display_document(st.session_state.rel_docs, selected_similar, "Similar")

else:
    st.write("Enter a query and click 'Search' to see results.")
