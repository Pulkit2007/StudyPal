import os
from dotenv import load_dotenv
import streamlit as st
from langchain_chroma import Chroma
from langchain_groq import ChatGroq
from langchain.chains import ConversationalRetrievalChain
from langchain_huggingface import HuggingFaceEmbeddings
from langchain.memory import ConversationBufferMemory
from torch import embedding
from chatbot_utils import get_chapter_list
from getvideo import getvideo

working_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(working_dir)

env_path = os.path.join(parent_dir, ".env")

load_dotenv(env_path)
DEVICE = os.getenv("DEVICE", "cpu")
EMBEDDING_MODEL = "sentence-transformers/all-mpnet-base-v2"
subjects_list = ["Biology", "Physics", "Chemistry"]


def get_vector_db(chapter, subject):
    if chapter == "All Chapters":
        return os.path.join(
            parent_dir,
            "vector_db",
            f"class_12_{subject.lower()}_vector_db",
        )
    return os.path.join(parent_dir, "chapters_vector_db", chapter)


def setup_chain(selected_chapter, selected_subject):
    vector_db_path = get_vector_db(selected_chapter, selected_subject)
    embeddings = HuggingFaceEmbeddings(
        model=EMBEDDING_MODEL,
        model_kwargs={
            "device": DEVICE,
            "model_kwargs": {"low_cpu_mem_usage": False},
        },
    )
    vectorstore = Chroma(
        persist_directory=vector_db_path, embedding_function=embeddings
    )
    groq_api_key = os.getenv("GROQ_API_KEY")
    if not groq_api_key:
        raise RuntimeError(
            "GROQ_API_KEY is missing. Add it to the project's .env file "
            "or set it as an environment variable."
        )

    llm = ChatGroq(
        model="llama-3.3-70b-versatile",
        temperature=0,
        groq_api_key=groq_api_key,
    )
    memory = ConversationBufferMemory(
        llm=llm, output_key="answer", memory_key="chat_history", return_messages=True
    )
    chain = ConversationalRetrievalChain.from_llm(
        llm=llm,
        memory=memory,
        chain_type="stuff",
        retriever=vectorstore.as_retriever(search_type="mmr", search_kwargs={"k": 3}),
        return_source_documents=True,
        get_chat_history=lambda h: h,
        verbose=True,
    )
    return chain


st.title("📚 Study Pal")

# Initialize the chat history and video history as session state in Streamlit
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []
if "video_history" not in st.session_state:
    st.session_state.video_history = []

selected_subject = st.selectbox(
    label="Select a Subject from class 12", options=subjects_list, index=None
)

if selected_subject:
    chapter_list = (get_chapter_list(selected_subject.lower()) or []) + ["All Chapters"]

    selected_chapter = st.selectbox(
        label=f"Select a Chapter from class 12 - {selected_subject}",
        options=chapter_list,
        index=0,
    )

    if selected_chapter:
        # Reset chat_chain if chapter changes
        if st.session_state.get("selected_chapter") != selected_chapter:
            st.session_state.chat_chain = setup_chain(
                selected_chapter, selected_subject
            )
        st.session_state.selected_chapter = selected_chapter

# Display previous messages
for idx, message in enumerate(st.session_state.chat_history):
    with st.chat_message(message["role"]):
        st.markdown(message["content"])
        # Show video references if present for assistant messages
        if message["role"] == "assistant" and idx < len(st.session_state.video_history):
            video_refs = st.session_state.video_history[idx]
            if video_refs:
                st.subheader("Video Reference")
                for title, link in video_refs:
                    st.info(f"{title}\n\nLink: {link}")

# Input field for user's message
user_input = st.chat_input("Ask AI")

if user_input:
    st.session_state.chat_history.append({"role": "user", "content": user_input})
    st.session_state.video_history.append(None)  # No video refs for user

    with st.chat_message("user"):
        st.markdown(user_input)

    with st.chat_message("assistant"):
        response = st.session_state.chat_chain({"question": user_input})
        st.markdown(response["answer"])

        search_query = ", ".join(
            [
                item["content"]
                for item in st.session_state.chat_history
                if item["role"] == "user"
            ]
        )

        video_titles, video_links = getvideo(search_query)

        st.subheader("Video Reference")
        video_refs = []
        for i in range(3):
            st.info(f"{video_titles[i]}\n\nLink: {video_links[i]}")
            video_refs.append((video_titles[i], video_links[i]))

        st.session_state.chat_history.append(
            {"role": "assistant", "content": response["answer"]}
        )
        st.session_state.video_history.append(video_refs)
