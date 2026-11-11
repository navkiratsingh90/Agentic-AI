
import streamlit as st
import os

from dotenv import load_dotenv
from typing import TypedDict, Annotated, List
from pydantic import Field
import uuid

from langchain_google_genai import ChatGoogleGenerativeAI
from langgraph.graph import START, END, StateGraph

load_dotenv()


# -------------------------------
# 1. State Schema
# -------------------------------

class ChatSchema(TypedDict, total=False):
    message: Annotated[str, Field(description="User message")]
    res: Annotated[str, Field(description="AI response")]
    curr_thread : str = "default_thread"
    chat_threads : List[str] = []


# -------------------------------
# 2. Initialize LLM
# -------------------------------

llm = ChatGoogleGenerativeAI(
    model="gemini-3.6-flash",
    google_api_key=os.getenv("GEMINI_API_KEY"),
    temperature=0.2
)


# -------------------------------
# 3. LangGraph Node
# -------------------------------

def initiate_chat(state: ChatSchema):
    mes = state["message"]
    res = llm.invoke(mes)

    return {"res": res.content}


# -------------------------------
# 4. Build LangGraph
# -------------------------------

graph = StateGraph(ChatSchema)

graph.add_node("chat_response", initiate_chat)

graph.add_edge(START, "chat_response")
graph.add_edge("chat_response", END)

model = graph.compile()


# -------------------------------
# 5. New Chat Function
# -------------------------------

def generate_thread_id():
    return str(uuid.uuid4())


def new_chat():
    new_thread_id = generate_thread_id()

    st.session_state.curr_thread = new_thread_id
    st.session_state.chat_threads.append(new_thread_id)

    st.session_state.chat_history[new_thread_id] = []


def switch_thread(thread_id):
    st.session_state.curr_thread = thread_id
    
# -------------------------------
# 6. Streamlit UI
# -------------------------------

st.set_page_config(
    page_title="LangGraph Chatbot",
    page_icon="🤖",
    layout="centered"
)

# Initialize session state
if "messages" not in st.session_state:
    st.session_state.messages = []

if "chat_count" not in st.session_state:
    st.session_state.chat_count = 1

if "curr_thread" not in st.session_state:
    st.session_state.curr_thread = "default_thread"

if "chat_threads" not in st.session_state:
    st.session_state.chat_threads = ["default_thread"]
    
if "chat_history" not in st.session_state:
    st.session_state.chat_history = {
        "default_thread": []
    }

# -------------------------------
# 7. Sidebar
# -------------------------------
with st.sidebar:

    st.title("🤖 Gemini AI")

    st.button(
        "＋ New Chat",
        use_container_width=True,
        on_click=new_chat
    )

    st.divider()

    st.subheader("Chat History")

    for thread_id in st.session_state.chat_threads:

        messages = st.session_state.chat_history[thread_id]
        # Generate chat title from first user message
        if messages:
            title = next(
                (msg["content"] for msg in messages
                 if msg["role"] == "user"),
                "New Chat"
            )[:25]
        else:
            title = "New Chat"

        if st.button(
            f"💬 {title}",
            key=thread_id,
            use_container_width=True,
            type="primary" if thread_id == st.session_state.curr_thread else "secondary"
        ):
            switch_thread(thread_id)
            st.rerun()

# -------------------------------
# 8. Main Chat UI
# -------------------------------

st.title("🤖 LangGraph AI Chatbot")
st.caption("Powered by Gemini + LangGraph")

for msg in st.session_state.chat_history[st.session_state.curr_thread]:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])


# -------------------------------
# 9. Chat Input
# -------------------------------

if prompt := st.chat_input("Ask anything..."):

    # Get current thread's messages
    messages = st.session_state.chat_history[
        st.session_state.curr_thread
    ]

    # Store user message
    messages.append({
        "role": "user",
        "content": prompt
    })

    with st.chat_message("user"):
        st.markdown(prompt)

    # Prepare conversation history
    history = "\n".join(
        f'{msg["role"]}: {msg["content"]}'
        for msg in messages[:-1]
    )

    full_prompt = f"""
    Previous conversation:
    {history}

    Current question:
    {prompt}
    """

    with st.chat_message("assistant"):

        def generate_response():

            for chunk, metadata in model.stream(
                {"message": full_prompt},
                stream_mode="messages"
            ):

                if isinstance(chunk.content, str):
                    if chunk.content:
                        yield chunk.content

                elif isinstance(chunk.content, list):
                    for block in chunk.content:
                        if isinstance(block, dict) and "text" in block:
                            yield block["text"]

        response = st.write_stream(generate_response())

    # Store assistant response
    messages.append({
        "role": "assistant",
        "content": response
    })

    # Save updated messages
    st.session_state.chat_history[
        st.session_state.curr_thread
    ] = messages
