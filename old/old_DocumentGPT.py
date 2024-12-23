import streamlit as st

import time

st.set_page_config(
    page_title="DocumentGPT",
    page_icon="📄",
)

st.title("DocumentGPT")


if "messages" not in st.session_state:
    st.session_state["messages"] = []


def send_message(message, role, save=True):
    with st.chat_message(role):
        st.write(message)
    if save:
        st.session_state["messages"].append({"message": message, "role": role})


for message in st.session_state["messages"]:
    send_message(
        message=message["message"],
        role=message["role"],
        save=False,
    )


input_message = st.chat_input("send a message to the ai")

if input_message:
    send_message(input_message, "human")
    time.sleep(2)
    send_message(f"You said: {input_message}", "ai")

    with st.sidebar:
        st.write(st.session_state)
