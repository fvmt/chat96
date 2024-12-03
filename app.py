import streamlit as st
from htmlTemplates import css, bot_template, user_template
from unstructuredTools import prepare_qdrant_rag
from inlinelogs import setup_logging
from logging import info
from chains import *


def handle_userinput(user_question):

    response = st.session_state.conversation({'question': user_question})
    info(response)
    st.session_state.chat_history = response['chat_history']

    for i, message in enumerate(st.session_state.chat_history):
        if i % 2 == 0:
            st.write(user_template.replace(
                "{{MSG}}", message.content), unsafe_allow_html=True)
        else:
            st.write(bot_template.replace(
                "{{MSG}}", message.content), unsafe_allow_html=True)


def main():
    load_dotenv()
    st.set_page_config(page_title="Test bot",
                       page_icon=":books:")


    st.write(css, unsafe_allow_html=True)

    if "conversation" not in st.session_state:
        st.session_state.conversation = None
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = None

    st.header("96h chatbot with rag and unstructured data pipelines :books:")
    user_question = st.text_input("Ask questions here:")
    setup_logging(st)

    if user_question:
        handle_userinput(user_question)

    with st.sidebar:
        st.subheader("Configure a source: ")

        data = st.text_input(
            "Enter a directory path to process with unstructured",
            r"c:\data\test",
            key="placeholder",
        )

        if st.button("Process"):
            with st.spinner("Processing"):
                if os.path.isdir(data):
                    info('Proceeding to vectorize the following path: '+ str(data))

                    prepare_qdrant_rag(input_directory=data)
                    info('Pipeline complete!')
                    st.session_state.conversation = get_conversation_chain(get_vectorstore())


if __name__ == '__main__':
    main()
