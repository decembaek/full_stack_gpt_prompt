# 가상환경 접속 후  playwright install  실행
# 터미널에서 실행하면 chromium 설치가 진행됨

import streamlit as st

# pip install fake-useragent
from fake_useragent import UserAgent

# playwright install
from langchain.document_loaders import AsyncChromiumLoader, SitemapLoader
from langchain.document_transformers import Html2TextTransformer

from langchain.text_splitter import RecursiveCharacterTextSplitter

# LLM
from langchain.chat_models import ChatOpenAI

# vectorstores
from langchain.vectorstores.faiss import FAISS

# embedders
from langchain.embeddings import OpenAIEmbeddings

# Runnables
from langchain.schema.runnable import RunnableLambda, RunnablePassthrough

# template
from langchain.prompts import ChatPromptTemplate

# LLM
llm = ChatOpenAI(
    temperature=0.1,
)

# Prompt
answers_prompt = ChatPromptTemplate.from_template(
    """
    Using ONLY the following context answer the user's question. If you can't just say you don't know, don't make anything up.
                                                  
    Then, give a score to the answer between 0 and 5.
    If the answer answers the user question the score should be high, else it should be low.
    Make sure to always include the answer's score even if it's 0.
    Context: {context}
                                                  
    Examples:
                                                  
    Question: How far away is the moon?
    Answer: The moon is 384,400 km away.
    Score: 5
                                                  
    Question: How far away is the sun?
    Answer: I don't know
    Score: 0
                                                  
    Your turn!
    Question: {question}
    """
)


def get_answers(inputs):
    docs = inputs["docs"]
    question = inputs["question"]
    answers_chain = answers_prompt | llm

    # answers = []
    # for doc in docs:
    #     result = answers_chain.invoke(
    #         {
    #             "question": question,
    #             "context": doc.page_content,
    #         }
    #     )
    #     answers.append(result.content)

    # st.write(answers)

    # return [
    #     answers_chain.invoke(
    #         {
    #             "question": question,
    #             "context": doc.page_content,
    #         }
    #     ).content
    #     for doc in docs
    # ]
    return {
        "question": question,
        "answers": [
            {
                "answer": answers_chain.invoke(
                    {"question": question, "context": doc.page_content}
                ).content,
                "source": doc.metadata["source"],
                "date": doc.metadata["lastmod"],
            }
            for doc in docs
        ],
    }


choose_prompt = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            """
            Use Only the following pre-existing answers to answer the user's question.

            Use the answers that have the highest score (more helpful) and favor the most recent ones.

            Return the sources of the answers as they are, do not change them.

            Answers: {answers}
            """,
        ),
        ("human", "{question}"),
    ]
)


def choose_answer(inputs):
    answers = inputs["answers"]
    question = inputs["question"]

    choose_chain = choose_prompt | llm
    condensed = "\n\n".join(
        f"Answer:{answer['answer']}\nSource:{answer['source']}\nDate:{answer['date']}\n"
        for answer in answers
    )
    # for answer in answers:
    #     condensed += f"Answer:{answer["answer"]}\nSource:{answer["source"]}\nDate:{answer["date"]}\n"
    return choose_chain.invoke(
        {
            "question": question,
            "answers": condensed,
        }
    )


def parse_page(soup):
    header = soup.find("header")
    footer = soup.find("footer")
    if header:
        header.decompose()
    if footer:
        footer.decompose()
    # text = header.get_text()
    return (
        str(soup.get_text())
        .replace("\n", " ")
        .replace("\xa0", " ")
        .replace("CloseSearch Submit Blog", "")
    )


# Sitemap Loader
@st.cache_data(show_spinner="Loading Web Site")
def load_website(url, user_agent):
    # print("user-agent")
    # print(user_agent)
    splitter = RecursiveCharacterTextSplitter.from_tiktoken_encoder(
        chunk_size=800,
        chunk_overlap=200,
    )

    loader = SitemapLoader(
        url,
        # filter_urls=[
        #     # r"^(.*\/blog\/).*",  # blog 포함하는 url만 통과
        #     r"^(?!.*\/blog\/).*",  # blog 포함하는 url을 제외한 나머지
        # ],
        parsing_function=parse_page,
    )
    loader.requests_per_second = 2
    # Set a realistic user agent
    loader.headers = {"User-Agent": user_agent}
    docs = loader.load_and_split(text_splitter=splitter)
    if not docs:
        raise ValueError("No documents found. Please check the URL and try again.")
    vector_store = FAISS.from_documents(docs, OpenAIEmbeddings())
    # return loader.load()
    # return docs
    return vector_store.as_retriever()


st.set_page_config(
    page_title="SiteGPT",
    page_icon="📰",
)

st.title("SiteGPT")

html2text_transformer = Html2TextTransformer()

# Initialize a UserAgent object
ua = UserAgent()

st.markdown(
    """"
    Site GPT에 오신걸 환영합니다.

    URL을 작성하여 사이트를 탐색하세요
    """
)

with st.sidebar:
    url = st.text_input("URL 을 작성하세요", placeholder="https://example.com")


# async chromium loader
# AsyncChromiumLoader는 비동기로 크롬을 실행하여 페이지를 로드합니다.
# if url:
#     loader = AsyncChromiumLoader([url])
#     docs = loader.load()
#     transformed = html2text_transformer.transform_documents(docs)
#     st.write(transformed)

# SitemapLoader 는 사이트에 sitemap.xml 을 찾고 없으면 에러를 리턴함
if url:
    if ".xml" not in url:
        with st.sidebar:
            st.error("Sitemap URL을 작성해주세요")
    else:
        retriever = load_website(url, ua.random)
        query = st.text_input("Ask a question to the website.")
        # st.write(docs)
        # docs = retriever.invoke("What is the price of GPT-4 Model")

        # docs
        if query:
            chain = (
                {"docs": retriever, "question": RunnablePassthrough()}
                | RunnableLambda(get_answers)
                | RunnableLambda(choose_answer)
            )

            # retriever.invoke 로 감, 그리고 RunnablePassthrough로 질문을 전달
            result = chain.invoke(query)
            st.markdown(result.content.replace("$", "\$"))

# https://openai.com/sitemap.xml
