import streamlit as st
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain.chains import ConversationalRetrievalChain
from langchain.prompts import ChatPromptTemplate
from langchain.vectorstores import FAISS
from langchain.text_splitter import RecursiveCharacterTextSplitter

st.title("💬AIチャット + RAG（ドキュメント検索強化）")

# --- APIキー入力欄 ---
api_key = st.text_input(
    "OpenAI APIキーを入力してください（入力後、エンターでセット）",
    type="password",
    value=st.session_state.get("api_key", ""),
)
if api_key:
    st.session_state["api_key"] = api_key
if "api_key" not in st.session_state or not st.session_state["api_key"]:
    st.info("OpenAI APIキーを入力してください。")
    st.stop()

# --- ドキュメントアップロード ---
uploaded_file = st.file_uploader("検索対象のテキストファイルをアップロードしてください", type=["txt"])
if uploaded_file:
    if "vectorstore" not in st.session_state or st.session_state.get("uploaded_file_name") != uploaded_file.name:
        raw_text = uploaded_file.read().decode("utf-8")
        splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
        docs = splitter.create_documents([raw_text])
        embeddings = OpenAIEmbeddings(openai_api_key=st.session_state["api_key"])
        st.session_state["vectorstore"] = FAISS.from_documents(docs, embeddings)
        st.session_state["uploaded_file_name"] = uploaded_file.name
    vectorstore = st.session_state["vectorstore"]
else:
    st.info("まず検索対象のファイルをアップロードしてください。")
    st.stop()

# --- チャット履歴管理（user/assistantペアのタプルで保持）---
if "chat_history" not in st.session_state:
    st.session_state["chat_history"] = []

# --- RAGチェーン ---
llm = ChatOpenAI(
    openai_api_key=st.session_state["api_key"],
    model="gpt-4o",
    temperature=0.7,
)
prompt = ChatPromptTemplate.from_messages([
    ("system", "あなたは親切なAIチャットアシスタントです。アップロードされたドキュメントの内容を参考に、必ず日本語で簡潔に回答してください。"),
    ("human", "【参考情報】\n{context}\n\n【会話履歴】\n{chat_history}\n\n【質問】\n{question}")
])
rag_chain = ConversationalRetrievalChain.from_llm(
    llm=llm,
    retriever=vectorstore.as_retriever(),
    combine_docs_chain_kwargs={"prompt": prompt}
)

# --- チャット履歴表示 ---
for role, content in st.session_state["chat_history"]:
    st.chat_message(role).write(content)

# --- チャット入力 ---
user_input = st.chat_input("メッセージを入力…")
if user_input:
    st.chat_message("user").write(user_input)
    # LangChainのchat_history形式は [("user", "..."), ("assistant", "..."), ...]
    # assistant発話を作る前に user発話のみで推論
    output = rag_chain.invoke({
        "question": user_input,
        "chat_history": st.session_state["chat_history"]
    })
    answer = output["answer"] if isinstance(output, dict) and "answer" in output else output
    st.chat_message("assistant").write(answer)
    st.session_state["chat_history"].append(("user", user_input))
    st.session_state["chat_history"].append(("assistant", answer))
