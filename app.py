import streamlit as st
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain.memory import StreamlitChatMessageHistory, ConversationBufferMemory
from langchain.chains import ConversationalRetrievalChain
from langchain.prompts import ChatPromptTemplate
from langchain.vectorstores import FAISS
from langchain.text_splitter import RecursiveCharacterTextSplitter

# --- タイトル ---
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
    raw_text = uploaded_file.read().decode("utf-8")
    splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
    docs = splitter.create_documents([raw_text])
    # ベクトルストア作成（毎回再作成の簡易例。実用ではキャッシュ推奨）
    embeddings = OpenAIEmbeddings(openai_api_key=st.session_state["api_key"])
    vectorstore = FAISS.from_documents(docs, embeddings)
else:
    st.info("まず検索対象のファイルをアップロードしてください。")
    st.stop()

# --- チャット履歴用メモリ ---
msgs = StreamlitChatMessageHistory()
memory = ConversationBufferMemory(
    chat_memory=msgs,
    return_messages=True,
    memory_key="history",
)

# --- ConversationalRetrievalChain（RAGチェーン） ---
llm = ChatOpenAI(
    openai_api_key=st.session_state["api_key"],
    model="gpt-4o",   # 最新モデルを指定！
    temperature=0.7,
)
prompt = ChatPromptTemplate.from_messages([
    ("system", "あなたは親切なAIチャットアシスタントです。アップロードされたドキュメントの内容を参考に、必ず日本語で簡潔に回答してください。"),
    ("human", "{context}\n{history}\n{question}")
])
rag_chain = ConversationalRetrievalChain.from_llm(
    llm=llm,
    retriever=vectorstore.as_retriever(),
    memory=memory,
    combine_docs_chain_kwargs={"prompt": prompt}
)

# --- チャット履歴表示 ---
for msg in msgs.messages:
    st.chat_message(msg.type).write(msg.content)

# --- チャット入力 ---
user_input = st.chat_input("メッセージを入力…")
if user_input:
    st.chat_message("user").write(user_input)
    output = rag_chain.run(user_input)
    st.chat_message("assistant").write(output)
