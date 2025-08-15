# rag_mcp_server.py
import requests
from fastmcp import FastMCP
import chromadb
from chromadb.config import Settings
import pdfplumber
from langchain.text_splitter import RecursiveCharacterTextSplitter

# 配置 SiliconFlow
SILICONFLOW_API_KEY = "sk-nbjqhblhcgproxqwnyhtddoostoiuivuvvydafrvlxjzzrtl"
EMBEDDER_URL = "https://api.siliconflow.com/v1/embeddings"
EMBEDDER_MODEL = "Qwen/Qwen3-Embedding-4B"
RERANK_URL = "https://api.siliconflow.com/v1/rerank" 
RERANK_MODEL = "Qwen/Qwen3-Reranker-8B" 
# 参数设置
MCP_NAME = "RAG MCP Server"
K_VALUE_RERANK = 10    # K value in rerank
K_VALUE_RETRIEVE = 100  
PORT = 8001
MAX_CHUNK_SIZE = 800
CHUNK_OVERLAP = 100
EMBED_GROUP_SIZE = 500 # 将chunk后的段落按多少个分为一组进行embedding，防止超过embedding token上限

# 初始化 Chroma
client = chromadb.Client(Settings(persist_directory="./chroma_data"))
collection = client.get_or_create_collection(name="docs", metadata={"hnsw:space":"cosine"})

# 注册 MCP Server
mcp = FastMCP(MCP_NAME)

# 设置 Langchain
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=MAX_CHUNK_SIZE,
    chunk_overlap=CHUNK_OVERLAP,
    separators=["\n\n", "\n", ".", ''],
)

# Embedding 函数
def embed_texts(texts: list[str]) -> list[list[float]]:
    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {SILICONFLOW_API_KEY}"
    }
    payload = {
        "model": EMBEDDER_MODEL,
        "input": texts
    }
    resp = requests.post(EMBEDDER_URL, headers=headers, json=payload)
    if resp.status_code != 200:
        raise Exception(f"Embedding API error: {resp.status_code}, {resp.text}")
    data = resp.json()
    return [item["embedding"] for item in data["data"]]

# Rerank 函数
def siliconflow_rerank(query: str, docs: list[str]):
    headers = {
        "Authorization": f"Bearer {SILICONFLOW_API_KEY}",
        "Content-Type": "application/json"
    }
    payload = {
        "model": RERANK_MODEL,
        "query": query,
        "documents": docs
    }
    resp = requests.post(RERANK_URL, headers=headers, json=payload)
    if resp.status_code != 200:
        raise Exception(f"Rerank API error: {resp.status_code}, {resp.text}")
    data = resp.json()
    return data["results"]
        
# Tool: 文档导入
@mcp.tool(name="ingest", description="Ingest documents into ChromaDB")
def ingest_tool(path):
    """ 
    Embed and upload pdf file to database.
    @param path: path of local file to upload.
    """
    texts = []
    with pdfplumber.open(path) as pdf:
        for p in pdf.pages:
            texts.append(p.extract_text() or "")
    full_text = "\n".join(texts)
    chunks = text_splitter.split_text(full_text)
    metadatas = []
    ids = []
    for i, chunk in enumerate(chunks):
        chunk_id = f"{i}"
        ids.append(chunk_id)
        metadatas.append({"chunk_id": chunk_id, "doc_id": 0, "chunk_index": i})

    embeddings = []
    n=EMBED_GROUP_SIZE
    for c in range(len(chunks)//n):
        embeddings = embeddings + embed_texts(chunks[c*n:c*n+n])
    embeddings = embeddings + embed_texts(chunks[(len(chunks)//n)*n:])

    collection.add(documents=chunks, metadatas=metadatas, ids=ids, embeddings=embeddings)
    return {"status": "ok", "inserted": len(ids)}

# Tool: Retrieve + Rerank
@mcp.tool(name="retrieve_and_rerank", description="Retrieve from ChromaDB and rerank")
def retrieve_and_rerank_tool(query):
    """
    embed user query and look for most similar information in database.
    @param query: user query
    """
    k = K_VALUE_RERANK
    q_emb = embed_texts([query])[0]

    results = collection.query(
        query_embeddings=[q_emb],
        n_results=K_VALUE_RETRIEVE,
        include=["documents", "metadatas"]
    )

    docs = results["documents"][0]
    metas = results["metadatas"][0]

    # SiliconFlow Reranker
    scores_info = siliconflow_rerank(query, docs)

    scored_docs = [
        (
            metas[item["index"]].get("chunk_id", ""),  # 从metadata里拿id
            docs[item["index"]],
            metas[item["index"]],
            item["relevance_score"]
        )
        for item in scores_info
    ]
    ranked = sorted(scored_docs, key=lambda x: x[3], reverse=True)[:k]
    cand = ({
        "candidates": [
            {"id": r[0], "text": r[1], "meta": r[2], "score": float(r[3])}
            for r in ranked
        ]
    })["candidates"]
    context_docs = "\n\n---\n\n".join([f"[{c['id']}]\n{c['text']}" for c in cand])
    return context_docs

# 启动server
if __name__ == "__main__":
    mcp.run(transport='sse', port=PORT)

# ingest_result = ingest_tool("C:\\Users\\33878\\Desktop\\work\\信息记录\\大语言模型.pdf")
# ingest_result = ingest_tool("C:\\Users\\33878\\Desktop\\work\\CV\\2025 summer\\张博实 简历.pdf")
# print("PDF导入结果:", ingest_result)