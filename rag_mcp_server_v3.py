# rag_mcp_server.py
import os
import requests
import aiohttp
from fastmcp import FastMCP, tools
import chromadb
from chromadb.config import Settings
import pdfplumber
import json
import asyncio
import duckdb

# 配置 SiliconFlow
SILICONFLOW_API_KEY = "sk-nbjqhblhcgproxqwnyhtddoostoiuivuvvydafrvlxjzzrtl"
EMBEDDER_URL = "https://api.siliconflow.com/v1/embeddings"
EMBEDDER_MODEL = "Qwen/Qwen3-Embedding-4B"
RERANK_URL = "https://api.siliconflow.com/v1/rerank" 
RERANK_MODEL = "Qwen/Qwen3-Reranker-8B" 

# 初始化 Chroma
client = chromadb.Client(Settings(persist_directory="./chroma_data"))
collection = client.get_or_create_collection(name="docs", metadata={"hnsw:space":"cosine"})

# 注册 MCP Server
mcp = FastMCP("RAG MCP Server")

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
    return data["results"]  # 假设返回是 { "results": [{"index":0,"score":...},...] }
        
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
    
    chunks = [c.strip() for c in full_text.split("\n\n") if c.strip()] # 简单按段分，有待优化
    metadatas = []
    ids = []
    for i, chunk in enumerate(chunks):
        chunk_id = f"{i}"
        ids.append(chunk_id)
        metadatas.append({"chunk_id": chunk_id, "doc_id": 0, "chunk_index": i})

    embeddings = embed_texts(chunks)
    collection.add(documents=chunks, metadatas=metadatas, ids=ids, embeddings=embeddings)
    return {"status": "ok", "inserted": len(ids)}

# Tool: Retrieve + Rerank
@mcp.tool(name="retrieve_and_rerank", description="Retrieve from ChromaDB and rerank")
def retrieve_and_rerank_tool(query):
    """
    embed user query and look for most similar information in database.
    @param query: user query
    """
    k = 10
    q_emb = embed_texts([query])[0]

    results = collection.query(
        query_embeddings=[q_emb],
        n_results=100,
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

# 信息提取工具优化

# Tool:信息扩展
# 当需要补充信息时，找到最相关信息行与其上下几行返回

# Tool:整篇扩写
# 当文章过于结构化时（有很多信息省略），对整篇文章进行智能扩写

if __name__ == "__main__":
    mcp.run(transport='sse', port=8001)

# async def test_ingest_pdf():
#     params = {
#         "source_type": "pdf",
#         "content": "C:\\Users\\33878\\Desktop\\work\\AI应用开发\\提示词测试.pdf",
#         "doc_id": "test_pdf_001"
#     }

# ingest_result = ingest_tool("C:\\Users\\33878\\Desktop\\work\\CV\\2025 summer\\张博实 简历.pdf")
# print("PDF导入结果:", ingest_result)
# query = "Porotech"
# retrieve_result = retrieve_and_rerank_tool(query)
# print(retrieve_result)

# asyncio.run(test_ingest_pdf())

# content=["苹果是一种水果", "香蕉是一种水果"]
# print(asyncio.run(siliconflow_rerank("香蕉是什么",content)))