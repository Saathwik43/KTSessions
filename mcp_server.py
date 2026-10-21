"""An MCP server that offers the two Blue Lantern tools to any MCP client.

The tools are the same two functions the Agent demo uses. The difference: they live in their
own program, and any MCP client (app.py, Claude Desktop, Claude Code) can discover and call them.

Run it on its own to check it starts (it waits for a client, press Ctrl+C to stop):
    python mcp_server.py
"""

import ast
import operator
import os

import chromadb
import ollama
from mcp.server.fastmcp import FastMCP

EMBED_MODEL = "nomic-embed-text"
HANDBOOK = os.path.join(os.path.dirname(os.path.abspath(__file__)), "document.txt")

CHUNK_SIZE = 500
CHUNK_OVERLAP = 100
TOP_K = 3

mcp = FastMCP("blue-lantern")


def chunk_text(text):
    chunks = []
    start = 0
    while start < len(text):
        chunks.append(text[start:start + CHUNK_SIZE])
        start += CHUNK_SIZE - CHUNK_OVERLAP
    return chunks


def embed(texts):
    return ollama.embed(model=EMBED_MODEL, input=texts)["embeddings"]


# Index the handbook on the first search, then reuse it.
collection = None


def get_collection():
    global collection
    if collection is None:
        with open(HANDBOOK, encoding="utf-8") as f:
            chunks = chunk_text(f.read())
        collection = chromadb.Client().create_collection("docs", configuration={"hnsw": {"space": "cosine"}})
        collection.add(ids=[str(i) for i in range(len(chunks))], documents=chunks, embeddings=embed(chunks))
    return collection


# --- Tools: @mcp.tool() publishes a function. The name, the type hints and the docstring
# become the tool description that every client sees.

@mcp.tool()
def search_handbook(query: str) -> str:
    """Search the Blue Lantern Coworking member handbook. Use this for any question about the space: plans, prices, rules, timings.

    Args:
        query: What to look for, e.g. "Night Owl plan price"
    """
    results = get_collection().query(query_embeddings=embed([query]), n_results=TOP_K)
    return "\n\n---\n\n".join(results["documents"][0])


OPS = {ast.Add: operator.add, ast.Sub: operator.sub,
       ast.Mult: operator.mul, ast.Div: operator.truediv}


@mcp.tool()
def calculate(expression: str) -> str:
    """Do arithmetic exactly. Use this for every calculation instead of doing maths yourself.

    Args:
        expression: Numbers with + - * / and brackets, e.g. "4500 * 0.8 * 6 + 3000"
    """
    def evaluate(node):
        if isinstance(node, ast.Constant) and isinstance(node.value, (int, float)):
            return node.value
        if isinstance(node, ast.BinOp) and type(node.op) in OPS:
            return OPS[type(node.op)](evaluate(node.left), evaluate(node.right))
        raise ValueError("only numbers and + - * / ( ) are allowed")
    return str(evaluate(ast.parse(expression.replace(",", ""), mode="eval").body))


if __name__ == "__main__":
    mcp.run()   # talks to the client over stdin/stdout (the "stdio" transport)
