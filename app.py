"""Browser interface for four demos: a chatbot, chat-with-a-document (RAG), an agent and an agent that gets its tools from an MCP server.

Run:  python -m streamlit run app.py
"""

import asyncio
import ast
import hashlib
import inspect
import operator
import os
import sys

import chromadb
import ollama
import streamlit as st
from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client

CHAT_MODEL = "gemma3:1b"
# Optional: the fine-tuned model (see GUIDE.md). If Ollama has a model with this name,
# "Chat with a document" shows a switch to use it.
FINETUNED_MODEL = "blue-lantern"
EMBED_MODEL = "nomic-embed-text"

CHUNK_SIZE = 500
CHUNK_OVERLAP = 100
TOP_K = 3


def chunk_text(text):
    chunks = []
    start = 0
    while start < len(text):
        chunks.append(text[start:start + CHUNK_SIZE])
        start += CHUNK_SIZE - CHUNK_OVERLAP
    return chunks


def embed(texts):
    return ollama.embed(model=EMBED_MODEL, input=texts)["embeddings"]


@st.cache_resource(show_spinner="Chunking and embedding the document...")
def build_index(text):
    """Chunk + embed + store a document. Cached, so each document is indexed only once."""
    chunks = chunk_text(text)
    name = "doc_" + hashlib.md5(text.encode()).hexdigest()
    collection = chromadb.Client().get_or_create_collection(
        name, configuration={"hnsw": {"space": "cosine"}}
    )
    collection.add(
        ids=[str(i) for i in range(len(chunks))],
        documents=chunks,
        embeddings=embed(chunks),
    )
    return collection


def stream_reply(messages, model=CHAT_MODEL):
    for part in ollama.chat(model=model, messages=messages, stream=True):
        yield part["message"]["content"]


def model_installed(name):
    """True if Ollama has a model with this name (with or without a tag like ':latest')."""
    return any(m.model == name or m.model.startswith(name + ":") for m in ollama.list().models)


# --- Agent: the model decides which tools to use -----------------------------

AGENT_MODEL = "gemma4:e2b"
MAX_STEPS = 5

AGENT_SYSTEM = (
    "You are the assistant for Blue Lantern Coworking. You only help with questions about Blue Lantern. "
    "For any question about the space, call search_handbook first and answer only from what it returns. "
    "For any arithmetic, call calculate. If the handbook does not say, reply that you don't know. "
    "If a question is not about Blue Lantern, say you can only help with Blue Lantern questions. "
    "For greetings, just reply."
)


def search_handbook(query: str) -> str:
    """Search the Blue Lantern Coworking member handbook. Use this for any question about the space: plans, prices, rules, timings.

    Args:
        query: What to look for, e.g. "Night Owl plan price"
    """
    with open("document.txt", encoding="utf-8") as f:
        collection = build_index(f.read())
    results = collection.query(query_embeddings=embed([query]), n_results=TOP_K)
    return "\n\n---\n\n".join(results["documents"][0])


OPS = {ast.Add: operator.add, ast.Sub: operator.sub,
       ast.Mult: operator.mul, ast.Div: operator.truediv}


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


# The allowlist: the only functions the model can reach.
TOOLS = {"search_handbook": search_handbook, "calculate": calculate}
MAX_ARG_LENGTH = 200   # longest text argument we accept from the model


def validate_args(name, args):
    """Check the model's arguments against the tool's signature. Returns a problem, or None if they're fine."""
    params = inspect.signature(TOOLS[name]).parameters
    missing = [p for p in params if p not in args]
    unexpected = [a for a in args if a not in params]
    if missing:
        return f"missing required arguments: {missing}"
    if unexpected:
        return f"unexpected arguments: {unexpected}"
    for arg, value in args.items():
        expected = params[arg].annotation           # the type hint, e.g. str
        if not isinstance(value, expected):
            return f"{arg} must be {expected.__name__}, got {type(value).__name__}"
        if isinstance(value, str) and not value.strip():
            return f"{arg} must not be empty"
        if isinstance(value, str) and len(value) > MAX_ARG_LENGTH:
            return f"{arg} is too long (max {MAX_ARG_LENGTH} characters)"
    return None


def dispatch(name, args):
    """Run a tool only if it passes all three gates. Never raises: every failure becomes a message the model can read."""
    if name not in TOOLS:                           # gate 1: is the tool on the allowlist?
        return f"Error: unknown tool: {name}"
    problem = validate_args(name, args)             # gate 2: are the arguments valid?
    if problem:
        return f"Error: {problem}"
    try:                                            # gate 3: run it, turning crashes into data
        return TOOLS[name](**args)
    except Exception as e:
        return f"Error: {type(e).__name__}: {e}"


def run_agent(messages):
    """The model asks for tools, our code runs them, until the model answers."""
    with st.status("Agent is working...", expanded=True) as status:
        for step in range(MAX_STEPS):
            response = ollama.chat(model=AGENT_MODEL, messages=messages, tools=list(TOOLS.values()), think=False)
            messages.append(response.message)

            if not response.message.tool_calls:
                answer = response.message.content
                if '<|"|>' in answer or "{expression:" in answer:   # tool call written as text
                    st.markdown(f"**Step {step + 1}:** model wrote a tool call as text, asking it to retry")
                    messages.append({"role": "user", "content":
                        "You wrote a tool call as plain text. Call the tool properly, "
                        "or give the final answer without tool syntax."})
                    continue
                status.update(label=f"Agent steps ({step} tool calls)", state="complete", expanded=False)
                return answer


            for call in response.message.tool_calls:
                name, args = call.function.name, call.function.arguments or {}
                result = dispatch(name, args)
                st.markdown(f"**Step {step + 1}:** `{name}` {args}")
                st.text(str(result)[:500])
                messages.append({"role": "tool", "content": str(result), "tool_name": name})

        status.update(label="Stopped: too many steps", state="error")
    return "Sorry, I couldn't finish that within the step limit."


# --- Agent (MCP): the same agent, but the tools come from a separate MCP server --

MCP_SERVER = StdioServerParameters(
    command=sys.executable,
    args=[os.path.join(os.path.dirname(os.path.abspath(__file__)), "mcp_server.py")],
)


def to_ollama_tool(tool):
    """Turn a tool description from the MCP server into the format ollama.chat(tools=...) expects."""
    return {"type": "function",
            "function": {"name": tool.name, "description": tool.description, "parameters": tool.inputSchema}}


async def run_mcp_agent(messages):
    """Start the MCP server, ask it for its tools, then run the same agent loop against them."""
    async with stdio_client(MCP_SERVER) as (read, write), ClientSession(read, write) as session:
        await session.initialize()
        tools = (await session.list_tools()).tools      # DISCOVER: ask the server what it offers
        with st.status("Agent is working...", expanded=True) as status:
            st.markdown("**Tools found on the MCP server:** " + ", ".join(f"`{t.name}`" for t in tools))
            for step in range(MAX_STEPS):
                response = ollama.chat(model=AGENT_MODEL, messages=messages,
                                       tools=[to_ollama_tool(t) for t in tools], think=False)
                messages.append(response.message)

                if not response.message.tool_calls:
                    answer = response.message.content
                    if '<|"|>' in answer or "{expression:" in answer:   # tool call written as text
                        st.markdown(f"**Step {step + 1}:** model wrote a tool call as text, asking it to retry")
                        messages.append({"role": "user", "content":
                            "You wrote a tool call as plain text. Call the tool properly, "
                            "or give the final answer without tool syntax."})
                        continue
                    status.update(label=f"Agent steps ({step} tool calls)", state="complete", expanded=False)
                    return answer

                for call in response.message.tool_calls:
                    name, args = call.function.name, call.function.arguments or {}
                    result = await session.call_tool(name, args)    # CALL: the server runs it, not us
                    text = "\n".join(c.text for c in result.content if c.type == "text")
                    if result.isError:
                        text = "Error: " + text
                    st.markdown(f"**Step {step + 1}:** `{name}` {args}  (run by the MCP server)")
                    st.text(text[:500])
                    messages.append({"role": "tool", "content": text, "tool_name": name})

            status.update(label="Stopped: too many steps", state="error")
    return "Sorry, I couldn't finish that within the step limit."


st.set_page_config(page_title="Ollama Demo")
mode = st.sidebar.radio("Demo", ["Chatbot", "Chat with a document", "Agent", "Agent (MCP)"])
st.sidebar.caption(f"Chat model: {CHAT_MODEL}\n\nEmbedding model: {EMBED_MODEL}")

# Each demo keeps its own chat history.
history = st.session_state.setdefault(mode, [])
reply_model = CHAT_MODEL

if mode == "Chatbot":
    st.title("Ollama Chatbot")
elif mode == "Agent":
    st.title("Blue Lantern Agent")
    st.caption(f"{AGENT_MODEL} with two tools: search_handbook and calculate")
elif mode == "Agent (MCP)":
    st.title("Blue Lantern Agent (MCP)")
    st.caption(f"{AGENT_MODEL} with the same two tools, served by mcp_server.py over MCP")
else:
    st.title("Chat with a document")
    uploaded = st.sidebar.file_uploader("Upload a text file", type=["txt", "md"])
    if uploaded:
        text, doc_name = uploaded.read().decode("utf-8"), uploaded.name
    else:
        with open("document.txt", encoding="utf-8") as f:
            text, doc_name = f.read(), "document.txt"
    collection = build_index(text)
    if model_installed(FINETUNED_MODEL) and st.sidebar.toggle("Use the fine-tuned model"):
        reply_model = FINETUNED_MODEL
    st.caption(f"Answering from **{doc_name}** ({collection.count()} chunks in ChromaDB) with `{reply_model}`")

if st.sidebar.button("Clear chat"):
    history.clear()

for message in history:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

if question := st.chat_input("Type a message"):
    with st.chat_message("user"):
        st.markdown(question)

    with st.chat_message("assistant"):
        if mode == "Chatbot":
            # Send the whole conversation so the model remembers earlier turns.
            messages = history + [{"role": "user", "content": question}]
        elif mode in ("Agent", "Agent (MCP)"):
            messages = ([{"role": "system", "content": AGENT_SYSTEM}]
                        + history + [{"role": "user", "content": question}])
        else:
            results = collection.query(query_embeddings=embed([question]), n_results=TOP_K)
            retrieved = results["documents"][0]
            with st.expander("Retrieved chunks"):
                for chunk, distance in zip(retrieved, results["distances"][0]):
                    st.caption(f"distance {distance:.2f}")
                    st.text(chunk)
            context = "\n\n---\n\n".join(retrieved)
            messages = [{
                "role": "user",
                "content": (
                    "Answer the question using only the context below. "
                    "If the answer is not in the context, say you don't know.\n\n"
                    f"Context:\n{context}\n\n"
                    f"Question: {question}"
                ),
            }]
        if mode == "Agent":
            reply = run_agent(messages)
            st.markdown(reply)
        elif mode == "Agent (MCP)":
            reply = asyncio.run(run_mcp_agent(messages))
            st.markdown(reply)
        else:
            reply = st.write_stream(stream_reply(messages, reply_model))

    history.append({"role": "user", "content": question})
    history.append({"role": "assistant", "content": reply})
