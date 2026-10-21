"""Demo 3: an agent that decides for itself which tools to use.

A chatbot only talks. An agent can also ACT: we give the model a list of tools
(normal Python functions), and on each turn it either answers or asks for a tool.

  1. ASK    - send the conversation plus the tool list to the model
  2. DECIDE - the model replies with a tool call, or with a final answer
  3. RUN    - our code runs the tool, only if it is on the allowlist AND its arguments are valid
  4. REPEAT - add the result to the conversation and go back to step 1

Run:  python agent.py
"""

import ast
import inspect
import operator

import chromadb
import ollama

AGENT_MODEL = "gemma4:e2b"   # needs a model that supports tool calling
EMBED_MODEL = "nomic-embed-text"

CHUNK_SIZE = 500
CHUNK_OVERLAP = 100
TOP_K = 3
MAX_STEPS = 5        # step limit so the agent always stops

SYSTEM = (
    "You are the assistant for Blue Lantern Coworking. You only help with questions about Blue Lantern. "
    "For any question about the space, call search_handbook first and answer only from what it returns. "
    "For any arithmetic, call calculate. If the handbook does not say, reply that you don't know. "
    "If a question is not about Blue Lantern, say you can only help with Blue Lantern questions. "
    "For greetings, just reply."
)


def chunk_text(text):
    chunks = []
    start = 0
    while start < len(text):
        chunks.append(text[start:start + CHUNK_SIZE])
        start += CHUNK_SIZE - CHUNK_OVERLAP
    return chunks


def embed(texts):
    return ollama.embed(model=EMBED_MODEL, input=texts)["embeddings"]


# Index the handbook once at startup, exactly like rag.py.
with open("document.txt", encoding="utf-8") as f:
    chunks = chunk_text(f.read())
collection = chromadb.Client().create_collection("docs", configuration={"hnsw": {"space": "cosine"}})
collection.add(ids=[str(i) for i in range(len(chunks))], documents=chunks, embeddings=embed(chunks))


# --- Tools: normal Python functions. The docstring tells the model what each does.

def search_handbook(query: str) -> str:
    """Search the Blue Lantern Coworking member handbook. Use this for any question about the space: plans, prices, rules, timings.

    Args:
        query: What to look for, e.g. "Night Owl plan price"
    """
    results = collection.query(query_embeddings=embed([query]), n_results=TOP_K)
    return "\n\n---\n\n".join(results["documents"][0])


OPS = {ast.Add: operator.add, ast.Sub: operator.sub,
       ast.Mult: operator.mul, ast.Div: operator.truediv}


def calculate(expression: str) -> str:
    """Do arithmetic exactly. Use this for every calculation instead of doing maths yourself.

    Args:
        expression: Numbers with + - * / and brackets, e.g. "4500 * 0.8 * 6 + 3000"
    """
    # Small models are bad at maths, so we do it for them. We never use eval():
    # only numbers and + - * / are allowed, so the model cannot run arbitrary code.
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


# --- The agent loop ----------------------------------------------------------

def run_agent(messages):
    """The model ASKS for tools, our code DECIDES whether to run them, until the model answers."""
    for step in range(1, MAX_STEPS + 1):
        response = ollama.chat(model=AGENT_MODEL, messages=messages, tools=list(TOOLS.values()), think=False)
        messages.append(response.message)

        if not response.message.tool_calls:          # no tool requested: this is the final answer
            answer = response.message.content
            if '<|"|>' in answer or "{expression:" in answer:   # small models sometimes write the call as text
                print(f"  [step {step}] model wrote a tool call as text, asking it to retry")
                messages.append({"role": "user", "content":
                    "You wrote a tool call as plain text. Call the tool properly, "
                    "or give the final answer without tool syntax."})
                continue
            return answer

        for call in response.message.tool_calls:
            name, args = call.function.name, call.function.arguments or {}
            print(f"  [step {step}] {name} {args}")
            result = dispatch(name, args)
            preview = str(result).replace("\n", " ")
            print(f"           -> {preview[:120]}{'...' if len(preview) > 120 else ''}")
            messages.append({"role": "tool", "content": str(result), "tool_name": name})

    return "Sorry, I couldn't finish that within the step limit."


print(f"Blue Lantern agent ({AGENT_MODEL}) with tools: {', '.join(TOOLS)}")
print("Ask about the coworking space. Type 'exit' to quit.\n")

messages = [{"role": "system", "content": SYSTEM}]

while True:
    question = input("You: ").strip()
    if question.lower() in ("exit", "quit"):
        break
    if not question:
        continue

    messages.append({"role": "user", "content": question})
    print(f"\nBot: {run_agent(messages)}\n")
