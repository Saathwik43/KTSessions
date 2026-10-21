# Chatbot vs RAG vs Agent: A Guide to the Three Chat Modes

Each mode adds one thing the previous one can't do:

- The **chatbot** can only talk.
- **RAG** lets it answer from your document.
- The **agent** lets it decide for itself what to look up and when to calculate.

All three run locally on Ollama. Try them in the browser with `app.py`, or in the terminal with `chatbot.py`, `rag.py` and `agent.py`.

A fourth step, **fine-tuning**, isn't a new mode. It changes the model a mode uses, so the model itself answers the way we want. See [Fine-tuning](#fine-tuning-changing-the-model-itself) below.

## At a glance

| | Chatbot | Chat with a document (RAG) | Agent |
| --- | --- | --- | --- |
| One-line idea | The model talks from what it learned in training | Our code finds the right part of the document and hands it to the model | The model chooses which tools to use; our code runs them |
| Who decides what happens | Fixed: always just reply | Fixed: always search, then reply | The model, step by step |
| Knows your document | No | Yes | Yes, when it decides to search |
| Exact maths | No, it guesses | No, it guesses | Yes, through the `calculate` tool |
| Remembers earlier turns | Yes | No (each question stands alone) | Yes |
| Models | `gemma3:1b` | `gemma3:1b` + `nomic-embed-text` | `gemma4:e2b` + `nomic-embed-text` |
| Model calls per question | 1 | 1 embedding + 1 chat | 1 to 5 chat calls, plus an embedding per search |
| Terminal script | `chatbot.py` | `rag.py` | `agent.py` |

How a question flows through each mode:

```
Chatbot   You ──> [whole conversation] ──> Model ──> Answer

RAG       You ──> Embed question ──> ChromaDB: 3 closest chunks
                                          │
                  [chunks + question] ──> Model ──> Answer

Agent     You ──> Model ──┬──> "answer"            ──> Answer
                ▲         └──> "call a tool"
                │                  │
                │         Our code checks the tool name and its arguments, then runs it
                └──────── tool result added to the conversation (max 5 rounds)
```

---

## Mode 1: Chatbot

The chatbot is a conversation with the model and nothing else. It is the baseline the other two modes improve on.

### How we built it

1. Keep a Python list called `messages`. In `chatbot.py` it starts with a system message: "You are a friendly, concise assistant."
2. When you type, add your message to the list.
3. Send the **whole list** to `ollama.chat(...)` with `stream=True`, so the reply appears word by word.
4. Add the model's reply to the list too, then wait for the next message.

The model has no memory of its own. It "remembers" your name only because step 3 resends the entire conversation every turn. In `app.py` the list lives in `st.session_state`, so it survives Streamlit reruns.

### Where it falls short

- **It only knows its training data.** Ask "How much is the Night Owl plan?" and it will make up a price, because it has never seen the Blue Lantern handbook.
- **It guesses at maths.** It predicts the next word, it doesn't calculate, so small models often get arithmetic wrong.
- **The conversation keeps growing.** Long chats get slower and can eventually exceed the model's context window.

---

## Mode 2: Chat with a document (RAG)

RAG (Retrieval-Augmented Generation) fixes the chatbot's biggest problem: it lets the model answer from **your** document instead of making things up.

### Why it's better than the chatbot

- **Answers come from the document.** "How much is the Night Owl plan?" now gets Rs. 9,000, taken from the handbook.
- **It can say "I don't know".** The prompt tells the model to answer only from the context, so off-topic questions like "Who is the CEO of Google?" get a refusal instead of a guess.
- **You can check its work.** The retrieved chunks are shown (in the terminal, or under "Retrieved chunks" in the browser), so you can see exactly what the model was given.
- **No retraining.** Change the document and the answers change. The model itself is untouched.

### How we built it

We can't paste a whole long document into a small model every time, so we hand it only the relevant parts.

1. **Chunk:** split the document into 500-character pieces, each overlapping the previous one by 100 characters. The overlap means a fact cut at a boundary still appears whole in one chunk.
2. **Embed:** turn each chunk into a vector (a list of numbers that captures its meaning) with `nomic-embed-text`. Texts with similar meanings get similar vectors.
3. **Store:** put the chunks and their vectors in ChromaDB, a vector database, using cosine distance.
4. **Retrieve:** embed the question the same way and ask ChromaDB for the 3 closest chunks.
5. **Generate:** send those chunks plus the question to `gemma3:1b`, with the instruction: "Answer the question using only the context below. If the answer is not in the context, say you don't know."

Steps 1 to 3 happen once per document. In `app.py`, `build_index` is wrapped in `@st.cache_resource`, so each document is embedded only once, even across reruns.

### Where it falls short

- **It always searches, even when it shouldn't.** "Hi!" still triggers a document search.
- **It searches only once, with your exact words.** A question that needs two facts from different parts of the document may only get one of them.
- **It still can't do maths.** "What do 6 months of Hot Desk cost for a student, with the deposit?" needs the price, the discount and the deposit, then a calculation. The model will find some of the facts and guess the sum.
- **No memory.** Each question is sent alone, so follow-ups like "and for the Night Owl plan?" lose their context.

---

## Mode 3: Agent

The agent flips control. In RAG, **our code** decides the steps: search once, then answer. In the agent, **the model** decides: it can search, search again with different words, calculate, or just reply. It keeps going until it has an answer.

### Why it's better than RAG

- **It chooses when to search.** "Hi!" gets a direct reply; a question about prices gets a search.
- **It can take several steps.** For the student Hot Desk question it can search for the price, search for the discount, then call `calculate("4500 * 0.8 * 6 + 3000")` and get exactly Rs. 24,600.
- **Exact maths.** Arithmetic is done by Python, not guessed by the model.
- **It remembers the conversation**, like the chatbot.
- **You can watch it think.** Every tool call and its result is printed (terminal) or shown in the "Agent is working..." panel (browser).

### How we built it

**1. Tools are ordinary Python functions.**

```python
def search_handbook(query: str) -> str:
    """Search the Blue Lantern Coworking member handbook. Use this for any
    question about the space: plans, prices, rules, timings.

    Args:
        query: What to look for, e.g. "Night Owl plan price"
    """
```

We pass the functions themselves to `ollama.chat(..., tools=[...])`. The Ollama library reads the **name, type hints and docstring** and turns them into a description the model can understand. The docstring is effectively the tool's instruction manual, so write it carefully.

`search_handbook` is the RAG retrieval step (steps 4 and 5 above), now packaged as a tool the model can choose to call.

**2. Three gates decide what actually runs.**

The model never runs code. It only **asks** for a tool by name, with arguments it chose. Every request goes through `dispatch()`, which passes it through three gates in order:

```python
def dispatch(name, args):
    if name not in TOOLS:                    # gate 1: is the tool on the allowlist?
        return f"Error: unknown tool: {name}"
    problem = validate_args(name, args)      # gate 2: are the arguments valid?
    if problem:
        return f"Error: {problem}"
    try:                                     # gate 3: run it, turning crashes into data
        return TOOLS[name](**args)
    except Exception as e:
        return f"Error: {type(e).__name__}: {e}"
```

- **Gate 1, the allowlist.** `TOOLS = {"search_handbook": ..., "calculate": ...}` is the only list of functions the model can reach. Any other name is refused and no Python runs.
- **Gate 2, argument validation.** `validate_args()` reads the tool's own signature with `inspect.signature` and checks the model's arguments against it: no missing arguments, no extra ones, each value has the type in its type hint, and text is neither empty nor longer than `MAX_ARG_LENGTH` (200 characters). A bad request never reaches the tool.
- **Gate 3, errors become data.** If the tool itself fails, the error goes back to the model as a message instead of crashing the app.

`dispatch()` never raises. Every failure becomes an `Error: ...` message the model can read, so it gets a chance to correct itself on the next step.

Why gates 1 and 2, when the system prompt already lists the tools? **A prompt is guidance, not enforcement.** It makes a bad call less likely; only the gates make it impossible.

| The model asks for | Stopped by | What the model gets back |
| --- | --- | --- |
| `book_venue(city="Hyderabad")` | Gate 1 | `Error: unknown tool: book_venue` |
| `calculate()` | Gate 2 | `Error: missing required arguments: ['expression']` |
| `calculate(expression="1+1", city="x")` | Gate 2 | `Error: unexpected arguments: ['city']` |
| `calculate(expression=42)` | Gate 2 | `Error: expression must be str, got int` |
| `search_handbook(query="   ")` | Gate 2 | `Error: query must not be empty` |
| `calculate(expression="__import__('os')")` | The tool's own check, then gate 3 | `Error: ValueError: only numbers and + - * / ( ) are allowed` |
| `calculate(expression="1/0")` | Gate 3 | `Error: ZeroDivisionError: division by zero` |

**3. The agent loop.**

```
repeat up to MAX_STEPS (5) times:
    send the conversation + tool list to the model
    if the model replied with text   -> that's the final answer, stop
    if the model asked for tools     -> run each one, add the result
                                        as a "tool" message, loop again
```

**4. Safety rails:**

- **Step limit (`MAX_STEPS = 5`).** The loop always ends, even if the model keeps asking for tools.
- **Three gates** on every tool call: allowlist, argument validation, errors as data (above).
- **Safe calculator.** `calculate` never uses `eval()`. It parses the expression and allows only numbers and `+ - * /`, so the model can't sneak in Python code.
- **Retry on fake tool calls.** Small models sometimes write a tool call as plain text instead of making a real call. The loop spots this and asks the model to try again.
- **A strict system prompt** keeps it on topic: search before answering, use `calculate` for any maths, say "I don't know" if the handbook doesn't say, and refuse non-Blue Lantern questions.

**5. A model that supports tool calling.** `gemma3:1b` can't call tools, so the agent uses `gemma4:e2b`. We pass `think=False` to skip the model's long internal reasoning and keep replies fast.

### Where it falls short

- **Slower.** One question can take up to 5 model calls instead of 1.
- **Less predictable.** The model chooses the steps, so the same question can take different paths. Small models sometimes skip a search or call the wrong tool.
- **More to secure.** Every tool you add is something the model can trigger. Keep the allowlist small and make each tool safe on its own, as `calculate` is.

### Same agent, tools from an MCP server

In the Agent above, the tools are Python functions inside the app, so only this app can use them. **MCP (Model Context Protocol)** is a standard way to put tools in their own program. Any MCP client can then find them and call them.

`mcp_server.py` holds the same two tools, `search_handbook` and `calculate`. Each is published with `@mcp.tool()`, and the name, type hints and docstring become the description every client sees. In `app.py`, pick **Agent (MCP)** in the sidebar. The agent loop is unchanged. Only two things differ:

1. **Discover.** Instead of a hard-coded list, the app starts the server and asks it, `list_tools()`. The page shows the tools it found.
2. **Call.** When the model asks for a tool, the app sends `call_tool(name, args)` to the server and the server runs it.

The app and the server talk over **stdio** (the app starts the server as a child process), so there are no ports, accounts or API keys. To see the real point of MCP, connect the same server to another client, for example Claude Code: `claude mcp add blue-lantern -- python mcp_server.py`. You wrote the tools once, and now two different apps use them.

---

## Fine-tuning: changing the model itself

The three modes all work by changing what we **send** the model. Fine-tuning changes the **model**: we train it a little more on our own examples, so its habits change.

| Method | What it changes | In this project |
| --- | --- | --- |
| Prompting | The instructions we send | The system prompt and the RAG instruction |
| RAG | The information we send | The handbook chunks from ChromaDB |
| Agent | What the model can **do** | The tools: `search_handbook` and `calculate` |
| Fine-tuning | **The model's own habits** | Training on our own examples |

**Teach behaviour, not facts.** Use fine-tuning for *how* the model answers: format, style, habits. Facts belong in RAG: when the handbook changes, you edit `document.txt` and the answers change straight away. A fact trained into the model stays until you retrain it.

**Try a prompt first.** A clear prompt is cheaper and easier to undo than fine-tuning. So we also measure a well-written prompt, and fine-tuning only counts as worth it if it beats that prompt.

### What we teach, and how we measure it

Every answer in one fixed format, and one fixed reply when the handbook doesn't cover the question:

```
**Answer:** Rs. 4,500 per month.                      **Answer:** I don't know. The handbook doesn't cover this.
**Details:** The Hot Desk plan gives you any free     **Details:** Please ask the community manager, Farah Khan,
desk in the open area.                                at the front desk.
**Source:** Membership Plans                          **Source:** Not in the handbook
```

Success criteria, decided before training:

| Criterion | Checked by |
| --- | --- |
| Format | The answer matches the three-line pattern |
| Source | The `**Source:**` line names the expected handbook section |
| Facts | The answer contains the expected key words, for example "Deep Sea" or "104" |
| No regressions | General questions with no handbook still get normal answers |

### Key ideas

- **LoRA:** the original model stays frozen, and we train a small add-on next to it. The trained add-on is called an **adapter**: a file of about 50 MB.
- **QLoRA:** LoRA with the frozen model stored in 4-bit, so it needs less GPU memory. In code, that is one setting: `load_in_4bit = True`.
- **Unsloth:** an open-source library that makes this training fast enough for a free GPU. We use its official Gemma 4 notebook.
- **Three sets of data:** the model **trains** on one, a **validation** set is checked during training to spot over-learning, and a locked **test** set is used only at the end to compare models.
- **Loss** measures how far the model's answers are from the examples. A falling loss is not proof the model improved. Only better answers on unseen questions are.

### The steps

| Step | Where | What happens |
| --- | --- | --- |
| 1. Prepare the data | Laptop | Train, validation and test files |
| 2. Set up the notebook | Colab | GPU, install, load Gemma 4, add LoRA, upload the data |
| 3. Measure the baseline | Colab | The untrained model on the test questions |
| 4. Train | Colab | 3 passes over the training data |
| 5. Evaluate | Colab | The same test on the fine-tuned model |
| 6. Save the adapter | Colab → laptop | Download `blue_lantern_adapter.zip` |
| 7. Use it in the chatbot | Colab → Hugging Face → Ollama | Merge, convert to GGUF, pull into Ollama |

Training needs an NVIDIA GPU, so steps 2 to 7 run on Google Colab's free T4. A laptop CPU is far too slow.

#### Step 1: prepare the data

```
cd finetune
python make_dataset.py
```

| File | Contents | Used for |
| --- | --- | --- |
| `blue_lantern_train.jsonl` | 46 conversations: 37 answerable, 9 not covered by the handbook | Training |
| `blue_lantern_validation.jsonl` | 5 conversations | Checked during training, never learned from |
| `blue_lantern_test.jsonl` | 9 questions, each with a context, the expected section and the key words a correct answer needs | Comparing models at the end |

All the examples are written in `make_dataset.py`, one line each. How the data is built:

- **Each example is the exact prompt `app.py` sends** in *Chat with a document* mode: the instruction, a context and the question. The context is the right handbook section plus an unrelated one, because real retrieval returns some irrelevant text.
- **Every answer must be true to `document.txt`.** The model copies mistakes too.
- **The test set is locked.** Its questions are never trained on. The script stops with an error if a test question is a near-copy of a training one, because that would make results look better than they are.

#### Step 2: set up the notebook

1. Open Unsloth's notebook, pinned to the version this guide was written against:
   https://colab.research.google.com/github/unslothai/notebooks/blob/b27cbf9c938011934e1aaa4f4ee0e5710eaf250d/nb/Gemma4_(E2B)-Text.ipynb
2. **Runtime → Change runtime type → T4 GPU**, then **File → Save a copy in Drive**.
3. Run the two **Installation** cells (about 3 minutes, no output).
4. In the **`FastModel.from_pretrained`** cell, set these two values, then run it:

   ```python
       max_seq_length = 2048,
       load_in_4bit = True,   # QLoRA
   ```

5. **Skip** every cell from `do_gemma_4_inference` down to the "Let's finetune Gemma 4!" heading. They are image and audio demos.
6. In the **`get_peft_model`** cell (adds the LoRA add-on), set `r = 16` and `lora_alpha = 16`, then run it. `r` is the size of the add-on; the notebook's default of 8 was too weak in our first run. Then, under *Data Prep*, run **`get_chat_template`**.
7. Replace the cell that loads `mlabonne/FineTome-100k`, and the `standardize_data_formats`, `formatting_prompts_func` and `dataset[100]` cells after it, with this one cell. Run it and upload the three `.jsonl` files from `finetune/`:

   ```python
   from google.colab import files
   files.upload()

   import json
   from datasets import load_dataset
   from unsloth.chat_templates import standardize_data_formats

   def formatting_prompts_func(examples):
       texts = [tokenizer.apply_chat_template(convo, tokenize = False, add_generation_prompt = False).removeprefix("<bos>")
                for convo in examples["conversations"]]
       return {"text": texts}

   def load(path):
       data = load_dataset("json", data_files = path, split = "train")
       return standardize_data_formats(data).map(formatting_prompts_func, batched = True)

   train_dataset = load("blue_lantern_train.jsonl")
   val_dataset = load("blue_lantern_validation.jsonl")
   TEST = [json.loads(line) for line in open("blue_lantern_test.jsonl", encoding="utf-8")]

   print(len(train_dataset), "train,", len(val_dataset), "validation,", len(TEST), "test")
   print(train_dataset[0]["text"])
   ```

   It prints `46 train, 5 validation, 9 test`, then one example as the model will see it: `<|turn>user ... <turn|>` followed by `<|turn>model **Answer:** ...`.

#### Step 3: measure the baseline

Add this cell. It defines the two prompts, how to ask the model, and the scoring:

```python
import re

# The prompt app.py sends today
PLAIN = ("Answer the question using only the context below. "
         "If the answer is not in the context, say you don't know.\n\n"
         "Context:\n{context}\n\nQuestion: {question}")

# The best prompt we can write, with the format spelled out
GOOD = ("Answer the question using only the context below.\n"
        "Reply in exactly this format, with these three lines:\n"
        "**Answer:** <a short answer>\n"
        "**Details:** <one supporting sentence from the context>\n"
        "**Source:** <the name of the handbook section you used>\n"
        "If the answer is not in the context, reply exactly:\n"
        "**Answer:** I don't know. The handbook doesn't cover this.\n"
        "**Details:** Please ask the community manager, Farah Khan, at the front desk.\n"
        "**Source:** Not in the handbook\n\n"
        "Context:\n{context}\n\nQuestion: {question}")

def generate(text, max_new_tokens = 150):
    messages = [{"role": "user", "content": [{"type": "text", "text": text}]}]
    inputs = tokenizer.apply_chat_template(
        messages, add_generation_prompt = True, return_tensors = "pt",
        tokenize = True, return_dict = True).to("cuda")
    out = model.generate(**inputs, max_new_tokens = max_new_tokens, do_sample = False)
    new_tokens = out[0][inputs["input_ids"].shape[-1]:]       # drop the prompt
    return tokenizer.decode(new_tokens, skip_special_tokens = True).strip()

FORMAT = re.compile(r"^\*\*Answer:\*\*.+\n\*\*Details:\*\*.+\n\*\*Source:\*\*.+$")

def score(reply, item):
    text = reply.replace("’", "'")                         # curly apostrophe -> straight
    last_line = text.splitlines()[-1] if text else ""
    return {"format": bool(FORMAT.match(text)),
            "source": item["source"].lower() in last_line.lower(),
            "fact": all(word.lower() in text.lower() for word in item["must_contain"])}

def evaluate(name, template):
    results = []
    for item in TEST:
        reply = generate(template.format(context = item["context"], question = item["question"]))
        results.append({"question": item["question"], "reply": reply, **score(reply, item)})
    n = len(results)
    print(f"{name}: format {sum(r['format'] for r in results)}/{n} · "
          f"source {sum(r['source'] for r in results)}/{n} · "
          f"facts {sum(r['fact'] for r in results)}/{n}")
    for r in results:
        marks = " ".join(f"{k} {'✓' if r[k] else '✗'}" for k in ("format", "source", "fact"))
        print(f"Q: {r['question']}\n{r['reply']}\n[{marks}]\n" + "-" * 60)

# Regression check: general questions with no handbook at all
GENERAL = ["What is 15 percent of 200? Answer with just the number.",
           "Write one sentence about the monsoon in India.",
           "Translate 'good morning' into Hindi."]

def regression(name):
    print(f"--- {name} ---")
    for q in GENERAL:
        print(f"Q: {q}\n{generate(q, 80)}\n")
```

Then run the baseline. The add-on starts at zero, so this is still the original Gemma 4:

```python
evaluate("Base model, plain prompt", PLAIN)
evaluate("Base model, good prompt", GOOD)
regression("Base model")
```

`do_sample = False` makes the model give the same answer every time, so differences between runs come from the model, not luck. Each test question is asked with its own context from the test file, the same shape as training and as the app.

#### Step 4: train

Replace the notebook's **`SFTTrainer`** and **`train_on_responses_only`** cells with this one cell, and run it:

```python
from trl import SFTTrainer, SFTConfig
from unsloth.chat_templates import train_on_responses_only

trainer = SFTTrainer(
    model = model, tokenizer = tokenizer,
    train_dataset = train_dataset, eval_dataset = val_dataset,
    args = SFTConfig(
        dataset_text_field = "text",
        per_device_train_batch_size = 1, gradient_accumulation_steps = 4,
        warmup_steps = 5,
        num_train_epochs = 8,          # 8 passes over the 46 examples, about 92 steps
        learning_rate = 2e-4, logging_steps = 5,
        eval_strategy = "epoch",       # check the validation set after each pass
        optim = "adamw_8bit", weight_decay = 0.001, lr_scheduler_type = "linear",
        seed = 3407, report_to = "none"))

# Learn ONLY from the answer part of each example
trainer = train_on_responses_only(
    trainer, instruction_part = "<|turn>user\n", response_part = "<|turn>model\n")

# CHECK: this must print only the answer, not the question or the handbook
labels = trainer.train_dataset[0]["labels"]
print(tokenizer.decode([t for t in labels if t != -100]))
```

**Read what the check prints before you train.** It must be only the answer:

```
**Answer:** 4th floor, Mayuri Towers, Madhapur, Hyderabad.
**Details:** It is a shared workspace in Hyderabad.
**Source:** Welcome<turn|>
```

Each example is about 85% handbook text and 15% answer. If the model learns from the whole text, it mostly gets better at repeating the handbook, and the loss falls without the answer style changing. `train_on_responses_only` hides everything except the answer from the training, and this check proves it did. If the check shows the question or the handbook, don't train.

Then train. It takes 12 to 15 minutes; keep the tab open until step 6 is done:

```python
trainer_stats = trainer.train()
```

Training loss should fall. So should validation loss, shown after each pass. If validation loss rises while training loss keeps falling, the model is memorising: train for fewer passes.

#### Step 5: evaluate

```python
evaluate("Fine-tuned, plain prompt", PLAIN)
regression("Fine-tuned model")
```

| What you see | What it means |
| --- | --- |
| Fine-tuned beats the good prompt on format and source | Fine-tuning was worth it: the habit is built in, with a short prompt |
| The good prompt does as well | Fine-tuning wasn't needed; use the better prompt. A valid result |
| A question from the **training** data still gets a plain answer | The training didn't teach the format. Check step 4's label check, then train for more passes |
| Facts went down | The model learned the format but reads the context worse. Fewer passes or more varied examples |
| General answers come out in the handbook format, or got worse | The model over-learned. Fewer passes, or a smaller `r` |

Read the answers as well as the scores. The automatic checks can't spot a detail the handbook doesn't state.

#### Step 6: save the adapter

Do this straight after training. If Colab disconnects first, the training is lost.

```python
model.save_pretrained("blue_lantern_adapter")
tokenizer.save_pretrained("blue_lantern_adapter")
!zip -r blue_lantern_adapter.zip blue_lantern_adapter
files.download("blue_lantern_adapter.zip")
```

Keep the zip in `finetune/`. It holds the adapter only (about 50 MB), not the base model.

#### Step 7: use it in the chatbot

**Ollama can't load a LoRA adapter.** Current versions refuse with `Error: LoRA adapters are no longer supported`. The adapter has to be merged into the base model and converted to one GGUF file, the format Ollama runs. Unsloth does both in one call, and uploads the result to Hugging Face.

**Check before you export.** The export takes about 15 minutes, so first ask the model a question the way the app does (the plain prompt with two handbook sections as context). If the answer isn't in the *Answer / Details / Source* format, exporting gives you a chatbot that behaves the same as before: fix the training first.

1. On huggingface.co: **Settings → Access Tokens → Create new token**, type **Write**.
2. In Colab, in the session where you trained:

   ```python
   import getpass
   HF_USER = input("Hugging Face username: ").strip()
   HF_TOKEN = getpass.getpass("Hugging Face WRITE token: ")

   model.push_to_hub_gguf(f"{HF_USER}/blue-lantern-gemma4", tokenizer,
                          quantization_method = "Q8_0", token = HF_TOKEN)
   ```

   It takes 10 to 15 minutes and uploads about 5 GB. The repo is public by default: fine for this made-up handbook, not for real data.

   **If that session is gone,** start a new one on a T4 and run the two Installation cells (in a blank notebook, copy them from Unsloth's notebook), then load the adapter from the zip before running the cell above. Loading it in a fresh session also proves the saved adapter works on its own:

   ```python
   from google.colab import files
   files.upload()                                   # blue_lantern_adapter.zip
   !unzip -q blue_lantern_adapter.zip

   from unsloth import FastModel
   model, tokenizer = FastModel.from_pretrained(
       model_name = "blue_lantern_adapter", max_seq_length = 2048, load_in_4bit = True)
   from unsloth.chat_templates import get_chat_template
   tokenizer = get_chat_template(tokenizer, chat_template = "gemma-4")
   ```

3. On your laptop, pull the model and give it the name the app looks for:

   ```
   ollama pull hf.co/YOUR-USERNAME/blue-lantern-gemma4
   ollama cp hf.co/YOUR-USERNAME/blue-lantern-gemma4 blue-lantern
   ```

4. Run the app. In *Chat with a document*, a **Use the fine-tuned model** switch now appears in the sidebar. It only shows when Ollama has a model named `blue-lantern` (`FINETUNED_MODEL` in `app.py`). Ask the same question with the switch off and on to compare. The fine-tuned model is slower than `gemma3:1b`, because it is bigger.

In a real product the model only **drafts**. Deterministic checks (does the `**Source:**` section exist in the handbook?) and, for anything published, a human decide what goes out.

### Results so far

**First run: the fine-tune did not work.** Notebook defaults (`r = 8`, `lora_alpha = 8`), the base model in 4-bit, 3 passes (about 36 steps).

| | Result |
| --- | --- |
| Training loss | Fell to about 0.2 |
| Fine-tuned, plain prompt (9 test questions, whole handbook as context) | Format 0/9 · Source 1/9 · Facts 8/9 |
| Regression check | Normal answers to all 3 general questions |
| Baseline scores | Not recorded |

The loss fell and the adapter's weights changed, but the model answered exactly like the untrained one. We narrowed it down in a fresh session, loading the adapter from the zip:

| Check | Result | So |
| --- | --- | --- |
| The same question with an app-style context (2 sections) | Still a plain answer | The test's context wasn't the cause |
| The prompt the model is given | Ends with the model-turn marker, the same as in training | The prompt format is right |
| Is the adapter attached? | Yes, 12.7 million trained values | It loaded correctly |
| A question that **was** in the training data | Still a plain answer | **The training didn't teach the format** |

The most likely cause: the model learned from the whole text of each example, which is mostly handbook, and 36 steps was light. So the loss fell without the answers changing.

**Second run: the fine-tune worked.** Three changes: `r = 16`, 8 passes (96 steps), and `train_on_responses_only` with the check, before training, that only the answer is learned. It trained 25.3 million values, 0.49% of the model, in under 6 minutes on the T4.

| Pass | Training loss | Validation loss | |
| --- | --- | --- | --- |
| 1 | 1.62 | 0.71 | Learning |
| 2 | 0.42 | 0.36 | Learning |
| 3 | 0.08 | **0.35** | Best point |
| 4 | 0.03 | 0.38 | Memorising |
| 5 | 0.006 | 0.38 | |
| 6 | 0.005 | 0.41 | |
| 7 | 0.003 | 0.42 | |
| 8 | 0.002 | 0.43 | |

After pass 3 the training loss kept falling while the validation loss rose. That is overfitting: the model went on memorising its 46 training answers without getting better on unseen ones. **Three passes was enough; eight was too many.** The rise is small, and the model still tested well.

Tested with the plain prompt and two handbook sections as context, on questions that are not in the training data:

| Question | Answer | |
| --- | --- | --- |
| Which room fits a team of 12? | `**Answer:** Deep Sea.` / `**Details:** It seats 12 people and has a video conferencing setup.` / `**Source:** Meeting Rooms` | Full format, right fact, right source |
| My friend wants to visit after work, around 9 PM. Is that ok? | `**Answer:** No.` / `**Details:** Guests are not allowed in the space after 8:00 PM.` / `**Source:** Guests` | Full format, right fact, right source |
| Do you have showers? | `I don't know. The handbook doesn't cover this.` | The right words, but only one line of the three |
| What is 15 percent of 200? (no handbook) | `30` | A normal answer |
| Write one sentence about the monsoon in India. (no handbook) | An ordinary sentence | A normal answer |

- **The format was learned** and carries over to questions the model never saw.
- **Nothing was lost.** General questions still get ordinary answers.
- **The "I don't know" reply was only half learned.** There are 9 examples of it against 37 answerable ones. More of them should fix it.
- **Not yet done:** the full scored test on all 9 questions with both baselines (steps 3 and 5), and step 7.

Three lessons from the two runs:

- **A falling loss proved nothing.** In the first run the loss fell and the model hadn't changed. Only the evaluation showed it.
- **Check what the model learns from before training.** The label check in step 4 is the difference between the two runs.
- **Watch the validation loss** to see when to stop, and test the model before spending time exporting it.

### When something goes wrong

| Problem | Fix |
| --- | --- |
| `Unsloth cannot find any torch accelerator? You need a GPU.` | The runtime is on a CPU. Set the runtime type to T4 and check with `!nvidia-smi`. This restarts the runtime, so run the Installation cells again |
| T4 is greyed out, or "Cannot connect to GPU backend" | The free GPU quota is used up. It usually comes back after some hours |
| Warning: `Unsloth should be imported before [trl, transformers, peft]` | Something imported those first. Put `import unsloth` at the top of that cell |
| `NameError` after leaving the tab idle | Colab reset the runtime: code edits are kept, everything in memory is gone. Re-run the cells in order and upload the files again |
| `ModuleNotFoundError: No module named 'unsloth'` | Colab gave you a fresh machine. Run the two Installation cells again. To clear memory without losing the packages, use **Runtime → Restart session**, not *Disconnect and delete runtime* |
| `Error: LoRA adapters are no longer supported` (Ollama) | Ollama can't load an adapter. Merge and convert it to GGUF first (step 7) |
| `FileNotFoundError` in the data cell | A file wasn't uploaded. Check the 📁 folder icon in Colab's sidebar |
| No GPU on the day of a session | Show your saved outputs and adapter, and say plainly that they are precomputed |

### Where it falls short

- **Slow to change.** A change in behaviour means editing the examples and training again. A prompt change takes seconds.
- **It can learn the wrong thing**, or nothing visible. Only the evaluation tells you.
- **It can't guarantee correct answers.** It teaches a habit; it doesn't stop the model stating something the source doesn't say.
- **It needs a GPU** to train, and Ollama needs the result merged and converted before it can run it.
- **It doesn't add knowledge reliably.** Keep using RAG for the facts.

---

## Customising each mode

All settings are constants at the top of each script. In `app.py`, the chatbot and RAG settings are at the top and the agent settings are under the `# --- Agent` heading.

### Chatbot

| What to change | Where | Effect |
| --- | --- | --- |
| Model | `MODEL` in `chatbot.py`, `CHAT_MODEL` in `app.py` | Swap in e.g. `gemma3:4b` or `llama3.2`. Run `ollama pull <name>` first. Bigger means smarter but slower. |
| Personality | The system message in `chatbot.py` | e.g. "You are a patient maths tutor. Explain step by step." `app.py`'s Chatbot mode has no system message; add one at the start of `messages` to give it a role. |
| Creativity | Add `options={"temperature": 0.2}` to `ollama.chat(...)` | Lower is more focused and repeatable; higher is more varied. |
| Memory length | Trim `messages` before sending, e.g. system message + last 10 | Keeps long chats fast and inside the context window. |

### RAG

| What to change | Where | Effect |
| --- | --- | --- |
| Document | `python rag.py my_notes.txt`, or upload in the browser sidebar | Answers come from your own text. Only `.txt` and `.md` are supported. |
| `CHUNK_SIZE` | Top of `rag.py` / `app.py` | Smaller chunks give more precise matches but less surrounding context; larger chunks do the opposite. |
| `CHUNK_OVERLAP` | Same place | More overlap means facts are less likely to be split, at the cost of more chunks. Keep it smaller than `CHUNK_SIZE`. |
| `TOP_K` | Same place | How many chunks the model sees. Raise it if answers miss facts; lower it if answers get confused by irrelevant text. |
| Prompt | The "Answer the question using only the context below..." text | e.g. ask for bullet points, a reply in Hindi, or quoting the sentence it used. |
| Embedding model | `EMBED_MODEL` | Try `mxbai-embed-large`. If you change it, re-embed the document: vectors from different models can't be compared. |
| Keep the index between runs | Replace `chromadb.Client()` with `chromadb.PersistentClient(path="chroma_db")` | Skips re-embedding on every start. Useful for large documents. |

### Agent

| What to change | Where | Effect |
| --- | --- | --- |
| Model | `AGENT_MODEL` | Must support tool calling, e.g. `qwen3` or `llama3.2`. Check the model's Ollama page for a "tools" tag. |
| `MAX_STEPS` | Top of `agent.py`, agent section of `app.py` | Allow more steps for multi-part questions; fewer for speed. |
| `MAX_ARG_LENGTH` | Next to `TOOLS` | The longest text argument gate 2 accepts. Raise it if a tool legitimately needs long input. |
| Validation rules | `validate_args()` | Add checks that apply to every tool, e.g. reject numbers below zero. Checks specific to one tool belong inside that tool. |
| Rules and scope | `SYSTEM` in `agent.py`, `AGENT_SYSTEM` in `app.py` | Change the business it represents, when to use each tool, and what to refuse. |
| Retrieval | `CHUNK_SIZE`, `CHUNK_OVERLAP`, `TOP_K` | Same as RAG, because `search_handbook` uses the same pipeline. |
| Tools | Write a function, add it to `TOOLS` | See below. |

**Adding a tool** takes three steps. For example, a tool that tells the agent today's date, so it can answer "Is the space open today?":

```python
from datetime import date

def today() -> str:
    """Get today's date and day of the week. Use this for any question about today or opening hours right now."""
    return date.today().strftime("%A, %d %B %Y")

TOOLS = {"search_handbook": search_handbook, "calculate": calculate, "today": today}
```

1. **Write a normal function** with type hints and a clear docstring. Say *when* to use it, not just what it does.
2. **Add it to `TOOLS`.** This is the only way the model can reach it. Gate 2 then validates its arguments automatically, using its type hints.
3. **Mention it in the system prompt** if the model should prefer it for certain questions, e.g. "For anything about today, call today first."

Rules for safe tools:

- Return a string. That's what gets sent back to the model.
- Give every argument a simple type hint (`str`, `int`, `float`). Gate 2 checks values against it, so a missing hint breaks validation. Note that Python treats `True` as an `int`, so check for booleans yourself if an `int` argument matters.
- Check the meaning of each argument inside the tool too. Gate 2 only knows the type: it can't tell that `-5` participants is nonsense. See how `calculate` refuses anything that isn't simple arithmetic.
- Prefer read-only tools. A tool that sends email or deletes files should ask a human first.

---

## Which mode to use when

| Use | When | Example |
| --- | --- | --- |
| Chatbot | General conversation; the model's own knowledge is enough | Brainstorming, explaining a concept, casual chat |
| RAG | Answers must come from a specific document, and each question needs a single lookup | FAQ bot, "chat with my notes", policy lookup |
| Agent | Questions need several steps, exact calculations, or a choice of actions | Price quotes that combine several facts, assistants that use several data sources |
| Fine-tuning | The model must answer in a consistent style or habit, and prompting doesn't get it there reliably | A fixed answer format, a brand voice, always declining to guess |

Start with the simplest mode that works. Each step up costs speed and predictability. Fine-tuning comes last: try a better prompt first, and use RAG for anything that is a fact.

## Try it: the same questions in all three modes

Ask each question in every mode and compare the answers.

| Question | Chatbot | RAG | Agent |
| --- | --- | --- | --- |
| `How much does the Night Owl plan cost?` | Makes up a price | Rs. 9,000 per month | Searches, then Rs. 9,000 per month |
| `Which day can I bring my dog?` | Guesses | Fridays (Furry Friday) | Searches, then Fridays |
| `I'm a student. What would 6 months of Hot Desk cost, including the deposit?` | Guesses | Finds some facts, often gets the maths wrong | Searches, then calculates: 4500 × 0.8 × 6 + 3000 = Rs. 24,600 |
| `Hi!` | Greets you | Searches the document anyway | Greets you, no tools |
| `Who is the CEO of Google?` | Answers from training | Says it doesn't know | Says it only helps with Blue Lantern |
| `My name is Ravi.` then `What is my name?` | Ravi | Doesn't know (no memory) | Has the memory, but may refuse because the system prompt keeps it to Blue Lantern topics |

---

## Reproducing this project

The same files, versions and seeds give the same data and the same setup. Model answers and training loss can't match exactly, because language models and GPUs vary slightly from run to run.

### The tested setup

| Component | Version used |
| --- | --- |
| Laptop | Windows 11, 16 GB RAM, no NVIDIA GPU (models run on the CPU) |
| Python and packages | Python 3.14.2; `requirements-lock.txt`: `ollama==0.6.3`, `chromadb==1.5.9`, `streamlit==1.65.0` |
| Ollama | 0.40.3 |
| Ollama models | `gemma3:1b` (ID `97558784bdc9`), `nomic-embed-text` (`0a109f422b47`), `gemma4:e2b` (`b37049369adf`) |
| Fine-tuning notebook | Unsloth `Gemma4_(E2B)-Text.ipynb`, commit `b27cbf9c938011934e1aaa4f4ee0e5710eaf250d` |
| Fine-tuning hardware | Google Colab, Tesla T4 |
| Fine-tuning packages | `finetune/colab_versions.txt`: torch 2.11.0+cu130, transformers 5.5.0, trl 1.15.0, peft 0.21.1, unsloth 2026.10.4, datasets 4.3.0 |
| Base model | `unsloth/gemma-4-E2B-it`, loaded in 4-bit |
| Training settings, first run | `r = 8`, `lora_alpha = 8`, 3 passes, learning rate 2e-4, batch 1 × gradient accumulation 4. This is the run in `blue_lantern_adapter.zip` |
| Training settings, second run | `r = 16`, `lora_alpha = 16`, 8 passes (96 steps), the same learning rate and batch, learning from the answers only. This is the run that worked |

`ollama list` shows each model's ID. If yours differs, you have a newer build of that model and answers may differ.

### Rebuild it

```
python -m pip install -r requirements-lock.txt
ollama pull gemma3:1b
ollama pull nomic-embed-text
ollama pull gemma4:e2b
cd finetune
python make_dataset.py
```

`make_dataset.py` writes the same three files, byte for byte, on any computer: the examples and the split are fixed in the script, the random choices use a fixed seed, and the line endings are fixed. Check them with `Get-FileHash blue_lantern_*.jsonl` (Windows) or `sha256sum blue_lantern_*.jsonl`:

| File | SHA-256 |
| --- | --- |
| `document.txt` | `5dcce8cefab8513f3826af0b1b56be9c73fb8312733c113cb16f8c66de478757` |
| `blue_lantern_train.jsonl` | `fd485e40e0e3e79ddf47ce45e5ffcbfa19d2676e58ff6b25a8a3477f834a6b71` |
| `blue_lantern_validation.jsonl` | `e8098a0aa2cab8707bacf0b9a8f699bbb6293465a84b278a34c93c98745fadd8` |
| `blue_lantern_test.jsonl` | `0137577362810e9cfdffaff4a17639e18ef5846aacf3867ec1d3ddd069058e0f` |

To record your own Colab versions, run this after the model has loaded and save the output as `finetune/colab_versions.txt`:

```python
import unsloth  # must be imported before transformers, trl and peft
import datasets, peft, torch, transformers, trl
for module in (torch, transformers, trl, peft, unsloth, datasets):
    print(f"{module.__name__}=={module.__version__}")
print("GPU:", torch.cuda.get_device_name(0))
```

### What won't match exactly

- **Training and validation loss** differ slightly between runs and GPUs. Expect the same pattern.
- **Fine-tuned answers** can differ in wording, and a score can move by one.
- **Answers in the app** vary in wording, and the agent can take a different path of tool calls. Judge by the facts and the steps.
- **Speed** depends on the hardware. On the tested laptop, agent answers took about 5 to 20 seconds.
