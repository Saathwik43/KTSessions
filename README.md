# Ollama Demos: Chatbot, Chat-with-a-Document (RAG), Agent and Fine-tuning

Three small projects that run an AI model entirely on your own computer using Ollama. No API key, no internet needed once set up, and nothing leaves your machine.

A fourth step, fine-tuning, trains the model on our own examples (section 3). It is the only part that needs the internet: the training runs on Google Colab's free GPU.

| File | What it is |
| --- | --- |
| `chatbot.py` | A chatbot in the terminal |
| `rag.py` | Ask questions about a document, in the terminal |
| `agent.py` | An agent that searches the document and does maths with tools, in the terminal |
| `app.py` | All three in a browser interface |
| `GUIDE.md` | How the three modes differ, how each was built, how to customise them, and how fine-tuning fits in |
| `DAY4_SESSION.md` | A 120-minute plan for teaching fine-tuning through this repository |
| `finetune/make_dataset.py` | Builds the fine-tuning data (see section 3) |
| `finetune/blue_lantern_train.jsonl` | 46 examples the model trains on |
| `finetune/blue_lantern_validation.jsonl` | 5 examples checked during training, never trained on |
| `finetune/blue_lantern_test.jsonl` | 9 locked test questions, used only to compare models at the end |
| `finetune/blue_lantern_adapter.zip` | The trained LoRA adapter from a Colab run (50 to 100 MB). Not in this repository: model files are kept out of git, and you get your own by following section 3 |
| `finetune/colab_versions.txt` | The package versions Colab used for that run |
| `document.txt` | Sample document (a made-up coworking space handbook) |
| `requirements.txt` | Python packages needed |
| `requirements-lock.txt` | The same packages at the exact versions this project was tested with |

## 1. One-time setup

You need Python 3.10 or newer.

**Step 1. Install Ollama** from https://ollama.com/download and open it. It runs quietly in the background (look for the llama icon in the system tray).

**Step 2. Download the models.** Open a terminal and run:

```
ollama pull gemma3:1b
ollama pull nomic-embed-text
ollama pull gemma4:e2b
```

- `gemma3:1b` (about 800 MB) is the chat model that writes the answers.
- `nomic-embed-text` (about 270 MB) is the embedding model that turns text into numbers for searching.
- `gemma4:e2b` is the agent model. It supports tool calling, which `gemma3:1b` does not. Only needed for the agent demo.

**Step 3. Install the Python packages.** In the project folder:

```
python -m pip install -r requirements.txt
```

To reproduce the tested setup exactly, install `requirements-lock.txt` instead. [GUIDE.md](GUIDE.md#reproducing-this-project) lists every version, model ID and checksum to compare against.

**Step 4. Check everything is ready:**

```
ollama list
```

You should see all three models listed.

## 2. Run the demos

Open a terminal in the project folder first.

### Demo 1: Chatbot (terminal)

```
python chatbot.py
```

Type a message and press Enter. Type `exit` to quit.

Try this to show that it remembers the conversation:

1. `My name is Ravi.`
2. `What is my name?`

### Demo 2: Chat with a document (terminal)

```
python rag.py
```

It reads `document.txt`, then waits for your questions. Try:

- `How much does the Night Owl plan cost?` (answer: Rs. 9,000 per month)
- `Which day can I bring my dog?` (answer: Fridays)
- `Who is the CEO of Google?` (it should say it doesn't know, because that is not in the document)

Before each answer it prints which chunks of the document it picked. A smaller distance means a closer match.

To use your own text file:

```
python rag.py my_notes.txt
```

### Demo 3: Agent (terminal)

```
python agent.py
```

The agent has two tools: `search_handbook` (searches `document.txt`, like `rag.py`) and `calculate` (does exact arithmetic). The model decides which tools to call, and every call is printed so you can watch it work. Try:

- `What time does the space open on Sunday?` (one search)
- `I'm a student. What would 6 months of Hot Desk cost, including the deposit?` (search, then calculate: 4500 * 0.8 * 6 + 3000 = Rs. 24,600)
- `Hi!` (no tools needed)
- `Who is the CEO of Google?` (it should refuse, because it only answers Blue Lantern questions)

Before any tool runs, the agent checks the tool name against an allowlist and the arguments against the tool's signature. If the model asks for something invalid, it gets an `Error: ...` message back and can correct itself. See [GUIDE.md](GUIDE.md) for the details.

### Demo 4: Browser interface

```
python -m streamlit run app.py
```

A browser tab opens at http://localhost:8501. The first time, Streamlit may ask for your email in the terminal; just press Enter to skip.

- Use the sidebar to switch between **Chatbot**, **Chat with a document** and **Agent**.
- In document mode you can upload your own `.txt` or `.md` file from the sidebar. Without an upload it uses `document.txt`.
- Open **Retrieved chunks** above an answer to see the text the model was given.

Press `Ctrl + C` in the terminal to stop it.

## 3. Fine-tuning

The three demos change what the model is **given**: instructions, document chunks, tools. Fine-tuning changes the **model itself**, by training it a little more on our own examples.

**What we teach it:** how to answer, not new facts. The facts stay in `document.txt`, where RAG can use them.

- **A fixed answer format:** every answer as *Answer / Details / Source*.
- **A fixed "I don't know" reply:** when the handbook doesn't say, it never guesses and sends the member to the front desk.

**The steps.** Every cell to run and edit is in [GUIDE.md](GUIDE.md#fine-tuning-changing-the-model-itself).

1. **Prepare the data** on your laptop: training, validation and a locked test set.

   ```
   cd finetune
   python make_dataset.py        # Train: 46 / Validation: 5 / Test: 9, leakage check passed
   ```

2. **Set up Unsloth's official Gemma 4 notebook** on Google Colab's free T4 GPU, with the model loaded in 4-bit (QLoRA), and upload the three data files.
3. **Measure the baseline:** the untrained model on the 9 test questions, with the app's prompt and with a better prompt.
4. **Train** for 8 passes, after checking that the model learns only from the answer part of each example.
5. **Evaluate** the fine-tuned model on the same test questions and compare.
6. **Save the adapter**, the small trained add-on, as `blue_lantern_adapter.zip`.
7. **Use it in the chatbot.** Ollama can't load an adapter directly, so it is merged into the base model, converted to GGUF and pulled into Ollama:

   ```
   ollama pull hf.co/YOUR-USERNAME/blue-lantern-gemma4
   ollama cp hf.co/YOUR-USERNAME/blue-lantern-gemma4 blue-lantern
   ```

   Once Ollama has a model named `blue-lantern`, *Chat with a document* shows a **Use the fine-tuned model** switch in the sidebar.

**Where we are.** The first training run didn't work: the loss fell, but the model answered exactly like the untrained one. The second run did: on questions it never saw, the model now answers in the *Answer / Details / Source* format with the right facts, and general questions are unaffected. The "I don't know" reply is only half learned. The guide's *Results so far* has both runs, the loss table and the test answers. Step 7 is under way.

**What you need:** a Google account for Colab. For step 7, a Hugging Face account with a *write* access token, and about 5 GB of free disk space.

## 4. How it works

### Ollama as an API

Ollama runs a local web server at `http://localhost:11434`. Any program can send it a request and get a reply, just like a cloud AI API. The `ollama` Python package used here is a thin wrapper around it. You can call it directly too:

```
curl http://localhost:11434/api/chat -d "{\"model\": \"gemma3:1b\", \"stream\": false, \"messages\": [{\"role\": \"user\", \"content\": \"Hello\"}]}"
```

### The chatbot

The model has no memory. The script keeps a list of every message so far and sends the whole list on each turn. That is the only reason it "remembers" your name.

### RAG (Retrieval-Augmented Generation)

The model was never trained on your document, so we look up the relevant parts and hand them over with the question.

1. **Chunk** - split the document into pieces of 500 characters, each overlapping the previous one by 100.
2. **Embed** - turn each piece into a vector (a list of numbers that captures its meaning) with `nomic-embed-text`.
3. **Store** - save the vectors in ChromaDB, the vector database.
4. **Retrieve** - embed the question the same way and ask ChromaDB for the 3 closest pieces.
5. **Generate** - send those 3 pieces plus the question to `gemma3:1b` and tell it to answer only from them.

### The agent

In RAG, our code always searches and then asks the model. An agent flips that: the model decides what to do.

1. **Ask** - send the conversation plus a list of tools (normal Python functions) to the model.
2. **Decide** - the model replies either with a final answer or with a request like `calculate("4500 * 0.8")`.
3. **Run** - our code runs the tool only if it passes three gates: the name is on the allowlist (`TOOLS`), the arguments match the tool's signature (`validate_args`), and any crash is turned into an error message instead of stopping the app. The model never runs code itself.
4. **Repeat** - the result is added to the conversation and we go back to step 1, up to `MAX_STEPS` times so it always stops.

### Fine-tuning

The model's habits are stored in its weights, the billions of numbers learned in training. Fine-tuning adjusts them with our own examples, so the new behaviour no longer depends on instructions in the prompt. We use **QLoRA**: instead of changing all the weights, we train a small add-on (LoRA) and leave the original model frozen, stored in 4-bit to save memory. That is why a free Colab GPU is enough. The result is the add-on itself, called an adapter. To run it in Ollama, the adapter is merged into the base model and converted to a single GGUF file. See [GUIDE.md](GUIDE.md) for when fine-tuning is the right tool and when it isn't.

## 5. Things you can change

All settings are at the top of each script.

| Setting | What it does |
| --- | --- |
| `CHAT_MODEL` | Swap in another model, for example `llama3.2` or `gemma3:4b`. Run `ollama pull <name>` first. Bigger models answer better but are slower. |
| `FINETUNED_MODEL` | The Ollama name of the fine-tuned model (`blue-lantern`). When Ollama has it, *Chat with a document* offers a switch to use it. |
| `CHUNK_SIZE`, `CHUNK_OVERLAP` | How the document is cut up. |
| `TOP_K` | How many chunks are given to the model per question. |
| `AGENT_MODEL` | The agent's model. It must support tool calling (for example `qwen3` or `llama3.2`). |
| `MAX_STEPS` | How many rounds the agent may take before it gives up. |
| `MAX_ARG_LENGTH` | The longest text argument the agent may pass to a tool. |

## 6. Troubleshooting

| Problem | Fix |
| --- | --- |
| `No module named 'streamlit'` (or `ollama`, `chromadb`) | Run `python -m pip install -r requirements.txt` and wait for it to finish. |
| `ollama` is not recognized as a command | Close and reopen the terminal. If it still fails, use the full path: `& "$env:LOCALAPPDATA\Programs\Ollama\ollama.exe" list` |
| Connection refused / failed to connect to Ollama | Ollama is not running. Open it from the Start menu. |
| `model not found` | Run the `ollama pull` commands from Step 2. |
| The first answer is slow | Normal. The model is being loaded into memory; later answers are faster. |
| `streamlit` is not recognized as a command | Use `python -m streamlit run app.py` instead of `streamlit run app.py`. |
| Answers from the document are wrong or vague | `gemma3:1b` is a very small model. Try a larger `CHAT_MODEL`, or raise `TOP_K`. |
| Only `.txt` and `.md` files work | By design, to keep the code short. Convert PDFs or Word files to text first. |
| `Error: LoRA adapters are no longer supported` | Ollama can't load an adapter. Merge it and convert to GGUF first (section 3, step 7). |
| No **Use the fine-tuned model** switch | Ollama has no model named `blue-lantern`. Run the `ollama cp` command in section 3, step 7, and check with `ollama list`. |
