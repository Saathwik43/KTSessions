# Day 4 Session Plan: Fine-Tuning, Taught Through This Repo

A 120-minute plan for the Day 4 mentor session. It covers the core of the Day 4 deck (fine-tuning, integration, pitch) by walking through this repository instead of the slides.

**The one message for the day:** fine-tuning changes how a model behaves, not what it knows, and you only know it worked if you test it.

## Before the session

- [ ] The app runs on your laptop: `python -m streamlit run app.py`
- [ ] The fine-tuned model is in Ollama under the name `blue-lantern`, so the **Use the fine-tuned model** switch shows in *Chat with a document*
- [ ] Your Colab notebook is open in a tab, with its outputs still visible. You won't train live
- [ ] `GUIDE.md` is open at [Fine-tuning](GUIDE.md#fine-tuning-changing-the-model-itself)
- [ ] Mentors have the repo link: https://github.com/Saathwik43/KTSessions.git

Nobody needs a GPU during the session. If a mentor asks to see training live and Colab has no GPU free, show the saved notebook outputs and say plainly that they are from an earlier run.

## The plan

| # | Min | Open | The point to make |
| --- | --- | --- | --- |
| 1 | 10 | `GUIDE.md` → the Fine-tuning intro | Prompting, RAG, an agent and fine-tuning each change something different |
| 2 | 5 | `GUIDE.md` → "What we teach, and how we measure it" | Pick one narrow behaviour and decide how to measure it before training |
| 3 | 15 | `finetune/make_dataset.py` | The examples are the behaviour |
| 4 | 15 | `GUIDE.md` → "Key ideas" and step 2 | LoRA and QLoRA in plain words, and why training runs on Colab |
| 5 | 10 | `GUIDE.md` → step 4 | Check what the model learns from before you train |
| 6 | 20 | `GUIDE.md` → "Results so far" | A falling loss proved nothing; the evaluation did |
| 7 | 10 | The app, live | The before and after, in the product |
| 8 | 10 | `GUIDE.md` → step 7 and "Where it falls short" | The adapter is the result; the model only drafts |
| 9 | 10 | `GUIDE.md` → "Reproducing this project" | If a stranger can't reproduce it, it isn't finished |
| 10 | 15 | This file → "The pitch" | Six parts, five minutes, with evidence |

### 1. Four ways to change what a model does (10 min)

Show the table at the top of the Fine-tuning section. Tie it to the earlier days: RAG gave the chatbot the handbook (Day 2), the agent gave it tools (Day 3), and today changes its habits.

- **Teach behaviour, not facts.** A price in the handbook changes: with RAG you edit `document.txt`. A fact trained into the model stays until you retrain.
- **Try a prompt first.** It is cheaper and easier to undo.

**Ask the room:** "A student wants to fine-tune the model on their college's fee structure. What do you tell them?" (Use RAG. Fees change.)

### 2. One behaviour, and how to measure it (5 min)

Show the two reply formats and the four success criteria: format, source, facts, no regressions. The criteria were written before training, so the results couldn't move the goalposts.

### 3. The data (15 min)

Open `finetune/make_dataset.py` and scroll through `TRAIN`, `VALIDATION` and `TEST`.

- Each line is a question, an answer and the handbook section it comes from.
- Three sets: the model **trains** on 46, 5 are **checked during training**, and 9 are a **locked test** used only at the end.
- The leakage check: a test question that is a near-copy of a training one makes results look better than they are.

**Do it live** (one second, no GPU):

```
cd finetune
python make_dataset.py
```

**Ask the room:** "Why can't we test the model on the same questions we trained it on?"

### 4. LoRA, QLoRA and Colab (15 min)

- **LoRA:** the model is a thick textbook. We don't rewrite it; we add sticky notes. Ours trained 0.49% of the model.
- **QLoRA:** the same, using the pocket edition of the textbook (4-bit) so it fits on a small GPU. In code it is one setting, `load_in_4bit = True`.
- **Unsloth** makes this fast enough for a free GPU. We changed its official notebook as little as possible.
- **Why Colab:** training needs an NVIDIA GPU. Most laptops don't have one.

Warn them: free Colab runs out of GPUs, and resets if you leave the tab idle.

### 5. Check before you train (10 min)

Show the last three lines of the step 4 cell and what they print: only the answer, no question, no handbook.

Each example is about 85% handbook text. If the model learns from the whole text, it gets better at repeating the handbook and the loss falls without the answers changing. That is what happened in our first run.

### 6. Two runs, one that failed (20 min)

This is the heart of the session. Tell it as a story from "Results so far".

- **Run 1:** the loss fell to 0.2. It looked like success. The model answered exactly like the untrained one: format 0 of 9.
- **Finding the cause:** four checks, each ruling something out, ending with "a question from its own training data still gets a plain answer".
- **Run 2:** learning from the answers only. The format appeared on questions the model had never seen, and general questions were unaffected.
- **The loss table:** validation loss was best at pass 3, then rose while training loss kept falling. That is overfitting, in real numbers.
- **What still isn't right:** the "I don't know" reply was only half learned. Say so.

**Ask the room:** "Looking at this loss table, where would you have stopped training?" (After pass 3.)

### 7. The demo (10 min)

In *Chat with a document*:

1. Switch **off**: ask "Which room fits a team of 12?" A plain sentence.
2. Switch **on**: the same question. *Answer / Details / Source*.
3. Switch **on**: "Do you have showers?" The "I don't know" reply.
4. Switch to *Chatbot* mode and ask "What is 15 percent of 200?" to show nothing else broke.

The fine-tuned model is slower than `gemma3:1b`, because it is bigger. Ask the next question while it answers.

### 8. From adapter to product (10 min)

- The result of training is an **adapter**: a small file loaded on top of the base model.
- Ollama can't load an adapter, so it is merged into the base model and converted to GGUF.
- In a real product **the model only drafts**. Code checks the draft against the source, and a person approves anything that gets published. The deck's DevRelay example works the same way: draft, verify, approve.

### 9. Reproducibility (10 min)

Show the tested-setup table, the checksums and `finetune/colab_versions.txt`. Point out what can be exact (data, versions, seeds) and what can't (loss values, wording).

**Ask the room:** "If I gave you only this repo, what would stop you from repeating our result?"

### 10. The pitch (15 min)

Six parts in five minutes, then two minutes of questions. Walk through the example below, then have two mentors pitch it back.

## The pitch, using this project

| Part | For Blue Lantern |
| --- | --- |
| 1. Problem | Members of a coworking space ask the same questions about plans, prices and rules all day, and the answers are buried in a handbook |
| 2. Target user | Members, and the community manager who answers them |
| 3. Solution | A chat assistant that answers from the handbook, shows where each answer came from, and says so when the handbook doesn't cover something |
| 4. Technical approach | Runs entirely on a laptop with Ollama. RAG (ChromaDB) supplies the facts. An agent with two tools handles multi-step questions with exact arithmetic. A QLoRA fine-tune of Gemma 4 gives every answer the same checkable format |
| 5. Demo and evidence | The student Hot Desk question: RAG says Rs. 27,000, the agent says Rs. 24,600. The fine-tuned switch off and on. The loss table and the test answers from `GUIDE.md` |
| 6. Roadmap | What works: all three modes and the fine-tuned format. What's left: the "I don't know" reply, and a full scored test. What's next: a tool that books meeting rooms |

How pitches are scored:

| Criterion | What good looks like |
| --- | --- |
| Problem clarity | A specific pain and a specific user, in one sentence |
| Technical correctness | An accurate account of the model, the data and the choices |
| Demo quality | One complete journey that actually works |
| Evidence | Claims backed by numbers or examples |
| Handling questions | Honest about the limits, clear on next steps |

## What to warn mentors about

1. **A low loss proves nothing.** Always test on questions the model hasn't seen.
2. **Students will want to fine-tune facts into the model.** Steer them back to RAG.
3. **Free Colab runs out of GPUs.** Have students check the evening before, and keep a trained model ready as a fallback. If you use it, say it is precomputed.
4. **Fine-tuning is the last thing to try**, after a better prompt and after RAG.
