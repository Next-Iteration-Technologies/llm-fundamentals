# Ask your documents

*A hands-on story in ten short chapters. You don't need to read any code. Copy each command, run it, and look at what comes back.*

---

## Before you start

Every team has documents with answers in them: manuals, guides, meeting notes, old mail. Finding the right page takes too long.

So you'll build an assistant. You ask it a question, it finds the right pages in your documents, and it answers from those pages. That way of working has a name: **RAG**, short for *retrieval-augmented generation*. "Retrieval" means finding the pages. "Generation" means writing the answer.

In this lab you'll build it, test it, and make it better, one question at a time.

### What you'll be able to do

By the end of the lab you can:

1. Name each step of a RAG pipeline and say what goes in and out of it
2. Explain why one good answer proves nothing, and read Recall@3 and MRR
3. Pick a chunking for a source and explain why, from what the chunks actually look like
4. Say when keyword search beats semantic search, and what rerank can and can't fix
5. Explain what an embedding model changes, and when a better one is worth it
6. Explain why a RAG answer can be correct and still out of date
7. Separate "found the right passage" from "gave the right answer" (the RAG triad)
8. Read a miss and say which part of the pipeline caused it
9. Decide when RAG is worth it compared with sending everything

### Pick your documents

The lab comes with two sets of documents. Each one has a list of questions where we already know which pages hold the answer.

| Name | What's in it | Questions |
|---|---|---|
| `dnd-srd` | The free rulebook of the game Dungeons & Dragons. You don't need to know the game: treat it like any manual. | 60 (4 of them it can't answer) |
| `contoso` | A made-up company's notes, mail, guides and logs, from E1–E11 | 18 (2 of them it can't answer) |

Any other set of documents works too, once it has its own settings file (see `docs/add-a-dataset.md`).

Choose one for the whole lab. Start with `dnd-srd`: it has more questions, which you'll see matters in Chapter 2.

Every command runs in the `python` folder of `rag-exercises`, the same place as E1–E11. On macOS, type `python3` where it says `python`.

```
cd rag-exercises/python
export RAG_DATASET=dnd-srd          # macOS
$env:RAG_DATASET = "dnd-srd"        # Windows (PowerShell)
```

Every command below now uses those documents. To switch later, set it again. A new terminal forgets it: set it again there.

Keep `notebook.md` open next to this file. At the end of each chapter you'll write one or two lines in it.

---

## Chapter 1. Ask it something

*By the end of this chapter you can: Name each step of a RAG pipeline and say what goes in and out of it.*

*Builds on E1–E3: the same steps you ran one at a time there, now in one go, on any set of documents.*

You type a question. A few seconds later, an answer comes back with references to the pages it used. What happened in between?

**Your guess:** before the assistant can answer, what do you think it has to do first? Write down three steps.

**Try it:** ask the first question from your documents' question list.
```
python rag.py walkthrough --question 1
```

The first time, this takes a while: every chunk has to be turned into numbers once (step 3). After that, it's saved. A model on your own laptop can also take a minute to write the answer.

**Look for:** the eight steps it prints, each with its name in capitals.

1. **LOAD:** read the documents
2. **CHUNK:** cut them into small pieces, called **chunks**
3. **EMBED:** turn each chunk into numbers that stand for its meaning. These numbers are called an **embedding**.
4. **SEARCH:** find the chunks closest to the question
5. **RERANK:** re-sort the best ones with a second, more careful reader. In this first setup it's switched off, so you only see what it *would* do. Chapter 4 switches it on.
6. **PROMPT:** build the message for the model: instructions, the chunks it found, and the question. This is exactly what the model reads.
7. **ANSWER:** what the model wrote, and which passages it cited
8. **CHECK:** did it find the passages we know contain the answer? Each one says **found (rank N)** or **not found**. In steps 4 and 5, a `*` marks them.

Try a question of your own too:
```
python rag.py walkthrough "your question here"
```
Step 8 is skipped for your own questions, because nobody has marked their answer yet.

**Talk about:** which step surprised you? Which one do you think goes wrong most often?

**Write down:** the eight steps, in your own words.

---

## Chapter 2. Was that luck?

*By the end of this chapter you can: Explain why one good answer proves nothing, and read Recall@3 and MRR.*

*Builds on E4: the same scores, now on the documents you picked, with a check of the test itself.*

The answer looked good. But that was one question. Would you trust a new colleague after one good answer?

To know if the assistant is good, we need a test with an answer key: a fixed list of questions where we already know which pages hold the answer. This is called a **golden set**. Both document sets come with one.

**Your guess:** if the assistant finds the right pages for 40 of 55 questions, is that good?

**Try it:**
```
python rag.py eval
```

It runs every question in the golden set through the same steps as Chapter 1 (without writing answers, so it's quick) and gives each one a score. One line per question, then the averages.

**Look for:** two scores.
- **Recall@3:** of the right pages, how many were in the top 3 results?
- **MRR:** how high was the first right page? First place scores 1, second 0.5, third 0.33.

A `✗` marks the questions where a right page did not make the top 5. You'll need those in Chapter 3.

But wait: can we trust the test itself? Two quick checks.
- A **perfect student**, who always picks the right pages, must score 100%.
- A **random guesser** must score close to 0%.

```
python rag.py eval --retriever perfect
python rag.py eval --retriever random
```

The first prints **PERFECT student**, the second **RANDOM guesser**. If either check fails, the test is broken, not the assistant.

**Look at the number of questions** at the top of the output (only questions with known right pages are scored). With 55 questions, one question moves the score by about 2%. With 17, by about 6%. The last line of the output says it for your documents. The fewer questions, the harder it is to tell a better setup from a lucky one.

**Talk about:** how many questions would *you* want before trusting a score?

**Write down:** your first Recall@3 and MRR, in the "My starting point" table of your notebook. This is your **starting point**: `eval` saved it under the name `baseline`. Everything after this is measured against it.

---

## Chapter 3. The answer was cut in half

*By the end of this chapter you can: Pick a chunking for a source and explain why, from what the chunks actually look like.*

*Builds on E1 (naive vs structure-aware cutting) and E11 (small pieces that hand over their whole section).*

Look at one question the assistant got wrong. In the output of `eval`, they're marked `✗`. Question 3 is one of them in both sets of documents; any other number from your `✗` list works too.

```
python rag.py show 3
```

For each right passage (**GOLD**), you see its text, and under it the chunk the assistant found (**FOUND rank N**). The text they share is marked `»like this«`. The line above the chunk says how much of the passage it holds. Often the right passage *was* found, but only half of it: the cut went right through the middle.

How you cut documents into chunks is called **chunking**. There are several ways:

| Way | Like… |
|---|---|
| **Fixed size** | Cutting a book every 200 words, mid-sentence if need be |
| **Recursive** | Cutting at headings first, then paragraphs, then sentences |
| **By sentence** | A few sentences per piece |
| **Semantic** | Cutting where the topic changes |
| **Hierarchical** | Searching small pieces, but handing over the whole section they came from |
| **Structure-aware** (`contoso` only) | Cutting each kind of document its own way: one row of a sheet, one mail, one log line |
| **Contextual** | Any of the above, plus one sentence written by a model saying where the piece comes from |

**Your guess:** which way will find the most right passages in your documents?

**Try it:**
```
python rag.py chunks
python rag.py compare --phase chunking
```

The first command shows what the chunks look like: how many, how big (in **tokens**, roughly words), and how many right passages got **cut in two**. The second scores every way of cutting with the same search and the same embedding model, so any difference comes from the cutting. The first time it runs, it turns every new kind of chunk into numbers, which takes a few minutes.

**Look for:**
- The same scores as Chapter 2, one row per way of cutting. **won / lost** lists the questions (by number) that got better or worse against the row above.
- Bigger chunks hold more text, so they win a little just by being bigger. The **R@2k** column is fairer: every way of cutting gets the same amount of text, about 2,000 tokens.
- Contextual cutting needs a model call for every chunk, so it's left out. Your trainer will show it.

**Talk about:** were you right? What did the winner do that the others didn't? Would the same way win on the other set of documents?

**Write down:** the best way of cutting, and one reason why.

---

## Chapter 4. It was there, just too far down

*By the end of this chapter you can: Say when keyword search beats semantic search, and what rerank can and can't fix.*

With better cutting, look at the misses again. Something new shows up: the right chunk now *exists*, but it came 9th. The assistant only reads the top few.

So cutting isn't enough. We also need to **search** better.

There are two basic ways to search:
- **Keyword search** looks for the same words. Good for names and codes. Bad when the question uses different words.
- **Meaning search** looks for the same meaning. Good for different words, bad for exact names.

| Way to search | What it does |
|---|---|
| Keyword | Same words |
| Meaning | Same meaning |
| **Hybrid** | Both, with their rankings merged |
| Hybrid + **rerank** | A careful second reader re-sorts the top 50 |
| **Rewrite the question** | A model rephrases the question before searching |

**Your guess:** are your documents full of names and codes, or explanations? Which search do you expect to do well?

**Try it:**
```
python rag.py compare --phase retrieval
```

Each row shows which questions it **won** and **lost** compared with the row above.

**Talk about:** rerank can move a right chunk to the top. Can it find a chunk that the first search missed? Why not?

**Write down:** your best way to search, and what rerank did.

---

## Chapter 5. Same meaning, different words

*By the end of this chapter you can: Explain what an embedding model changes, and when a better one is worth it.*

Some questions still fail when they use different words from the documents. Meaning search should catch those. So why doesn't it?

Meaning search depends on the model that turns text into numbers, the **embedding model**. A weak one gets meaning wrong.

**Your guess:** will a better embedding model help a lot, a little, or not at all?

**Try it:**
```
python rag.py compare --phase embedding
```

**Talk about:** was the improvement worth a slower, bigger model?

**Write down:** which embedding model you'd pick, and why.

---

## Chapter 6. Two documents disagree

*By the end of this chapter you can: Explain why a RAG answer can be correct and still out of date.*

*This chapter needs documents with dates, so it uses `contoso` whichever set you picked.*

Two of Contoso's documents disagree. A guide says logs are kept 14 days. The system's settings say 30. Both are in the documents. Which one should the assistant believe?

**Your guess:** which will it pick, and why?

**Try it:**
```
python rag.py compare --phase filters --dataset contoso
```

This adds labels to the search, such as date, platform and who is asking, plus a rule: *when two sources disagree, prefer the newest.*

**Talk about:** a right answer from an old document is still a wrong answer. Where could this happen in your own documents?

**Write down:** one way the assistant can be correct and still out of date.

---

## Chapter 7. Right pages, wrong answer?

*By the end of this chapter you can: Separate "found the right passage" from "gave the right answer" (the RAG triad).*

So far we've only checked whether the assistant found the right pages. Now look at what it actually *says*.

Three checks, called the **RAG triad**:
- Did it find pages that fit the question?
- Is everything it says **backed by those pages**, or did it make something up?
- Does it **actually answer** the question?

And one more: when the answer isn't in the documents, does it say "I don't find this", or does it invent one?

**Your guess:** if the search gets 20% better, do the answers get 20% better?

**Try it:**
```
python rag.py answers --best --limit 10
```

**Talk about:** a model already knows a lot about popular topics, like a well-known game. How could that make a bad search look good?

**Write down:** one answer that was right for the wrong reason, or wrong despite the right pages.

---

## Chapter 8. Why does it still miss?

*By the end of this chapter you can: Read a miss and say which part of the pipeline caused it.*

Even the best setup misses some. Each miss has a cause, and each cause points back to a chapter:

| What happened | Go back to |
|---|---|
| The right passage was cut apart | Chapter 3, cutting |
| Found, but too far down | Chapter 4, searching |
| Not found at all | Chapter 4 or 5 |
| Everything came from the wrong document | Chapter 5, meaning |
| Only part of a multi-part answer was found | Chapter 3 or 4 |

**Try it:**
```
python rag.py misses --best
```

**Talk about:** which cause is most common? What would you try next?

**Write down:** the most common cause, and your next fix.

---

## Chapter 9. Why not give it everything?

*By the end of this chapter you can: Decide when RAG is worth it compared with sending everything.*

Today's big models can read about 750,000 words at once. Your documents might fit. So why search at all? Why not hand over every document and let the model find the answer?

**Your guess:** which gives better answers, the assistant with search, or the model with everything? Which costs more?

**Try it:** first with 5 questions. It shows the cost before it sends anything.
```
python rag.py final --limit 5
```

Same model, same questions, two ways:
- **With search:** a few pages
- **With everything:** all the documents, every time

**Talk about:** what if your documents were a thousand times bigger, like a whole company's archive? What changes then?

**Write down:** when you'd use search, and when you'd give it everything.

---

## Chapter 10. Before and after

Ask the question from Chapter 1 again, with your best setup, next to your starting point:

```
python rag.py walkthrough --question 1 --compare-with baseline
```

What changed at each of the eight steps?

If you have time, set `RAG_DATASET` to the other set of documents and run Chapters 2–4 again. Do the same choices win?

**Write down:** the three choices that made the biggest difference, in order. That's what you'd tell the next team building one of these.
