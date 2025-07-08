# UKLns 
### (Under Kilo LiNeS) 

An experimental toolkit to **chat with** and **search** small GitHub repositories—without ever cloning them locally.

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

---

## Demo
![UKLns Demo](assets/Demo.gif)

## Overview

When exploring GitHub, you’ll often find good small project. UKLns (short for **Under Kilo Lines**) collects these projects into a JSON index and lets you interact with them via:

1. **Semantic search** using vector embeddings—just describe what you’re looking for.  
2. A custom **RAG‑based chat system** that retrieves relevant code snippets and README content on‑the‑fly.  
3. Convenient commands (e.g., to show  README) so you never have to clone an entire repo.

This makes UKLns ideal for learning, quick prototyping, or anytime you want to peek into small projects without clone it.

### Model
--> "all-MiniLM-L6-v2" --- Semantin Search JSON  
--> "all-MiniLM-L6-v2" --- RAG retrival (maybe change it)  
--> "Qwen3-0.6B" --- Chat  

---

## Features

| Feature                      | Description                                                    |
| ---------------------------- | -------------------------------------------------------------- |
|  Vector embedding search   | Find repos by natural‑language description                     |
|  RAG chat interface        | Ask questions about any indexed project                        |
|  Inline README display     | Render a repo’s README in your terminal                        |
|  Zero‑clone retrieval      | Grab specific files or code snippets without `git clone`       |
|  Local model orchestration | Runs models on your machine; Downloads directly from Hugging Face if missing |

---

## Installation

1. **Clone this repo**:

   ```bash
   git clone https://github.com/tucob97/uklns.git
   cd uklns
    ````

2. **Set up a virtual environment**:

   ```bash
   python3 -m venv .venv
   source .venv/bin/activate
   ```

3. **Install requirements**:

   ```bash
   pip install -r requirements.txt
   ```

4. **Configure your JSON index**:

   * Add GitHub repo URLs (small projects if you like it) into `repos.json`.

---

## Quick Start

```bash
# Simply launch the program
python main.py 

```

---

## ⚠️ Disclaimer

UKLns will automatically download any missing embedding or language model from Hugging Face the first time you run it. Ensure you have an internet connection and sufficient disk space. Models remain cached in your local HF cache directory.

---

## Requirements

See `requirements.txt` for full dependency list. Key packages include:

* `gitingest` — index GitHub repos without cloning
* `sentence-transformers` — generate vector embeddings
* `transformers` — local language model inference

---

## License

This project is MIT‑licensed © Edoardo Failla. See [LICENSE](LICENSE) for details.

---

## Contributing

Feel free to open issues or submit PRs. Share new small‑code repos to include in `repos.json` or improve the RAG logic!

## License

This project is licensed under the MIT License © Edoardo Failla.  
See [LICENSE](LICENSE) for details.
