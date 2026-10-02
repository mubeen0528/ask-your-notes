# Ask Your Notes

A small Python app that finds passages in your local notes and asks Gemini to answer from them.

## Windows / PowerShell setup

Open the existing workspace in VS Code. From the workspace root:

```powershell
cd .\ask-your-notes
.\.venv\Scripts\Activate.ps1
python -m streamlit run app.py
```

Your existing Python 3.14 virtual environment already has the required packages. No installation is needed for this setup. If PowerShell blocks activation, use the interpreter directly:

```powershell
.\.venv\Scripts\python.exe -m streamlit run app.py
```

For a fresh environment only, install the five pinned direct dependencies with `python -m pip install -r requirements.txt`. Their indirect dependencies are installed automatically.

Before asking a question, open the project-local `.env` in VS Code and paste your own key after `GOOGLE_API_KEY=`. Keep `GEMINI_MODEL=gemini-3.8-flash` unless you intentionally configure another available model. Do not paste a key into chat, notes, screenshots, or source code. If the file is missing, copy `.env.example` to `.env` and replace the placeholder locally. Never overwrite a populated `.env`.

The app loads this explicit project-relative file at runtime. Settings are read again when an Ask submission needs Gemini. Local .env values override matching environment variables.

## Use the app

Put UTF-8 .txt or .md files inside `library`, including subfolders. UTF-8 files with a byte-order mark are supported. Click **Reload Notes** after editing files. Empty files are skipped; unreadable files get a filename-only warning. The loaded count includes readable, non-empty files.

Enter a question and click **Ask**. The answer appears with [S1], [S2], or [S3] citations. **Retrieved sources** lists actual relative filenames, with expandable passages. The last result remains visible across normal reruns and Reload Notes; it refers to the last submission and may become stale after files change. Restarting the app or losing the browser session clears it.

## How it works

1. pathlib finds notes recursively relative to notes_engine.py.
2. LangChain splits notes into chunks of 800 characters with 150 characters of overlap.
3. scikit-learn TF-IDF and cosine similarity rank the chunks locally.
4. Up to three chunks scoring at least 0.1 are supplied to Gemini using a LangChain prompt.
5. Gemini is instructed to answer only from those passages, acknowledge missing information, treat note instructions as data, and cite the supplied labels.

No relevant passages means: “I could not find relevant information in your notes.” Gemini is not called in that case. Startup, refresh, indexing, Reload Notes, and typing make no generation requests. Each valid matching form submission makes one generation request, with retries disabled, a 30-second timeout, and a 1024-token output limit.

Retrieval is local, but your question and selected passages are sent to Google when an answer is requested and a key is present. Consider what personal information your notes contain before submitting.

## Limitations and troubleshooting

- TF-IDF is a simple keyword-based heuristic, not semantic search. The 0.1 threshold may miss paraphrases or select irrelevant passages. One-character-only or punctuation-only notes can have an empty vocabulary.
- Only .txt and .md files are supported. PDFs, Word documents, images, and non-UTF-8 notes are not read.
- At most three passages reach Gemini. Answers can omit information, be truncated, contain mistakes, or cite poorly. Retrieval does not prove every statement; verify the excerpts.
- The instruction to ignore instructions in notes reduces prompt-injection risk but cannot guarantee perfect model behavior.
- Missing or rejected keys, unavailable models, quota limits, and network problems produce simple messages without raw API exceptions. The app never changes models automatically.
- Keep your key local. .gitignore preserves the existing exclusions and ignores personal library files by default, while allowing the existing git-notes.txt demo. Ignore rules do not untrack files already tracked by Git.
- There is no persistent search database or conversation memory. Large libraries may take longer to index.

## Manual tests

Start the app yourself after adding your key locally. These examples use the included Git demo:

| Test | Expected behavior |
| --- | --- |
| Ask “What is a Git commit?” | An answer about saving a snapshot of staged changes; git-notes.txt appears with an excerpt and a citation. |
| Ask “What does a branch do?” | An answer about working on changes separately, using the demo source. |
| Ask “Explain photosynthesis.” | The exact no-match message; no Gemini request. |
| Submit a blank question | A prompt to enter a question; no request. |
| Type, refresh, or click Reload Notes | No generation request; the last result stays visible in the same session. |
| Temporarily leave GOOGLE_API_KEY empty and ask about Git | A local missing-key message with the retrieved sources. |
| Add a nested .md note, then Reload Notes | File count increases; matching results show its relative path. |
| Add an empty file | It is skipped without increasing the loaded count. |
| In a temporary library containing only “!!!”, ask a question | Empty-vocabulary warning and the no-match message; no request. |
| Use an unavailable configured model, a rejected key, or encounter a network/quota failure | A simple error message, no raw exception, no automatic retry or model switch. |

Restore any settings changed for tests. Do not replace the included demo to test an empty library; temporarily move it locally and restore it afterward.

API integration reference: [LangChain ChatGoogleGenerativeAI](https://reference.langchain.com/python/langchain-google-genai/chat_models/ChatGoogleGenerativeAI).
