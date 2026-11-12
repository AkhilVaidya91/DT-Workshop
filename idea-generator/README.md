# Idea Generator

A minimal two-part application for generating and browsing ideas.

## Setup

From this directory, install the dependencies:

```bash
python -m pip install -r requirements.txt
```

Add your Gemini API key directly to `.env.example`:

```text
GEMINI_API_KEY=your-gemini-api-key
```

The generator reads this value directly from `.env.example`; it does not use an environment variable.

## Generate ideas

Generate the CSV with exactly one Gemini request:

```bash
python generate_ideas.py
```

The script prints progress while it waits for Gemini, saves the response to `gemini_response.json` for inspection, and validates it before writing `ideas.csv`. Existing data is preserved unless regeneration is explicitly requested:

```bash
python generate_ideas.py --overwrite
```

## Browse ideas

After generating the CSV, start the Gradio application:

```bash
python app.py
```

Each browser session receives random ideas without repeats until all ideas have been shown.
