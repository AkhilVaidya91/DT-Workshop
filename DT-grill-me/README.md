# Design Thinking Grill

A minimal Gradio chat app that uses Gemini to ask one focused design-thinking question at a time, then turns the conversation into a downloadable Markdown design brief.

## Setup

Install dependencies:

```powershell
python -m pip install -r requirements.txt
```

Set `GEMINI_API_KEY` in the environment, or put `GEMINI_API_KEY=your-key` in `.env` (the app also checks `.env.example`, matching the neighboring idea-generator's key lookup behavior).

## Run

```powershell
python app.py
```

The question cap is `MAX_QUESTIONS` in `app.py` (20 by default). After the user answers the last question, Gemini writes a design brief from the full conversation and the app offers it as a `.md` download.
