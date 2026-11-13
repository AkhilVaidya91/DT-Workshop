import os
import tempfile
from pathlib import Path

import gradio as gr
from google import genai

MODEL = "gemini-4-hydrogen"
MAX_QUESTIONS = 20
BASE_DIR = Path(__file__).parent
KEY_FILES = [BASE_DIR / ".env", BASE_DIR / ".env.example"]


def load_api_key():
    key = os.environ.get("GEMINI_API_KEY", "").strip()
    if key:
        return key
    for path in KEY_FILES:
        if not path.exists():
            continue
        for line in path.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if line.startswith("GEMINI_API_KEY="):
                value = line.split("=", 1)[1].strip().strip('"').strip("'")
                if value:
                    return value
    raise RuntimeError("GEMINI_API_KEY not found in environment, .env, or .env.example.")


def conversation_text(history, latest_message):
    lines = []
    for item in history:
        role = item.get("role", "user").capitalize()
        content = item.get("content", "")
        if isinstance(content, str):
            lines.append(f"{role}: {content}")
    lines.append(f"User: {latest_message}")
    return "\n".join(lines)


def respond(message, history, asked_count):
    history = history or []
    asked_count = int(asked_count or 0)
    transcript = conversation_text(history, message)
    try:
        client = genai.Client(api_key=load_api_key())
        if asked_count >= MAX_QUESTIONS:
            prompt = f"""You are a rigorous design-thinking partner. Using the full conversation below, write a useful, specific product or service design brief in Markdown. Distinguish confirmed facts, user decisions, assumptions, and open questions; do not invent requirements. Include project overview, audience and context, problem and unmet need, current alternatives and limitations, design opportunity, objectives, requirements, constraints, success criteria, desired artifact, unresolved questions, and next recommended validation step. Clearly label anything unresolved.\n\nFull conversation:\n{transcript}"""
            response = client.models.generate_content(model=MODEL, contents=prompt)
            answer = (response.text or "").strip()
            if not answer:
                raise RuntimeError("Gemini returned an empty summary.")
            new_count = asked_count
            done = True
        else:
            prompt = f"""You are a critical design-thinking interviewer. Follow these rules: ask exactly one focused, high-priority question per turn; investigate the underlying problem before accepting a proposed solution; maintain a frontier so you do not ask downstream questions before prerequisites are settled; challenge vague claims and distinguish facts, assumptions, and user decisions; do not invent requirements; adaptively explore audience/context, unmet need, current alternatives, desired improvement, requirements/constraints, success criteria, validation, suitable artifact, and evaluation. Ask only what is useful for this idea. Make each question specific, explain why the decision matters when useful, offer reasonable options where helpful, and give a brief recommendation with reasoning. Use this format: **Q{asked_count + 1} - [short title]** followed by the question, then **My recommendation:** and a brief recommendation. Do not conclude or write the final brief yet.\n\nQuestion limit: {MAX_QUESTIONS}. Questions already asked: {asked_count}. Questions remaining including this one: {MAX_QUESTIONS - asked_count}. Ask exactly the next question, numbered Q{asked_count + 1}.\n\nConversation so far:\n{transcript}"""
            response = client.models.generate_content(model=MODEL, contents=prompt)
            answer = (response.text or "").strip()
            if not answer:
                raise RuntimeError("Gemini returned an empty question.")
            new_count = asked_count + 1
            done = False
    except Exception as error:
        answer = f"I couldn’t reach Gemini: {error}"
        new_count = asked_count
        done = False

    updated = history + [
        {"role": "user", "content": message},
        {"role": "assistant", "content": answer},
    ]
    download = None
    if done:
        with tempfile.NamedTemporaryFile(mode="w", encoding="utf-8", suffix=".md", delete=False) as file:
            file.write(answer)
            download = file.name
    return updated, new_count, gr.update(value=download, visible=done), "Done — your design brief is ready to download." if done else f"Question {new_count} of {MAX_QUESTIONS}"


def reset():
    return [], 0, gr.update(value=None, visible=False), "Describe your idea to begin."


with gr.Blocks(title="Design Thinking Grill") as demo:
    gr.Markdown("# Design Thinking Grill\nDescribe your idea. I’ll ask focused questions to help clarify the problem and shape a testable design brief.")
    chat = gr.Chatbot(label="Your design conversation", type="messages", height=480)
    count = gr.State(0)
    with gr.Row():
        message = gr.Textbox(placeholder="Describe your idea or answer the latest question…", label="Your message", scale=5)
        send = gr.Button("Send", variant="primary", scale=1)
    status = gr.Markdown("Describe your idea to begin.")
    download = gr.DownloadButton("Download design brief (.md)", visible=False)
    clear = gr.Button("Start over")
    send.click(respond, inputs=[message, chat, count], outputs=[chat, count, download, status]).then(lambda: "", outputs=message)
    message.submit(respond, inputs=[message, chat, count], outputs=[chat, count, download, status]).then(lambda: "", outputs=message)
    clear.click(reset, outputs=[chat, count, download, status])

if __name__ == "__main__":
    demo.launch()
