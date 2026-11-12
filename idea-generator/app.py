"""Minimal Gradio application for browsing generated ideas."""

import csv
import random
from pathlib import Path

import gradio as gr


CSV_FILE = Path(__file__).with_name("ideas.csv")


def load_ideas() -> tuple[list[dict[str, str]], str]:
    if not CSV_FILE.exists():
        return [], f"{CSV_FILE.name} was not found. Run generate_ideas.py first."

    try:
        with CSV_FILE.open(newline="", encoding="utf-8") as file:
            rows = list(csv.DictReader(file))
    except (OSError, csv.Error) as error:
        return [], f"Could not read {CSV_FILE.name}: {error}"

    ideas = [{"title": row["title"], "description": row["description"]} for row in rows]
    return ideas, ""


IDEAS, LOAD_ERROR = load_ideas()


def show_idea(used_indexes: list[int]) -> tuple[str, str, list[int], str]:
    if LOAD_ERROR:
        return "", "", used_indexes, LOAD_ERROR

    if len(used_indexes) == len(IDEAS):
        used_indexes = []
        reset_message = "All ideas have been shown. Starting a new round."
    else:
        reset_message = ""

    available = [index for index in range(len(IDEAS)) if index not in used_indexes]
    selected_index = random.choice(available)
    used_indexes = [*used_indexes, selected_index]
    selected = IDEAS[selected_index]
    return selected["title"], selected["description"], used_indexes, reset_message


with gr.Blocks(title="Idea Generator") as demo:
    gr.Markdown("# Give Me an Idea")
    gr.Markdown("Need inspiration? Get a random idea to explore.")
    button = gr.Button("Give Me an Idea", variant="primary")
    title = gr.Markdown()
    description = gr.Markdown()
    status = gr.Markdown(LOAD_ERROR)
    history = gr.State([])

    button.click(
        show_idea,
        inputs=history,
        outputs=[title, description, history, status],
    )


if __name__ == "__main__":
    demo.launch()
