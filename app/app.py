"""Gradio demo: paste a recruitment email, get the validated JSON record.

Runs the fine-tuned adapter (4-bit base + LoRA) on a GPU. On Colab use notebooks/03_demo.ipynb,
which passes --share for a temporary public link.

    python app/app.py --adapter <adapter dir or hub id> [--merge] [--share]
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from scripts.generate import parse_date_header  # noqa: E402
from src.eval import parse_output  # noqa: E402
from src.infer import DEFAULT_MODEL, generate, load  # noqa: E402
from src.prompting import FINETUNE_SYSTEM_PROMPT, chat_messages  # noqa: E402
from src.validation import FIELDS, validate  # noqa: E402

# Fictional demo emails (not from any split).
EXAMPLES = [
    """From: Camille Vasseur <camille.vasseur@norvelia.fr>
Date: Tue, 03 Nov 2026 10:12:40 +0100
Subject: Stage Data Engineer - test technique

Bonjour,

Merci pour votre candidature au stage de fin d'études Data Engineer chez Norvelia.
Nous vous proposons un test technique en ligne (environ 2h) : https://jobs.norvelia.fr/test/de-2207
Merci de le compléter avant vendredi. Le poste est en hybride (2 jours de télétravail par semaine).

Bien cordialement,
Camille Vasseur
Talent Acquisition Specialist, Norvelia""",
    """From: Noreply <no-reply@careers.brightloom.io>
Date: Mon, 09 Nov 2026 08:01:12 +0000
Subject: Your application: Backend Engineer (CDI)

Hi,

Thank you for applying for the Backend Engineer position at Brightloom. Our team has received
your application and will review it carefully. If your profile matches, we will contact you.

This is an automated message, please do not reply.""",
    """From: Hugo Marchetti <hugo@tessaline.com>
Date: Thu, 12 Nov 2026 17:45:03 +0100
Subject: Re: alternance ML

hello! merci pour le call de mardi, c'était top. On aimerait te faire passer le final round
avec notre CTO, 45 min en visio. Tu peux me dire tes dispos pour la semaine pro ?
Le poste reste full remote btw.

Hugo
Head of ML, Tessaline""",
]


def build(tok, model):
    import gradio as gr

    def extract(email: str):
        if not email.strip():
            return "", "Paste an email first.", ""
        text, latency = generate(tok, model, chat_messages(email, system=FINETUNE_SYSTEM_PROMPT))
        record = parse_output(text)
        if record is None:
            return text, "❌ Output is not valid JSON.", f"{latency:.2f} s"
        _, sent = parse_date_header(email)
        errors = validate(record, sent.date() if sent else None)
        status = "✅ Valid: matches the schema and the cross-field rules." if not errors else \
            "⚠️ " + "; ".join(errors)
        if not sent:
            status += "  (No parseable Date: header, so relative deadlines cannot be resolved.)"
        shown = {k: record[k] for k in FIELDS if k in record} if not errors else record
        return json.dumps(shown, ensure_ascii=False, indent=2), status, f"{latency:.2f} s"

    with gr.Blocks(title="Recruitment email → JSON") as demo:
        gr.Markdown("## Recruitment email → JSON\nQwen3-4B-Instruct-2507 fine-tuned with QLoRA "
                    "(schema v1). Include the `From:` / `Date:` / `Subject:` headers: the Date header is "
                    "used to resolve relative deadlines such as *avant vendredi*.")
        with gr.Row():
            with gr.Column():
                email = gr.Textbox(label="Email", lines=18, placeholder="From: ...\nDate: ...\nSubject: ...\n\n...")
                run = gr.Button("Extract", variant="primary")
            with gr.Column():
                out = gr.Code(label="Record", language="json")
                status = gr.Markdown()
                latency = gr.Textbox(label="Generation time", interactive=False)
        gr.Examples(EXAMPLES, inputs=email)
        run.click(extract, inputs=email, outputs=[out, status, latency])
    return demo


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default=DEFAULT_MODEL)
    ap.add_argument("--adapter", required=True)
    ap.add_argument("--merge", action="store_true", help="merge the adapter for faster generation")
    ap.add_argument("--share", action="store_true", help="temporary public gradio.live link")
    args = ap.parse_args()
    tok, model = load(args.model, args.adapter, merge=args.merge)
    build(tok, model).queue().launch(share=args.share)


if __name__ == "__main__":
    main()
