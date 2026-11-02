import gradio as gr
import random

# --------------------------------------------------
# Placeholder logic — replace with your ML pipeline
# --------------------------------------------------

CHORDS = ["Am", "C", "D", "Em", "F", "G", "E", "A", "Dm"]

def analyze_audio(audio, intended_chord):
    if audio is None:
        return (
            "—",
            "Upload or record audio first.",
            "Waiting for audio",
            "No recording analyzed yet."
        )

    # Temporary mock prediction
    predicted_chord = random.choice(CHORDS)
    confidence = random.randint(78, 99)

    if predicted_chord == intended_chord:
        feedback = f"Nice work! We detected {predicted_chord}."
        status = "Chord matched"
    else:
        feedback = (
            f"We detected {predicted_chord}, but you selected "
            f"{intended_chord}. Try again and let the notes ring clearly."
        )
        status = "Keep practicing"

    result = f"{predicted_chord} · {confidence}% confidence"

    return predicted_chord, result, status, feedback


# --------------------------------------------------
# Styling
# --------------------------------------------------

CSS = """
.gradio-container {
    max-width: 1050px !important;
    margin: auto !important;
    background: #0b0d14 !important;
    color: #f1f2f9 !important;
}

footer {
    display: none !important;
}

#app-header {
    text-align: center;
    padding: 28px 0 22px 0;
}

#app-title {
    font-size: 38px;
    font-weight: 800;
    letter-spacing: -1.5px;
    color: #ffffff;
    margin-bottom: 6px;
}

#app-subtitle {
    font-size: 15px;
    color: #9ca3b8;
}

.panel {
    border: 1px solid #262a3a !important;
    border-radius: 16px !important;
    background: #131622 !important;
    padding: 18px !important;
}

.section-title {
    color: #f1f2f9;
    font-size: 17px;
    font-weight: 700;
}

#analyze-button {
    background: #8b5cf6 !important;
    color: white !important;
    border: none !important;
    border-radius: 10px !important;
    font-weight: 700 !important;
}

#analyze-button:hover {
    background: #7c3aed !important;
}

#predicted-chord textarea {
    font-size: 32px !important;
    font-weight: 800 !important;
    text-align: center !important;
    color: #c4b5fd !important;
}

#confidence-box textarea {
    color: #a5b4fc !important;
    font-weight: 600 !important;
}

#feedback-box textarea {
    color: #d1d5e7 !important;
}

#footer-note {
    text-align: center;
    color: #737b91;
    font-size: 12px;
    padding: 18px 0;
}
"""


# --------------------------------------------------
# Interface
# --------------------------------------------------

with gr.Blocks(
    theme=gr.themes.Base(
        primary_hue="violet",
        secondary_hue="slate",
        neutral_hue="slate",
        font=[gr.themes.GoogleFont("Inter"), "sans-serif"],
    ),
    css=CSS,
    title="Guitar Guidance",
) as demo:

    gr.HTML("""
        <div id="app-header">
            <div id="app-title">🎸 Guitar Guidance</div>
            <div id="app-subtitle">
                Your personal AI guitar coach · Practice smarter, play better.
            </div>
        </div>
    """)

    with gr.Row(equal_height=False):

        # LEFT: Input and practice settings
        with gr.Column(scale=5):
            with gr.Group(elem_classes=["panel"]):
                gr.Markdown("### 🎙️ Audio input")

                audio_input = gr.Audio(
                    sources=["upload", "microphone"],
                    type="filepath",
                    label="Your guitar recording",
                )

                gr.Markdown(
                    "Upload a recording or use your microphone to get started."
                )

            with gr.Group(elem_classes=["panel"]):
                gr.Markdown("### 🎯 Practice mode")

                intended_chord = gr.Dropdown(
                    choices=CHORDS,
                    value="Am",
                    label="Chord you're practicing",
                    info="Choose the chord you intended to play.",
                )

                analyze_button = gr.Button(
                    "Analyze my playing",
                    variant="primary",
                    elem_id="analyze-button",
                    size="lg",
                )

        # RIGHT: Results
        with gr.Column(scale=6):
            with gr.Group(elem_classes=["panel"]):
                gr.Markdown("### 📊 Analysis results")

                predicted_chord = gr.Textbox(
                    value="—",
                    label="Detected chord",
                    interactive=False,
                    elem_id="predicted-chord",
                )

                confidence = gr.Textbox(
                    value="Waiting for audio",
                    label="Model confidence",
                    interactive=False,
                    elem_id="confidence-box",
                )

                status = gr.Textbox(
                    value="Ready when you are",
                    label="Practice status",
                    interactive=False,
                )

            with gr.Group(elem_classes=["panel"]):
                gr.Markdown("### 💬 Coach feedback")

                feedback = gr.Textbox(
                    value=(
                        "Your feedback will appear here after "
                        "you analyze a recording."
                    ),
                    label="Feedback",
                    lines=4,
                    interactive=False,
                    elem_id="feedback-box",
                )

                gr.Markdown(
                    "*Detailed feedback on missed notes and chord quality "
                    "will be available in a future version.*"
                )

    gr.HTML("""
        <div id="footer-note">
            GUITAR GUIDANCE · AI-POWERED PRACTICE · PROTOTYPE
        </div>
    """)

    analyze_button.click(
        fn=analyze_audio,
        inputs=[audio_input, intended_chord],
        outputs=[
            predicted_chord,
            confidence,
            status,
            feedback,
        ],
    )


if __name__ == "__main__":
    demo.launch()
```