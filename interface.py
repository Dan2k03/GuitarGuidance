
import time
import random
import gradio as gr

# --------------------------------------------------
# Placeholder session logic
# Replace with actual audio capture and ML later.
# --------------------------------------------------

CHORDS = ["Am", "C", "D", "Em", "F", "G", "A", "Dm", "E"]

CHORD_NAMES = {
    "Am": "A minor",
    "C": "C major",
    "D": "D major",
    "Em": "E minor",
    "F": "F major",
    "G": "G major",
    "A": "A major",
    "Dm": "D minor",
    "E": "E major",
}


def start_session():
    """Initialize a new practice session."""
    return (
        True,
        time.time(),
        "SESSION ACTIVE",
        "00:00",
        "Listening for audio...",
        "—",
        "Waiting for audio",
        0,
        [],
        [],
    )


def end_session(activity, timeline):
    """End the current session and preserve its history."""
    return (
        False,
        None,
        "SESSION ENDED",
        "00:00",
        "Session complete.",
        "—",
        "No active chord",
        0,
        activity,
        timeline,
    )


def update_session(active, start_time, intended, activity, timeline):
    """
    Simulate live chord detection.

    Replace the random prediction with your audio-processing
    pipeline and trained classifier when ready.
    """
    if not active or start_time is None:
        return (
            "00:00",
            "Start a session to begin.",
            "—",
            "Waiting for audio",
            0,
            activity,
            timeline,
        )

    elapsed = int(time.time() - start_time)
    timer = f"{elapsed // 60:02d}:{elapsed % 60:02d}"

    # Placeholder prediction
    detected = random.choice(CHORDS)
    confidence = random.randint(75, 99)
    chord_name = CHORD_NAMES[detected]

    # Add an event every 3 seconds, avoiding duplicate
    # entries for the same elapsed second.
    last_logged_second = (
        activity[0]["Elapsed"]
        if activity
        else -1
    )

    if elapsed > 0 and elapsed % 3 == 0 and last_logged_second != elapsed:
        timestamp = timer
        result = "Match" if detected == intended else "Review"

        new_event = {
            "Elapsed": elapsed,
            "Time": timestamp,
            "Detected chord": detected,
            "Intended chord": intended,
            "Result": result,
        }

        activity = [new_event] + activity[:19]
        timeline = (timeline + [detected])[-12:]

    status = (
        f"Matched intended chord: {intended}"
        if detected == intended
        else f"Detected {detected} — intended chord is {intended}"
    )

    return (
        timer,
        status,
        detected,
        chord_name,
        confidence,
        activity,
        timeline,
    )


def format_activity(events):
    """Convert session events into a table-friendly format."""
    return [
        [
            event["Time"],
            event["Detected chord"],
            event["Intended chord"],
            event["Result"],
        ]
        for event in events
    ]


def format_timeline(chords):
    """Format the progression history for display."""
    if not chords:
        return "Your detected chord progression will appear here."

    return "  →  ".join(chords)


# --------------------------------------------------
# Styling
# --------------------------------------------------

CSS = """
.gradio-container {
    max-width: 1180px !important;
    margin: auto !important;
    background: #0c0d12 !important;
    color: #eeeeF5 !important;
}

footer {
    display: none !important;
}

#header {
    border-bottom: 1px solid #292a36;
    padding: 20px 4px;
    margin-bottom: 28px;
}

#brand {
    font-size: 15px;
    font-weight: 800;
    letter-spacing: 2px;
    color: #f4f2ff;
}

#page-title {
    font-size: 30px;
    font-weight: 750;
    letter-spacing: -0.8px;
    margin-bottom: 5px;
}

#page-subtitle {
    color: #9293a5;
    font-size: 14px;
}

.panel {
    border: 1px solid #292a36 !important;
    border-radius: 14px !important;
    background: #14151d !important;
    padding: 20px !important;
}

.panel-title {
    font-size: 12px !important;
    font-weight: 700 !important;
    letter-spacing: 1.3px !important;
    color: #a6a7bb !important;
}

#session-status input {
    color: #b9a5ff !important;
    font-weight: 700 !important;
    letter-spacing: 1px;
    text-align: right;
    background: transparent !important;
    border: none !important;
}

#detected-chord input {
    font-size: 76px !important;
    font-weight: 800 !important;
    text-align: center !important;
    color: #c4b5fd !important;
    height: 110px !important;
    background: transparent !important;
    border: none !important;
}

#chord-name input {
    text-align: center !important;
    color: #c1c1d1 !important;
    background: transparent !important;
    border: none !important;
    font-size: 16px !important;
}

#session-timer input {
    font-size: 32px !important;
    font-weight: 700 !important;
    font-variant-numeric: tabular-nums;
    text-align: center !important;
    color: #f4f2ff !important;
}

#start-button {
    background: #7856e8 !important;
    border: none !important;
    color: white !important;
    font-weight: 700 !important;
    border-radius: 9px !important;
}

#start-button:hover {
    background: #6845d4 !important;
}

#end-button {
    background: #292a36 !important;
    border: 1px solid #3a3b4a !important;
    color: #eeeeF5 !important;
    font-weight: 600 !important;
    border-radius: 9px !important;
}

#live-status input {
    color: #c4b5fd !important;
    background: #1b1929 !important;
    border-color: #39304f !important;
}

#footer-note {
    text-align: center;
    color: #696a7c;
    font-size: 11px;
    letter-spacing: 1.4px;
    padding: 26px 0 10px 0;
}
"""


# --------------------------------------------------
# Gradio interface
# --------------------------------------------------

with gr.Blocks(
    # Use Gradio's default font for compatibility.
    theme=gr.themes.Base(
        primary_hue="violet",
        secondary_hue="slate",
        neutral_hue="slate",
    ),
    css=CSS,
    title="Guitar Guidance",
) as demo:

    # Session state
    session_active = gr.State(False)
    session_start = gr.State(None)
    activity_state = gr.State([])
    timeline_state = gr.State([])

    gr.HTML("""
    <div id="header">
        <div id="brand">GUITAR GUIDANCE</div>
    </div>
    """)

    with gr.Row():
        with gr.Column(scale=7):
            gr.Markdown(
                "# Practice session",
                elem_id="page-title",
            )
            gr.Markdown(
                "Play naturally. Get feedback as you go.",
                elem_id="page-subtitle",
            )

        with gr.Column(scale=3):
            session_status = gr.Textbox(
                value="SESSION READY",
                label="",
                interactive=False,
                elem_id="session-status",
            )

    # Main session area
    with gr.Row(equal_height=True):

        # Live chord display
        with gr.Column(scale=7):
            with gr.Group(elem_classes=["panel"]):
                gr.Markdown(
                    "LIVE CHORD",
                    elem_classes=["panel-title"],
                )

                detected_chord = gr.Textbox(
                    value="—",
                    label="",
                    interactive=False,
                    elem_id="detected-chord",
                )

                chord_name = gr.Textbox(
                    value="Waiting for audio",
                    label="",
                    interactive=False,
                    elem_id="chord-name",
                )

                confidence = gr.Slider(
                    minimum=0,
                    maximum=100,
                    value=0,
                    label="Detection confidence (%)",
                    interactive=False,
                )

                live_status = gr.Textbox(
                    value="Start a session to begin.",
                    label="Status",
                    interactive=False,
                    elem_id="live-status",
                )

        # Session controls
        with gr.Column(scale=5):
            with gr.Group(elem_classes=["panel"]):
                gr.Markdown(
                    "SESSION CONTROLS",
                    elem_classes=["panel-title"],
                )

                session_timer = gr.Textbox(
                    value="00:00",
                    label="Session duration",
                    interactive=False,
                    elem_id="session-timer",
                )

                intended_chord = gr.Dropdown(
                    choices=CHORDS,
                    value="Am",
                    label="Chord to practice",
                    info="Select the chord you intend to play.",
                )

                start_button = gr.Button(
                    "Start session",
                    variant="primary",
                    elem_id="start-button",
                )

                end_button = gr.Button(
                    "End session",
                    elem_id="end-button",
                )

                gr.Markdown(
                    "Audio capture and chord detection are simulated "
                    "in this prototype."
                )

    # Session timeline
    with gr.Group(elem_classes=["panel"]):
        gr.Markdown(
            "SESSION TIMELINE",
            elem_classes=["panel-title"],
        )

        timeline_display = gr.Textbox(
            value="Your detected chord progression will appear here.",
            label="Detected progression",
            interactive=False,
            lines=2,
        )

    # Recent activity
    with gr.Group(elem_classes=["panel"]):
        gr.Markdown(
            "RECENT ACTIVITY",
            elem_classes=["panel-title"],
        )

        activity_table = gr.Dataframe(
            headers=[
                "Time",
                "Detected chord",
                "Intended chord",
                "Result",
            ],
            value=[],
            interactive=False,
            label="Session events",
        )

    gr.HTML("""
    <div id="footer-note">
        GUITAR GUIDANCE · PRACTICE SESSION PROTOTYPE
    </div>
    """)

    # Start session
    start_button.click(
        fn=start_session,
        inputs=[],
        outputs=[
            session_active,
            session_start,
            session_status,
            session_timer,
            live_status,
            detected_chord,
            chord_name,
            confidence,
            activity_state,
            timeline_state,
        ],
    ).then(
        fn=lambda: "Your detected chord progression will appear here.",
        outputs=[timeline_display],
    ).then(
        fn=lambda: [],
        outputs=[activity_table],
    )

    # End session
    end_button.click(
        fn=end_session,
        inputs=[
            activity_state,
            timeline_state,
        ],
        outputs=[
            session_active,
            session_start,
            session_status,
            session_timer,
            live_status,
            detected_chord,
            chord_name,
            confidence,
            activity_state,
            timeline_state,
        ],
    ).then(
        fn=format_activity,
        inputs=[activity_state],
        outputs=[activity_table],
    ).then(
        fn=format_timeline,
        inputs=[timeline_state],
        outputs=[timeline_display],
    )

    # Refresh the interface once per second.
    refresh_timer = gr.Timer(1.0)

    refresh_timer.tick(
        fn=update_session,
        inputs=[
            session_active,
            session_start,
            intended_chord,
            activity_state,
            timeline_state,
        ],
        outputs=[
            session_timer,
            live_status,
            detected_chord,
            chord_name,
            confidence,
            activity_state,
            timeline_state,
        ],
    ).then(
        fn=format_activity,
        inputs=[activity_state],
        outputs=[activity_table],
    ).then(
        fn=format_timeline,
        inputs=[timeline_state],
        outputs=[timeline_display],
    )


if __name__ == "__main__":
    demo.launch()
