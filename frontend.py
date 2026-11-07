"""
frontend.py - Desktop GUI for Gabay, the 24/7 health guide chatbot.

A Tkinter-based desktop application styled like a social media messaging window.
Run with:  python frontend.py

It imports GabayBackend from backend.py and handles:
  * A header with profile picture (colored circle with 'G'), name, handle, online status
  * Chat bubbles (user on the right, bot on the left) with timestamps
  * Scrollable chat area, message input box, Send button, Enter-to-send
  * 'Gabay is typing...' indicator while the model generates
  * Quick-reply buttons for common questions
  * Clear chat button
  * Model calls run in a background thread so the window never freezes
  * Word-by-word streaming of the bot response
"""

import threading
from datetime import datetime

try:
    import tkinter as tk
    from tkinter import ttk
except ModuleNotFoundError as exc:
    if exc.name not in {"tkinter", "_tkinter"}:
        raise
    raise SystemExit(
        "Tkinter is not installed for this Python interpreter. Install your "
        "distribution's Tkinter package, then run frontend.py again."
    ) from exc

try:
    from backend import GabayBackend, CHATBOT_NAME, TAGLINE, HANDLE
except ImportError as exc:  # pragma: no cover
    raise SystemExit(f"Could not import backend.py: {exc}")


# ---------------------------------------------------------------------------
# Theme / color palette (healthcare teal + white)
# ---------------------------------------------------------------------------

BG_COLOR = "#edf4f2"
HEADER_COLOR = "#145b5b"
HEADER_TEXT = "#f7fffd"
BUBBLE_USER = "#d8f2e9"
BUBBLE_BOT = "#ffffff"
BUBBLE_BOT_BORDER = "#d8e6e2"
TEXT_DARK = "#183536"
TEXT_MUTED = "#6c8583"
ACCENT_TEAL = "#087f78"
ACCENT_HOVER = "#066b66"
FONT_FAMILY = "TkDefaultFont"


# ---------------------------------------------------------------------------
# Widgets
# ---------------------------------------------------------------------------

class ProfileCircle(tk.Canvas):
    """A small colored circle with the letter 'G' acting as a profile picture."""

    def __init__(self, parent, size=40, **kwargs):
        super().__init__(parent, width=size, height=size,
                         highlightthickness=0, bg=HEADER_COLOR, **kwargs)
        self.size = size
        self._draw()

    def _draw(self):
        s = self.size
        self.create_oval(2, 2, s - 2, s - 2, fill="#d8f2e9", outline="")
        self.create_oval(11, 8, 16, 13, fill=HEADER_COLOR, outline="")
        self.create_oval(30, 8, 35, 13, fill=HEADER_COLOR, outline="")
        self.create_line(
            13.5, 11, 13.5, 19, 17, 24, 22, 27, 24, 27,
            29, 24, 32.5, 19, 32.5, 11,
            smooth=True, splinesteps=16, fill=HEADER_COLOR, width=2.5,
        )
        self.create_line(24, 27, 24, 32, 29, 34, fill=HEADER_COLOR, width=2.5)
        self.create_oval(28, 32, 35, 39, fill=HEADER_COLOR, outline="")
        self.create_oval(31, 31, 44, 44, fill="#ffffff", outline="#d8e6e2")
        self.create_text(37.5, 37.5, text="+", fill=ACCENT_TEAL,
                         font=(FONT_FAMILY, 10, "bold"))


class ChatBubble(tk.Frame):
    """A single chat bubble with message text and timestamp."""

    def __init__(self, parent, message, timestamp, is_user, **kwargs):
        super().__init__(parent, bg=BG_COLOR, **kwargs)
        self.is_user = is_user

        bubble_bg = BUBBLE_USER if is_user else BUBBLE_BOT
        text_color = TEXT_DARK
        anchor = "e" if is_user else "w"
        padx = (48, 12) if is_user else (12, 48)

        # The bubble (label with wrapped text + border effect via frame bg)
        self.bubble = tk.Frame(self, bg=bubble_bg, bd=0, highlightthickness=0)
        self.bubble.pack(anchor=anchor, padx=padx, pady=(4, 0))

        self.msg_label = tk.Label(
            self.bubble, text=message, bg=bubble_bg, fg=text_color,
            font=(FONT_FAMILY, 10), justify=tk.LEFT, wraplength=280,
            padx=13, pady=10, anchor="w",
        )
        self.msg_label.pack()

        # Bot bubbles get a thin border effect by wrapping bg frame
        if not is_user:
            self.bubble.config(highlightbackground=BUBBLE_BOT_BORDER,
                               highlightthickness=1)

        # Timestamp below the bubble
        self.time_label = tk.Label(
            self, text=timestamp, bg=BG_COLOR, fg=TEXT_MUTED,
            font=(FONT_FAMILY, 8),
        )
        self.time_label.pack(anchor=anchor, padx=padx, pady=(2, 5))


class TypingIndicator(tk.Frame):
    """'Gabay is typing...' indicator shown while the model generates."""

    def __init__(self, parent, **kwargs):
        super().__init__(parent, bg=BG_COLOR, **kwargs)
        self.label = tk.Label(
            self, text="Gabay is typing", bg=BG_COLOR, fg=ACCENT_TEAL,
            font=(FONT_FAMILY, 9),
        )
        self.label.pack(anchor="w", padx=18, pady=(2, 4))
        self.dots = 0
        self._job = None
        self._running = False

    def start(self):
        self._running = True
        self._animate()
        self.pack(anchor="w", fill=tk.X)

    def stop(self):
        self._running = False
        if self._job is not None:
            self.after_cancel(self._job)
            self._job = None
        self.pack_forget()

    def _animate(self):
        if not self._running:
            return
        self.dots = (self.dots % 3) + 1
        self.label.config(text="Gabay is typing" + "." * self.dots)
        self._job = self.after(500, self._animate)


# ---------------------------------------------------------------------------
# Main application
# ---------------------------------------------------------------------------

class GabayApp:
    def __init__(self, root):
        self.root = root
        self.root.title(f"{CHATBOT_NAME} - {HANDLE}")
        self.root.geometry("460x760")
        self.root.minsize(410, 640)
        self.root.configure(bg=BG_COLOR)

        self.backend = GabayBackend()
        self._stream_job = None  # holds after() id for word-by-word streaming
        self._stream_words = []  # remaining words to stream
        self._stream_index = 0

        self._build_ui()
        self._add_welcome_message()

    # -- UI construction --------------------------------------------------

    def _build_ui(self):
        header = tk.Frame(self.root, bg=HEADER_COLOR, height=112)
        header.pack(fill=tk.X)
        header.pack_propagate(False)

        prof = ProfileCircle(header, size=46)
        prof.pack(side=tk.LEFT, padx=(18, 12), pady=30)

        info = tk.Frame(header, bg=HEADER_COLOR)
        info.pack(side=tk.LEFT, fill=tk.Y, expand=True, padx=(0, 10))

        tk.Label(
            info, text=CHATBOT_NAME, bg=HEADER_COLOR, fg=HEADER_TEXT,
            font=(FONT_FAMILY, 17, "bold"),
        ).pack(anchor="w", pady=(10, 0))
        tk.Label(
            info, text=HANDLE, bg=HEADER_COLOR, fg="#c5e4df",
            font=(FONT_FAMILY, 10),
        ).pack(anchor="w", pady=(0, 2))
        tk.Label(
            info, text="\u25cf Online  \u00b7  24/7 health guide",
            bg=HEADER_COLOR, fg="#c5e4df", font=(FONT_FAMILY, 9),
        ).pack(anchor="w", pady=(0, 2))
        tk.Label(
            info, text=TAGLINE, bg=HEADER_COLOR, fg="#e2f1ed",
            font=(FONT_FAMILY, 9),
        ).pack(anchor="w")

        quick = tk.Frame(self.root, bg=BG_COLOR)
        quick.pack(fill=tk.X, padx=12, pady=(10, 2))
        self.quick_buttons = []
        for column, label in enumerate(["Services", "Schedule", "Location", "Emergency"]):
            quick.grid_columnconfigure(column, weight=1, uniform="quick-replies")
            button = tk.Button(
                quick, text=label, bg=ACCENT_TEAL, fg=HEADER_TEXT,
                font=(FONT_FAMILY, 9, "bold"), bd=0, padx=4, pady=8,
                cursor="hand2", activebackground=ACCENT_HOVER,
                activeforeground=HEADER_TEXT,
                command=lambda value=label: self._send_quick(value),
            )
            button.grid(row=0, column=column, sticky="ew", padx=(0, 6 if column < 3 else 0))
            self.quick_buttons.append(button)

        chat_container = tk.Frame(self.root, bg=BG_COLOR)
        chat_container.pack(fill=tk.BOTH, expand=True, padx=10, pady=(5, 2))

        self.canvas = tk.Canvas(chat_container, bg=BG_COLOR, highlightthickness=0)
        self.scrollbar = ttk.Scrollbar(
            chat_container, orient="vertical", command=self.canvas.yview
        )
        self.chat_frame = tk.Frame(self.canvas, bg=BG_COLOR)
        self.chat_frame.bind(
            "<Configure>",
            lambda event: self.canvas.configure(scrollregion=self.canvas.bbox("all")),
        )
        self.chat_window = self.canvas.create_window(
            (0, 0), window=self.chat_frame, anchor="nw"
        )
        self.canvas.configure(yscrollcommand=self.scrollbar.set)
        self.canvas.bind("<Configure>", self._resize_chat_window)
        self.canvas.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        self.scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
        self.canvas.bind_all("<MouseWheel>", self._on_mousewheel)
        self.canvas.bind_all("<Button-4>", lambda event: self.canvas.yview_scroll(-1, "units"))
        self.canvas.bind_all("<Button-5>", lambda event: self.canvas.yview_scroll(1, "units"))

        self.typing = TypingIndicator(self.root)

        input_frame = tk.Frame(self.root, bg=BG_COLOR)
        input_frame.pack(fill=tk.X, padx=12, pady=(4, 12))
        composer = tk.Frame(
            input_frame, bg=BUBBLE_BOT, highlightbackground=BUBBLE_BOT_BORDER,
            highlightthickness=1,
        )
        composer.pack(fill=tk.X)

        self.entry = tk.Entry(
            composer, font=(FONT_FAMILY, 11), bd=0, relief=tk.FLAT,
            bg=BUBBLE_BOT, fg=TEXT_DARK, insertbackground=ACCENT_TEAL,
        )
        self.entry.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=(12, 6), ipady=11)
        self.entry.bind("<Return>", lambda event: self._send_message())

        self.send_button = tk.Button(
            composer, text="Send", bg=ACCENT_TEAL, fg=HEADER_TEXT,
            font=(FONT_FAMILY, 10, "bold"), bd=0, padx=17, pady=9,
            cursor="hand2", activebackground=ACCENT_HOVER,
            activeforeground=HEADER_TEXT, command=self._send_message,
        )
        self.send_button.pack(side=tk.RIGHT, padx=5, pady=5)

        self.clear_button = tk.Button(
            input_frame, text="Clear chat", bg=BG_COLOR, fg=TEXT_MUTED,
            font=(FONT_FAMILY, 9), bd=0, padx=8, pady=5, cursor="hand2",
            activebackground=BG_COLOR, activeforeground=TEXT_DARK,
            command=self._clear_chat,
        )
        self.clear_button.pack(anchor="e", pady=(4, 0))
        self.entry.focus_set()

    # -- chat helpers -----------------------------------------------------

    def _add_welcome_message(self):
        welcome = (
            "Hello! I'm Gabay, your 24/7 health guide from Gabay General Hospital "
            "({handle}).\n\nI can help you with our services, clinic schedule, "
            "location, appointments, visiting hours, and general health information.\n\n"
            "\u26a0\ufe0f If you're having a medical emergency, please call 911 or go "
            "to the nearest hospital immediately."
        ).format(handle=HANDLE)
        self._add_bot_message(welcome, animate=False)

    def _add_bubble(self, message, timestamp, is_user, animate=False):
        bubble = ChatBubble(self.chat_frame, message, timestamp, is_user)
        bubble.pack(anchor="e" if is_user else "w", fill=tk.X, pady=2)
        self._scroll_to_bottom()
        return bubble

    def _add_user_message(self, message):
        ts = datetime.now().strftime("%I:%M %p")
        self._add_bubble(message, ts, is_user=True)

    def _add_bot_message(self, message, animate=True):
        ts = datetime.now().strftime("%I:%M %p")
        if animate:
            self._stream_bot_message(message, ts)
        else:
            self._add_bubble(message, ts, is_user=False)

    def _scroll_to_bottom(self):
        self.canvas.after_idle(lambda: self.canvas.yview_moveto(1.0))

    def _resize_chat_window(self, event):
        self.canvas.itemconfigure(self.chat_window, width=event.width)

    def _on_mousewheel(self, event):
        # Windows/Mac wheel
        self.canvas.yview_scroll(int(-1 * (event.delta / 120)), "units")
        return "break"

    # -- actions ----------------------------------------------------------

    def _send_quick(self, label):
        self.entry.delete(0, tk.END)
        self.entry.insert(0, label)
        self._send_message()

    def _send_message(self):
        text = self.entry.get().strip()
        if not text:
            return
        self.entry.delete(0, tk.END)

        self._add_user_message(text)

        # Show typing indicator and start backend work in a thread.
        self.typing.start()
        self.entry.config(state=tk.DISABLED)
        self.send_button.config(state=tk.DISABLED)
        self.clear_button.config(state=tk.DISABLED)
        for button in self.quick_buttons:
            button.config(state=tk.DISABLED)

        thread = threading.Thread(target=self._worker, args=(text,), daemon=True)
        thread.start()

    def _worker(self, text):
        """Runs in a background thread: streams the response from the backend."""
        for piece in self.backend.stream_response(text):
            # Push each piece to the main thread for streaming display.
            self.root.after(0, lambda p=piece: self._append_stream_chunk(p))
        # After stream ends, ensure final metadata/log is ready.
        self.root.after(0, self._finish_stream)

    def _append_stream_chunk(self, chunk):
        """Append a streamed chunk to the current bot bubble (called on main thread)."""
        # If there is no active streaming bubble, create one.
        if not hasattr(self, "_stream_bubble") or self._stream_bubble is None:
            ts = datetime.now().strftime("%I:%M %p")
            self._stream_bubble = self._create_stream_bubble(ts)

        self._stream_bubble.append_text(chunk)
        self._scroll_to_bottom()

    def _create_stream_bubble(self, ts):
        """Create an empty bot bubble that we append streamed text into."""
        bubble = _StreamableBubble(self.chat_frame, "", ts)
        bubble.pack(anchor="w", fill=tk.X, pady=2)
        self._scroll_to_bottom()
        return bubble

    def _finish_stream(self):
        """Called on the main thread after streaming completes."""
        self.typing.stop()
        self.entry.config(state=tk.NORMAL)
        self.send_button.config(state=tk.NORMAL)
        self.clear_button.config(state=tk.NORMAL)
        for button in self.quick_buttons:
            button.config(state=tk.NORMAL)
        self.entry.focus_set()
        self._stream_bubble = None

    def _clear_chat(self):
        # Remove all widgets in chat_frame
        for widget in self.chat_frame.winfo_children():
            widget.destroy()
        # Reset backend memory so follow-ups don't reference cleared chat
        self.backend.memory = []
        self._add_welcome_message()

    # -- lifecycle --------------------------------------------------------

    def run(self):
        self.root.mainloop()


class _StreamableBubble(tk.Frame):
    """A bot bubble whose text can be appended to during streaming."""

    def __init__(self, parent, message, timestamp, **kwargs):
        super().__init__(parent, bg=BG_COLOR, **kwargs)
        self.timestamp = timestamp

        self.bubble = tk.Frame(self, bg=BUBBLE_BOT, bd=0,
                               highlightbackground=BUBBLE_BOT_BORDER,
                               highlightthickness=1)
        self.bubble.pack(anchor="w", padx=(10, 60), pady=(4, 0))

        self.msg_label = tk.Label(
            self.bubble, text=message, bg=BUBBLE_BOT, fg=TEXT_DARK,
            font=(FONT_FAMILY, 10), justify=tk.LEFT, wraplength=280,
            padx=12, pady=8, anchor="w",
        )
        self.msg_label.pack()

        self.time_label = tk.Label(
            self, text=timestamp, bg=BG_COLOR, fg=TEXT_MUTED,
            font=(FONT_FAMILY, 8),
        )
        self.time_label.pack(anchor="w", padx=(10, 60), pady=(0, 6))

    def append_text(self, chunk):
        current = self.msg_label.cget("text")
        self.msg_label.config(text=current + chunk)


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    # Make imports work when run directly from the Gabay folder.
    import sys
    import os
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

    from backend import CHATBOT_NAME, HANDLE  # noqa: E402  (for title only)

    root = tk.Tk()
    app = GabayApp(root)
    app.run()
