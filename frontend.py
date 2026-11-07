"""
frontend.py - Desktop GUI for Yakap, the healthcare information assistant.

A Tkinter-based desktop application styled like a social media messaging window.
Run with:  python frontend.py

It imports YakapBackend from backend.py and handles:
    * A themed healthcare profile mark, account name, handle, and online status
    * Chat bubbles (user on the right, Yakap on the left) with timestamps
  * Scrollable chat area, message input box, Send button, Enter-to-send
    * 'Yakap is typing...' indicator while the model generates
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
    from backend import YakapBackend, CHATBOT_NAME, ORG_NAME, TAGLINE, HANDLE
except ImportError as exc:  # pragma: no cover
    raise SystemExit(f"Could not import backend.py: {exc}")


# ---------------------------------------------------------------------------
# Themes
# ---------------------------------------------------------------------------

THEME = {
    "bg": "#f3f5f2",
    "header": "#382a3c",
    "header_text": "#fffdfd",
    "user_bubble": "#dcefe9",
    "bot_bubble": "#ffffff",
    "bubble_border": "#d8e1dc",
    "text": "#283a37",
    "muted": "#71817c",
    "accent": "#b24f70",
    "accent_hover": "#963b5a",
    "logo_bg": "#f8e3ea",
    "logo_mark": "#8e3450",
    "online": "#a6e2c0",
    "mark": "heart",
}
FONT_FAMILY = "TkDefaultFont"


# ---------------------------------------------------------------------------
# Widgets
# ---------------------------------------------------------------------------

class ProfileCircle(tk.Canvas):
    """A small themed health mark used as the account profile picture."""

    def __init__(self, parent, theme=None, size=46, **kwargs):
        self.theme = theme or THEME
        super().__init__(parent, width=size, height=size,
                         highlightthickness=0, bg=self.theme["header"], **kwargs)
        self.size = size
        self._draw()

    def _draw(self):
        s = self.size
        mark_color = self.theme["logo_mark"]
        self.create_oval(2, 2, s - 2, s - 2, fill=self.theme["logo_bg"], outline="")
        if self.theme["mark"] == "heart":
            self.create_polygon(
                23, 36, 9, 23, 8, 17, 11, 11, 17, 9, 23, 15,
                29, 9, 35, 11, 38, 17, 37, 23,
                fill=mark_color, outline="", smooth=True,
            )
            self.create_text(23, 22, text="+", fill="#ffffff",
                             font=(FONT_FAMILY, 12, "bold"))
            return

        self.create_oval(11, 8, 16, 13, fill=mark_color, outline="")
        self.create_oval(30, 8, 35, 13, fill=mark_color, outline="")
        self.create_line(
            13.5, 11, 13.5, 19, 17, 24, 22, 27, 24, 27,
            29, 24, 32.5, 19, 32.5, 11,
            smooth=True, splinesteps=16, fill=mark_color, width=2.5,
        )
        self.create_line(24, 27, 24, 32, 29, 34, fill=mark_color, width=2.5)
        self.create_oval(28, 32, 35, 39, fill=mark_color, outline="")
        self.create_oval(31, 31, 44, 44, fill="#ffffff",
                         outline=self.theme["bubble_border"])
        self.create_text(37.5, 37.5, text="+", fill=self.theme["accent"],
                         font=(FONT_FAMILY, 10, "bold"))


class ChatBubble(tk.Frame):
    """A single chat bubble with message text and timestamp."""

    def __init__(self, parent, message, timestamp, is_user, theme=None, **kwargs):
        self.theme = theme or THEME
        super().__init__(parent, bg=self.theme["bg"], **kwargs)
        self.is_user = is_user

        bubble_bg = self.theme["user_bubble"] if is_user else self.theme["bot_bubble"]
        text_color = self.theme["text"]
        anchor = "e" if is_user else "w"
        padx = (48, 12) if is_user else (12, 48)

        # The bubble (label with wrapped text + border effect via frame bg)
        self.bubble = tk.Frame(self, bg=bubble_bg, bd=0, highlightthickness=0)
        self.bubble.pack(anchor=anchor, padx=padx, pady=(4, 0))

        tk.Label(
            self.bubble,
            text="YOU" if is_user else "YAKAP",
            bg=bubble_bg,
            fg=self.theme["text"] if is_user else self.theme["accent"],
            font=(FONT_FAMILY, 8, "bold"),
        ).pack(anchor="w", padx=13, pady=(8, 0))

        self.msg_label = tk.Label(
            self.bubble, text=message, bg=bubble_bg, fg=text_color,
            font=(FONT_FAMILY, 10), justify=tk.LEFT, wraplength=280,
            padx=13, pady=4, anchor="w",
        )
        self.msg_label.pack()

        # Bot bubbles get a thin border effect by wrapping bg frame
        if not is_user:
            self.bubble.config(highlightbackground=self.theme["bubble_border"],
                               highlightthickness=1)

        # Timestamp below the bubble
        self.time_label = tk.Label(
            self, text=timestamp, bg=self.theme["bg"], fg=self.theme["muted"],
            font=(FONT_FAMILY, 8),
        )
        self.time_label.pack(anchor=anchor, padx=padx, pady=(2, 5))


class TypingIndicator(tk.Frame):
    """'Yakap is typing...' indicator shown while the model generates."""

    def __init__(self, parent, chatbot_name=CHATBOT_NAME, theme=None, **kwargs):
        self.chatbot_name = chatbot_name
        self.theme = theme or THEME
        super().__init__(parent, bg=self.theme["bg"], **kwargs)
        self.label = tk.Label(
            self, text=f"{self.chatbot_name} is typing", bg=self.theme["bg"],
            fg=self.theme["accent"],
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
        self.label.config(text=f"{self.chatbot_name} is typing" + "." * self.dots)
        self._job = self.after(500, self._animate)


# ---------------------------------------------------------------------------
# Main application
# ---------------------------------------------------------------------------

class YakapApp:
    def __init__(
        self,
        root,
        chatbot_name=CHATBOT_NAME,
        organization_name=ORG_NAME,
        tagline=TAGLINE,
        handle=HANDLE,
        theme=None,
    ):
        self.root = root
        self.chatbot_name = chatbot_name
        self.organization_name = organization_name
        self.tagline = tagline
        self.handle = handle
        self.theme = theme or THEME
        self.root.title(f"{self.chatbot_name} - {self.handle}")
        self.root.geometry("460x760")
        self.root.minsize(410, 640)
        self.root.configure(bg=self.theme["bg"])

        self.backend = YakapBackend(
            chatbot_name=self.chatbot_name,
            organization_name=self.organization_name,
            handle=self.handle,
            tagline=self.tagline,
        )
        self._stream_job = None  # holds after() id for word-by-word streaming
        self._stream_words = []  # remaining words to stream
        self._stream_index = 0

        self._build_ui()
        self._add_welcome_message()

    # -- UI construction --------------------------------------------------

    def _build_ui(self):
        theme = self.theme
        header = tk.Frame(self.root, bg=theme["bg"])
        header.pack(fill=tk.X, padx=14, pady=(14, 4))

        profile_card = tk.Frame(header, bg=theme["header"], height=96)
        profile_card.pack(fill=tk.X)
        profile_card.pack_propagate(False)

        prof = ProfileCircle(profile_card, theme=theme, size=54)
        prof.pack(side=tk.LEFT, padx=(16, 13), pady=20)

        info = tk.Frame(profile_card, bg=theme["header"])
        info.pack(side=tk.LEFT, fill=tk.Y, expand=True, pady=(12, 8))
        tk.Label(
            info, text="YAKAP CARE DESK", bg=theme["header"],
            fg="#f0cbd7", font=(FONT_FAMILY, 8, "bold"),
        ).pack(anchor="w")
        tk.Label(
            info, text=self.chatbot_name, bg=theme["header"],
            fg=theme["header_text"], font=(FONT_FAMILY, 20, "bold"),
        ).pack(anchor="w", pady=(1, 0))
        tk.Label(
            info, text=f"{self.handle}  \u00b7  Online",
            bg=theme["header"], fg=theme["online"],
            font=(FONT_FAMILY, 9),
        ).pack(anchor="w")

        tagline_bar = tk.Frame(
            header, bg=theme["bot_bubble"],
            highlightbackground=theme["bubble_border"], highlightthickness=1,
        )
        tagline_bar.pack(fill=tk.X)
        tk.Label(
            tagline_bar, text=self.tagline, bg=theme["bot_bubble"],
            fg=theme["accent"], font=(FONT_FAMILY, 10, "bold"),
        ).pack(anchor="w", padx=14, pady=9)

        quick = tk.Frame(self.root, bg=theme["bg"])
        quick.pack(fill=tk.X, padx=14, pady=(9, 3))
        tk.Label(
            quick, text="START WITH A TOPIC", bg=theme["bg"],
            fg=theme["muted"], font=(FONT_FAMILY, 8, "bold"),
        ).grid(row=0, column=0, columnspan=2, sticky="w", pady=(0, 6))
        self.quick_buttons = []
        topics = ["Hospital services", "Hours & schedule", "Location", "Emergency help"]
        for column in range(2):
            quick.grid_columnconfigure(column, weight=1, uniform="quick-replies")
        for index, label in enumerate(topics):
            row, column = divmod(index, 2)
            button = tk.Button(
                quick, text=label, bg=theme["bot_bubble"], fg=theme["text"],
                font=(FONT_FAMILY, 9, "bold"), bd=0, padx=8, pady=9,
                highlightbackground=theme["bubble_border"], highlightthickness=1,
                cursor="hand2", activebackground=theme["user_bubble"],
                activeforeground=theme["text"],
                command=lambda value=label: self._send_quick(value),
            )
            button.grid(
                row=row + 1, column=column, sticky="ew",
                padx=(0, 6) if column == 0 else (6, 0), pady=3,
            )
            self.quick_buttons.append(button)

        chat_container = tk.Frame(self.root, bg=theme["bg"])
        chat_container.pack(fill=tk.BOTH, expand=True, padx=12, pady=(4, 2))

        self.canvas = tk.Canvas(chat_container, bg=theme["bg"], highlightthickness=0)
        self.scrollbar = ttk.Scrollbar(
            chat_container, orient="vertical", command=self.canvas.yview
        )
        self.chat_frame = tk.Frame(self.canvas, bg=theme["bg"])
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

        self.typing = TypingIndicator(
            self.root, chatbot_name=self.chatbot_name, theme=theme
        )

        input_frame = tk.Frame(self.root, bg=theme["bg"])
        input_frame.pack(fill=tk.X, padx=12, pady=(4, 12))
        composer = tk.Frame(
            input_frame, bg=theme["bot_bubble"], highlightbackground=theme["bubble_border"],
            highlightthickness=1,
        )
        composer.pack(fill=tk.X)

        self.entry = tk.Entry(
            composer, font=(FONT_FAMILY, 11), bd=0, relief=tk.FLAT,
            bg=theme["bot_bubble"], fg=theme["text"], insertbackground=theme["accent"],
        )
        self.entry.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=(12, 6), ipady=11)
        self.entry.bind("<Return>", lambda event: self._send_message())

        self.send_button = tk.Button(
            composer, text="Send", bg=theme["accent"], fg=theme["header_text"],
            font=(FONT_FAMILY, 10, "bold"), bd=0, padx=17, pady=9,
            cursor="hand2", activebackground=theme["accent_hover"],
            activeforeground=theme["header_text"], command=self._send_message,
        )
        self.send_button.pack(side=tk.RIGHT, padx=5, pady=5)

        self.clear_button = tk.Button(
            input_frame, text="Clear chat", bg=theme["bg"], fg=theme["muted"],
            font=(FONT_FAMILY, 9), bd=0, padx=8, pady=5, cursor="hand2",
            activebackground=theme["bg"], activeforeground=theme["text"],
            command=self._clear_chat,
        )
        self.clear_button.pack(anchor="e", pady=(4, 0))
        self.entry.focus_set()

    # -- chat helpers -----------------------------------------------------

    def _add_welcome_message(self):
        welcome = (
            f"Hello! I'm {self.chatbot_name}, the healthcare information assistant "
            f"for {self.organization_name} ({self.handle}).\n\n"
            "I can help with "
            "verified hospital information and general health "
            "education. I cannot diagnose or provide personal medical consultation.\n\n"
            "\u26a0\ufe0f If you're having a medical emergency, please call 911 or go "
            "to the nearest hospital immediately."
        )
        self._add_bot_message(welcome, animate=False)

    def _add_bubble(self, message, timestamp, is_user, animate=False):
        bubble = ChatBubble(
            self.chat_frame, message, timestamp, is_user, theme=self.theme
        )
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
        bubble = _StreamableBubble(self.chat_frame, "", ts, theme=self.theme)
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

    def __init__(self, parent, message, timestamp, theme=None, **kwargs):
        self.theme = theme or THEME
        super().__init__(parent, bg=self.theme["bg"], **kwargs)
        self.timestamp = timestamp

        self.bubble = tk.Frame(self, bg=self.theme["bot_bubble"], bd=0,
                               highlightbackground=self.theme["bubble_border"],
                               highlightthickness=1)
        self.bubble.pack(anchor="w", padx=(10, 60), pady=(4, 0))

        tk.Label(
            self.bubble, text="YAKAP", bg=self.theme["bot_bubble"],
            fg=self.theme["accent"], font=(FONT_FAMILY, 8, "bold"),
        ).pack(anchor="w", padx=13, pady=(8, 0))

        self.msg_label = tk.Label(
            self.bubble, text=message, bg=self.theme["bot_bubble"],
            fg=self.theme["text"],
            font=(FONT_FAMILY, 10), justify=tk.LEFT, wraplength=280,
            padx=13, pady=4, anchor="w",
        )
        self.msg_label.pack()

        self.time_label = tk.Label(
            self, text=timestamp, bg=self.theme["bg"], fg=self.theme["muted"],
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
    # Make imports work when run directly from the Yakap folder.
    import sys
    import os
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

    root = tk.Tk()
    app = YakapApp(root)
    app.run()
