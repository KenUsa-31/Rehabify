"""backend.py - AI logic for Rehabify Health, the rehabilitation information assistant.

This module handles everything related to the AI backend:
  * Loading and chunking the knowledge base (knowledge_base.json)
  * Retrieving the most relevant chunks using TF-IDF + cosine similarity (RAG)
  * Building a persona-driven system prompt
  * Managing short-term conversation memory (last 6 messages)
    * Applying safety guardrails (emergency detection and health safety)
    * Generating answers via local Ollama or a hosted OpenAI-compatible model
  * Logging every interaction to chat_log.csv

Run this module through app.py or frontend.py.
"""

import csv
import json
import os
import re
import threading
import time
from datetime import datetime
from urllib.error import HTTPError
from urllib.request import Request, urlopen

try:
    import ollama
except ImportError:  # pragma: no cover - handled at runtime with a friendly error
    ollama = None

try:
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.metrics.pairwise import cosine_similarity
    import numpy as np
except ImportError:  # pragma: no cover - handled at runtime with a friendly error
    TfidfVectorizer = None
    cosine_similarity = None
    np = None


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
MODEL_NAME = os.environ.get(
    "REHABIFY_MODEL",
    os.environ.get("REHABLINK_MODEL", os.environ.get("YAKAP_MODEL", "gpt-oss:20b")),
)
OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY", "")
OPENAI_BASE_URL = os.environ.get("OPENAI_BASE_URL", "https://api.openai.com/v1").rstrip("/")
OPENAI_MODEL = os.environ.get("OPENAI_MODEL", "gpt-4o-mini")
OLLAMA_OPTIONS = {"num_predict": 192}
MODEL_KEEP_ALIVE = "30m"
ORG_NAME = "Rehabify Health"  # Fictional organization for the demo
CHATBOT_NAME = "Rehabify Health"
TAGLINE = "Rehabify Health: supporting every step of recovery"
HANDLE = "@RehabifyHealth"
EMERGENCY_HOTLINE = "911"           # Local emergency hotline (change as needed)

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
KB_FILE = os.path.join(BASE_DIR, "knowledge_base.json")
LOG_FILE = os.path.join(BASE_DIR, "chat_log.csv")

MEMORY_SIZE = 6                     # Number of past messages to keep in memory
SIMILARITY_THRESHOLD = 0.15         # Minimum cosine similarity to treat a chunk as relevant

# Keywords that signal a medical emergency. These trigger a fixed safety message
# BEFORE the model is ever called.
EMERGENCY_KEYWORDS = [
    "chest pain", "difficulty breathing", "can't breathe", "cannot breathe",
    "shortness of breath", "severe bleeding", "heavy bleeding", "stroke",
    "heart attack", "suicide", "suicidal", "kill myself", "end my life",
    "overdose", "poisoning", "unconscious", "not breathing", "seizure",
    "hirap huminga", "hirap akong huminga", "nahihirapang huminga",
    "hindi makahinga", "di makahinga",
    "matinding pagdurugo", "malakas na pagdurugo", "atake sa puso",
    "magpakamatay", "na-overdose",
]

# Health-domain keywords determine when the health-information disclaimer is needed.
HEALTH_KEYWORDS = [
    "health", "hospital", "doctor", "clinic", "patient", "medicine", "medical",
    "disease", "illness", "symptom", "treatment", "appointment", "vaccine",
    "vaccination", "insurance", "billing", "pharmacy", "emergency", "pain",
    "fever", "injury", "surgery", "pregnancy", "baby", "child", "heart",
    "blood", "bp", "blood pressure", "diabetes", "covid", "flu",
    "visiting", "visitor", "schedule", "schedule", "located", "location",
    "address", "service", "services", "department", "pediatric", "pediatrics",
    "cardiology", "obstetrics", "gynecology", "radiology", "laboratory",
    "headache", "cough", "rash", "allergy", "allergies", "shingles",
    "diagnose", "prescribe", "rehabilitation", "rehab", "physical therapy",
    "occupational therapy", "speech therapy", "physiotherapy", "mobility",
    "recovery", "stroke recovery", "assistive device", "prosthetic", "amputation",
    "kalusugan", "ospital", "doktor", "gamot", "rehabilitasyon", "pagpapagaling",
    "sintomas", "lagnat", "ubo", "masakit", "sakit",
]

SAFETY_MESSAGE = (
    "\u26a0\ufe0f It sounds like you may be experiencing a medical emergency. "
    "Please call your local emergency hotline at {hotline} or go to the nearest "
    "hospital emergency department immediately. Do not wait for a reply here."
).format(hotline=EMERGENCY_HOTLINE)

DISCLAIMER = (
    "\n\n---\n*Disclaimer: Rehabify Health is an AI assistant, not a doctor, nurse, or other "
    "licensed healthcare professional. This is general educational information, "
    "not a personal medical consultation, diagnosis, or treatment plan. Please "
    "consult a qualified healthcare professional for advice about your situation.*"
)

# ---------------------------------------------------------------------------
# Knowledge base + RAG retrieval
# ---------------------------------------------------------------------------

class KnowledgeBase:
    """Loads knowledge_base.json, splits entries into chunks, and builds a
    TF-IDF index so we can retrieve the top-k most relevant chunks for any query.
    """

    def __init__(self, filepath=KB_FILE):
        self.filepath = filepath
        self.entries = self._load_entries()
        # Each chunk: dict with "text", "topic", "source"
        self.chunks = []
        self.vectorizer = None
        self.tfidf_matrix = None
        self._build_index()

    def _load_entries(self):
        if not os.path.exists(self.filepath):
            return []
        with open(self.filepath, "r", encoding="utf-8") as f:
            data = json.load(f)
        if not isinstance(data, list):
            return []
        return data

    def _split_into_chunks(self, text, max_words=80):
        """Split a content string into smaller chunks of roughly max_words words.
        Smaller chunks give TF-IDF more precise retrieval signals.
        """
        words = text.split()
        chunks = []
        for i in range(0, len(words), max_words):
            chunk = " ".join(words[i:i + max_words])
            if chunk.strip():
                chunks.append(chunk)
        return chunks

    def _build_index(self):
        for entry in self.entries:
            topic = entry.get("topic", "General")
            content = entry.get("content", "")
            source = entry.get("source", ORG_NAME)
            for chunk in self._split_into_chunks(content):
                self.chunks.append({"text": chunk, "topic": topic, "source": source})

        if TfidfVectorizer is None or not self.chunks:
            self.vectorizer = None
            self.tfidf_matrix = None
            return

        texts = [c["text"] for c in self.chunks]
        self.vectorizer = TfidfVectorizer(stop_words="english")
        self.tfidf_matrix = self.vectorizer.fit_transform(texts)

    def retrieve(self, query, top_k=3):
        """Return the top_k most relevant chunks (with topic & source) for query."""
        results = []
        if self.vectorizer is None or self.tfidf_matrix is None or not self.chunks:
            return results

        query_vec = self.vectorizer.transform([query])
        sims = cosine_similarity(query_vec, self.tfidf_matrix).flatten()
        top_indices = sims.argsort()[::-1][:top_k]

        for idx in top_indices:
            if sims[idx] >= SIMILARITY_THRESHOLD:
                results.append({
                    "text": self.chunks[idx]["text"],
                    "topic": self.chunks[idx]["topic"],
                    "source": self.chunks[idx]["source"],
                    "similarity": float(sims[idx]),
                })
        return results


# ---------------------------------------------------------------------------
# Backend chatbot
# ---------------------------------------------------------------------------

class RehabLinkBackend:
    """Main chatbot backend. Combines RAG retrieval, memory, guardrails, and
    Model generation into a single conversational interface.
    """

    def __init__(
        self,
        chatbot_name=CHATBOT_NAME,
        organization_name=ORG_NAME,
        handle=HANDLE,
        tagline=TAGLINE,
        knowledge_base=None,
    ):
        self.chatbot_name = chatbot_name
        self.organization_name = organization_name
        self.handle = handle
        self.tagline = tagline
        self.kb = knowledge_base or KnowledgeBase(KB_FILE)
        self.memory = []  # list of {"role": "user"/"assistant", "content": str}
        self.lock = threading.Lock()
        self.last_metadata = {}  # metadata from the most recent streamed response

    # -- memory -----------------------------------------------------------

    def _add_memory(self, role, content):
        self.memory.append({"role": role, "content": content})
        if len(self.memory) > MEMORY_SIZE:
            self.memory = self.memory[-MEMORY_SIZE:]

    def _get_memory_messages(self):
        return list(self.memory)

    # -- system prompt ----------------------------------------------------

    def _build_system_prompt(self):
        """Build the configured chatbot persona, rules, and disclaimer."""
        return (
            f"You are {self.chatbot_name}, the social media assistant for "
            f"{self.organization_name}. {self.tagline}. Social media handle: "
            f"{self.handle}.\n\n"
            "PERSONA:\n"
            "You are a warm, friendly, professional rehabilitation information assistant. "
            "Focus on rehabilitation, recovery, function, participation, accessibility, "
            "and the roles of the rehabilitation team. Speak in a gentle, guide-like tone. "
            "Use light emoji use (e.g. \u2705, \u26a0\ufe0f, "
            "\u270d\ufe0f, \u2764\ufe0f) but do not overdo it. Be concise and easy to read, "
            "as users are reading you on social media.\n\n"
            "LANGUAGE:\n"
            "Reply in English by default. If the user writes in Filipino or Taglish "
            "(a mix of Filipino and English), reply in the same Filipino/Taglish style. "
            "Keep the tone friendly, professional, and caring regardless of language.\n\n"
            "RULES:\n"
            "1. Keep the conversation centered on rehabilitation and general recovery "
            "education. Answer unrelated questions briefly, then invite the user back "
            "to rehabilitation. Be clear when information may be uncertain or out of date.\n"
            "2. {org} is a fictional demo organization. Use only retrieved context for "
            "organization-specific facts. Never invent services, prices, hours, policies, "
            "contact details, or availability. If the context does not answer a "
            "facility-specific question, say exactly: 'I don't have that information.'\n"
            "3. You are not a doctor, nurse, or clinician, and you cannot examine or "
            "assess the user. For health questions, provide general educational "
            "information only, not a personal consultation, rehabilitation assessment, "
            "exercise program, or treatment plan. Never diagnose, prescribe, set "
            "individualized exercise intensity, or recommend a treatment. Explain that "
            "rehabilitation goals and activities must be planned with the person's "
            "qualified care team. Encourage a clinician for personal advice.\n"
            "4. For medical emergencies (chest pain, difficulty breathing, severe "
            "bleeding, suicidal thoughts, stroke, etc.), tell the user to call the "
            "local emergency hotline at {hotline} or go to the nearest hospital "
            "emergency department immediately.\n"
            "5. Do not claim to have live web access or current information unless it "
            "is included in the retrieved context.\n"
            "6. Always end health-related answers with the disclaimer below.\n\n"
            "DISCLAIMER (append to health-related answers):\n"
            "{disclaimer}\n\n"
            "Retrieved context from the knowledge base will be provided below your "
            "instructions. Use it to ground your answer and mention the source when "
            "relevant."
        ).format(
            hotline=EMERGENCY_HOTLINE,
            org=self.organization_name,
            disclaimer=DISCLAIMER.strip("*"),
        )

    # -- guardrails -------------------------------------------------------

    def _is_emergency(self, text):
        lowered = text.lower()
        return any(kw in lowered for kw in EMERGENCY_KEYWORDS)

    def _is_health_related(self, text):
        lowered = text.lower()
        return any(kw in lowered for kw in HEALTH_KEYWORDS)

    # -- prompt construction ----------------------------------------------

    def _build_user_prompt(self, query, retrieved):
        """Combine retrieved context (RAG) with the user question."""
        context_parts = []
        sources = []
        for chunk in retrieved:
            context_parts.append(f"[{chunk['topic']}] {chunk['text']}")
            sources.append(f"{chunk['topic']} ({chunk['source']})")

        context = (
            "\n".join(context_parts)
            if context_parts
            else "No relevant knowledge-base context was found."
        )
        source_list = "; ".join(sources) if sources else "none"

        return (
            f"RETRIEVED CONTEXT (from knowledge base):\n{context}\n\n"
            f"SOURCES: {source_list}\n\n"
            f"USER QUESTION:\n{query}\n\n"
            f"Use retrieved context for {self.organization_name}-specific facts and cite "
            "its source when relevant. If the context is empty, provide general "
            "rehabilitation education cautiously, but do not guess facility-specific facts. "
            "Follow your persona and rules above."
        )

    # -- ollama generation ------------------------------------------------

    def _ollama_available(self):
        return ollama is not None

    @staticmethod
    def _field(value, name, default=None):
        """Read a field from either an Ollama dict response or typed object."""
        if isinstance(value, dict):
            return value.get(name, default)
        return getattr(value, name, default)

    def _model_available(self):
        if OPENAI_API_KEY:
            return True, None
        if os.environ.get("VERCEL"):
            return False, (
                "Hosted model is not configured. Add OPENAI_API_KEY and "
                "OPENAI_MODEL to the Vercel project environment variables."
            )
        if not self._ollama_available():
            return False, "Ollama Python library is not installed."
        try:
            models = ollama.list()
            names = []
            for model in self._field(models, "models", []):
                name = self._field(model, "model", "")
                if name:
                    names.append(name)
            if MODEL_NAME not in names:
                return False, (
                    f"Model '{MODEL_NAME}' not found in Ollama. "
                    f"Run: ollama pull {MODEL_NAME}"
                )
            return True, None
        except Exception as exc:  # Ollama not running / connection error
            return False, (
                f"Could not connect to Ollama. Make sure Ollama is running. "
                f"Error: {exc}"
            )

    def _build_messages(self, query, retrieved):
        messages = [{"role": "system", "content": self._build_system_prompt()}]
        messages.extend(self._get_memory_messages())
        messages.append({
            "role": "user",
            "content": self._build_user_prompt(query, retrieved),
        })
        return messages

    @staticmethod
    def _openai_request(messages, stream):
        payload = {
            "model": OPENAI_MODEL,
            "messages": messages,
            "max_tokens": OLLAMA_OPTIONS["num_predict"],
            "stream": stream,
        }
        request = Request(
            f"{OPENAI_BASE_URL}/chat/completions",
            data=json.dumps(payload).encode("utf-8"),
            headers={
                "Authorization": f"Bearer {OPENAI_API_KEY}",
                "Content-Type": "application/json",
            },
            method="POST",
        )
        try:
            return urlopen(request, timeout=60)
        except HTTPError as exc:
            raise RuntimeError(
                f"Hosted model request failed with HTTP {exc.code}."
            ) from exc

    @classmethod
    def _openai_stream(cls, messages):
        with cls._openai_request(messages, stream=True) as response:
            for raw_line in response:
                line = raw_line.decode("utf-8").strip()
                if not line.startswith("data: "):
                    continue
                data = line[6:]
                if data == "[DONE]":
                    break
                chunk = json.loads(data)
                choices = chunk.get("choices", [])
                if choices:
                    piece = choices[0].get("delta", {}).get("content", "")
                    if isinstance(piece, str) and piece:
                        yield piece

    @classmethod
    def _openai_generate(cls, messages):
        with cls._openai_request(messages, stream=False) as response:
            result = json.load(response)
        return result["choices"][0]["message"].get("content", "")

    def generate(self, query, retrieved):
        """Non-streaming generation. Returns the full assistant response string."""
        ok, err = self._model_available()
        if not ok:
            return f"\u26a0\ufe0f Backend error: {err}"

        messages = self._build_messages(query, retrieved)

        try:
            if OPENAI_API_KEY:
                content = self._openai_generate(messages)
            else:
                response = ollama.chat(
                    model=MODEL_NAME,
                    messages=messages,
                    options=OLLAMA_OPTIONS,
                    keep_alive=MODEL_KEEP_ALIVE,
                )
                message = self._field(response, "message", {})
                content = self._field(message, "content", "")
        except Exception as exc:
            content = f"\u26a0\ufe0f Error generating response: {exc}"
        return content

    def stream_generate(self, query, retrieved):
        """Streaming generator yielding response chunks word-by-word.
        After exhaustion, self.last_metadata holds {"response_time": float}.
        """
        start = time.time()
        ok, err = self._model_available()
        if not ok:
            yield f"\u26a0\ufe0f Backend error: {err}"
            self.last_metadata = {"response_time": time.time() - start}
            return

        messages = self._build_messages(query, retrieved)

        try:
            full = []
            if OPENAI_API_KEY:
                pieces = self._openai_stream(messages)
            else:
                stream = ollama.chat(
                    model=MODEL_NAME,
                    messages=messages,
                    stream=True,
                    options=OLLAMA_OPTIONS,
                    keep_alive=MODEL_KEEP_ALIVE,
                )
                pieces = (
                    self._field(self._field(chunk, "message", {}), "content", "")
                    for chunk in stream
                )
            for piece in pieces:
                if piece:
                    full.append(piece)
                    yield piece
        except Exception as exc:
            yield f"\u26a0\ufe0f Error generating response: {exc}"
            full_text = ""
        else:
            full_text = "".join(full)

        self.last_metadata = {"response_time": time.time() - start}

    # -- public API -------------------------------------------------------

    def get_response(self, user_message):
        """Process one user message end-to-end:
        1. Emergency guardrail (fixed safety message, no model call).
        2. RAG retrieval (top-3 chunks, when relevant).
        3. Configured model generation grounded in retrieved context or general knowledge.
        4. Keep facility-specific facts grounded in the knowledge base.
        5. Update memory and log the interaction to chat_log.csv.

        Returns a dict with keys: response, topics, response_time, emergency.
        """
        result = {
            "response": "",
            "topics": [],
            "response_time": 0.0,
            "emergency": False,
        }
        start = time.time()

        # 1. Emergency guardrail \u2014 fixed safety message, no model call.
        if self._is_emergency(user_message):
            result["response"] = SAFETY_MESSAGE
            result["emergency"] = True
            result["response_time"] = time.time() - start
            self._add_memory("user", user_message)
            self._add_memory("assistant", result["response"])
            self._log(user_message, [], result["response"], result["response_time"])
            return result

        # 2. Retrieve relevant context via RAG.
        retrieved = self.kb.retrieve(user_message, top_k=3)
        result["topics"] = [r["topic"] for r in retrieved]

        # Generate with retrieved context when available, or general knowledge otherwise.
        response = self.generate(user_message, retrieved)
        result["response"] = response
        result["response_time"] = time.time() - start

        if self._is_health_related(user_message) and "Disclaimer" not in response:
            result["response"] += DISCLAIMER

        # Update memory and log the interaction.
        self._add_memory("user", user_message)
        self._add_memory("assistant", result["response"])

        self._log(user_message, result["topics"], result["response"], result["response_time"])
        return result

    def stream_response(self, user_message):
        """Streaming version of get_response. Yields text chunks; after the
        generator is exhausted, self.last_metadata contains the metadata dict.
        Emergency guardrails are applied before any model call.
        """
        start = time.time()
        metadata = {
            "response": "",
            "topics": [],
            "response_time": 0.0,
            "emergency": False,
        }

        if self._is_emergency(user_message):
            metadata["response"] = SAFETY_MESSAGE
            metadata["emergency"] = True
            metadata["response_time"] = time.time() - start
            for ch in metadata["response"]:
                yield ch
            self.last_metadata = metadata
            self._add_memory("user", user_message)
            self._add_memory("assistant", metadata["response"])
            self._log(user_message, [], metadata["response"], metadata["response_time"])
            return

        retrieved = self.kb.retrieve(user_message, top_k=3)
        metadata["topics"] = [r["topic"] for r in retrieved]

        # Stream from the configured model with retrieved context or general knowledge.
        collected = []
        for piece in self.stream_generate(user_message, retrieved):
            collected.append(piece)
            yield piece

        full = "".join(collected)
        metadata["response"] = full
        metadata["response_time"] = self.last_metadata.get("response_time", time.time() - start)
        if self._is_health_related(user_message) and "Disclaimer" not in full:
            metadata["response"] += DISCLAIMER
            # Re-yield the disclaimer so it appears in the stream.
            for ch in DISCLAIMER:
                yield ch

        self.last_metadata = metadata
        self._add_memory("user", user_message)
        self._add_memory("assistant", metadata["response"])
        self._log(user_message, metadata["topics"], metadata["response"], metadata["response_time"])

    # -- logging ----------------------------------------------------------

    def _log(self, question, topics, response, response_time):
        """Append one interaction row to chat_log.csv."""
        if os.environ.get("VERCEL"):
            return
        file_exists = os.path.exists(LOG_FILE)
        try:
            with open(LOG_FILE, "a", newline="", encoding="utf-8") as f:
                writer = csv.writer(f)
                if not file_exists:
                    writer.writerow([
                        "timestamp", "question", "retrieved_topics",
                        "response", "response_time_seconds",
                    ])
                writer.writerow([
                    datetime.now().isoformat(timespec="seconds"),
                    question,
                    "; ".join(topics),
                    response,
                    f"{response_time:.3f}",
                ])
        except Exception:
            # Logging must never break the chatbot.
            pass


# ---------------------------------------------------------------------------
# Quick self-test when run directly
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    bot = RehabLinkBackend()
    print("Knowledge base chunks indexed:", len(bot.kb.chunks))
    out = bot.get_response("What does rehabilitation mean?")
    print("Response:", out["response"][:200], "...")
    print("Topics:", out["topics"], "Time:", out["response_time"])
