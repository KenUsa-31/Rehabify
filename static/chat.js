const form = document.querySelector("#chat-form");
const input = document.querySelector("#message-input");
const sendButton = document.querySelector("#send-button");
const characterCount = document.querySelector("#character-count");
const messages = document.querySelector("#messages");
const assistantState = document.querySelector("#assistant-state");
const newChatButton = document.querySelector("#new-chat");
const welcomeMessage = messages.querySelector("[data-welcome]").cloneNode(true);
const conversationHistory = [];

function setStatus(label, state = "") {
  assistantState.className = `assistant-state ${state}`.trim();
  assistantState.lastElementChild.textContent = label;
}

function scrollToLatest() {
  messages.scrollTop = messages.scrollHeight;
}

function makeMessage(text, isUser) {
  const row = document.createElement("article");
  row.className = `message-row ${isUser ? "user-message" : "assistant-message"}`;

  const avatar = document.createElement("div");
  avatar.className = "message-avatar";
  avatar.setAttribute("aria-hidden", "true");
  avatar.textContent = isUser ? "YOU" : "R";

  const stack = document.createElement("div");
  stack.className = "message-stack";

  const sender = document.createElement("span");
  sender.className = "message-sender";
  sender.textContent = isUser ? "You" : "Rehabify Health · guide";

  const bubble = document.createElement("div");
  bubble.className = "message-bubble";
  bubble.textContent = text;

  const time = document.createElement("span");
  time.className = "message-time";
  time.textContent = new Intl.DateTimeFormat(undefined, {
    hour: "numeric",
    minute: "2-digit",
  }).format(new Date());

  stack.append(sender, bubble, time);
  row.append(avatar, stack);
  messages.append(row);
  scrollToLatest();
  return { bubble, stack };
}

function appendStreamEvent(event, reply) {
  if (event.token) {
    reply.bubble.classList.remove("is-streaming");
    reply.bubble.textContent += event.token;
    reply.content += event.token;
    scrollToLatest();
  }

  if (event.done) {
    reply.complete = true;
    if (event.topics?.length) {
      const source = document.createElement("span");
      source.className = "source-note";
      source.textContent = `Related topics: ${event.topics.join(" · ")}`;
      reply.stack.append(source);
    }
    setStatus(event.emergency ? "Urgent guidance" : "Ready", event.emergency ? "is-alert" : "");
  }

  if (event.error) {
    reply.failed = true;
    reply.bubble.textContent = event.error;
    setStatus("Connection issue", "is-alert");
  }
}

async function consumeEventStream(response, reply) {
  const reader = response.body.getReader();
  const decoder = new TextDecoder();
  let buffer = "";

  function readEvent(block) {
    const dataLine = block.split(/\r?\n/).find((line) => line.startsWith("data: "));
    if (!dataLine) return;
    appendStreamEvent(JSON.parse(dataLine.slice(6)), reply);
  }

  while (true) {
    const { value, done } = await reader.read();
    buffer += decoder.decode(value || new Uint8Array(), { stream: !done });
    let boundary = buffer.indexOf("\n\n");
    while (boundary !== -1) {
      readEvent(buffer.slice(0, boundary));
      buffer = buffer.slice(boundary + 2);
      boundary = buffer.indexOf("\n\n");
    }
    if (done) break;
  }

  if (buffer.trim()) readEvent(buffer);
}

async function sendMessage(message) {
  makeMessage(message, true);
  const reply = makeMessage("", false);
  reply.content = "";
  reply.complete = false;
  reply.failed = false;
  reply.bubble.classList.add("is-streaming");
  setStatus("Writing a reply", "is-busy");
  input.disabled = true;
  sendButton.disabled = true;
  newChatButton.disabled = true;

  try {
    const response = await fetch("/api/chat", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ message, history: conversationHistory.slice(-6) }),
    });

    if (!response.ok) {
      const payload = await response.json();
      throw new Error(payload.error || "The message could not be sent.");
    }

    await consumeEventStream(response, reply);
    reply.bubble.classList.remove("is-streaming");
    if (reply.complete && !reply.failed) {
      conversationHistory.push(
        { role: "user", content: message },
        { role: "assistant", content: reply.content },
      );
      conversationHistory.splice(0, Math.max(0, conversationHistory.length - 6));
    }
  } catch (error) {
    reply.bubble.classList.remove("is-streaming");
    reply.bubble.textContent = error.message || "Could not reach Rehabify Health. Please try again.";
    setStatus("Connection issue", "is-alert");
  } finally {
    input.disabled = false;
    newChatButton.disabled = false;
    input.focus();
    updateComposer();
  }
}

function updateComposer() {
  const length = input.value.length;
  characterCount.textContent = `${length} / 2000`;
  sendButton.disabled = input.disabled || !input.value.trim();
  input.style.height = "auto";
  input.style.height = `${Math.min(input.scrollHeight, 128)}px`;
}

form.addEventListener("submit", (event) => {
  event.preventDefault();
  const message = input.value.trim();
  if (!message || input.disabled) return;
  input.value = "";
  updateComposer();
  sendMessage(message);
});

input.addEventListener("input", updateComposer);
input.addEventListener("keydown", (event) => {
  if (event.key === "Enter" && !event.shiftKey) {
    event.preventDefault();
    form.requestSubmit();
  }
});

document.querySelectorAll("[data-prompt]").forEach((button) => {
  button.addEventListener("click", () => {
    if (input.disabled) return;
    input.value = button.dataset.prompt;
    updateComposer();
    form.requestSubmit();
  });
});

newChatButton.addEventListener("click", async () => {
  if (newChatButton.disabled) return;
  newChatButton.disabled = true;
  try {
    const response = await fetch("/api/clear", { method: "POST" });
    if (!response.ok) throw new Error("Could not clear the conversation.");
    conversationHistory.length = 0;
    messages.replaceChildren(welcomeMessage.cloneNode(true));
    setStatus("Ready");
    input.value = "";
    updateComposer();
    input.focus();
  } catch (error) {
    setStatus(error.message, "is-alert");
  } finally {
    newChatButton.disabled = false;
  }
});

updateComposer();