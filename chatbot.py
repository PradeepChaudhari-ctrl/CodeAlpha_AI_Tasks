"""
CodeAlpha AI Internship — Task 2: Chatbot for FAQs
Author: CodeAlpha Intern
Description: An NLP-powered FAQ chatbot using TF-IDF cosine similarity,
             with a Tkinter chat UI. No external API key required.
"""

import tkinter as tk
from tkinter import scrolledtext
import threading
import string
import math
from collections import Counter

# ─────────────────────────── FAQ Dataset ─────────────────────────────
FAQ_DATA = [
    # General
    ("What is CodeAlpha?",
     "CodeAlpha is a leading software development company dedicated to driving innovation "
     "and excellence across emerging technologies, including AI, web, and mobile development."),
    ("What services does CodeAlpha offer?",
     "CodeAlpha offers software development, AI/ML solutions, web development, mobile app development, "
     "UI/UX design, and internship programs for students."),
    ("How can I contact CodeAlpha?",
     "You can reach CodeAlpha at services@codealpha.tech, WhatsApp: +91 9336576683, or visit www.codealpha.tech."),
    # Internship
    ("How do I apply for the CodeAlpha internship?",
     "You can apply for a CodeAlpha internship through their official website at www.codealpha.tech "
     "or by reaching out via email at services@codealpha.tech."),
    ("What is the duration of the internship?",
     "The internship duration is typically 1 month, during which interns complete AI project tasks "
     "and submit them for evaluation."),
    ("What perks does the internship offer?",
     "Internship perks include an offer letter, completion certificate (QR verified), unique ID certificate, "
     "letter of recommendation (based on performance), job placement support, and resume building assistance."),
    ("How many tasks do I need to complete?",
     "You need to complete at least 2 or 3 out of the 4 assigned tasks to be eligible for the internship certificate. "
     "Submitting only 1 task will be considered incomplete."),
    ("How do I submit my internship tasks?",
     "Submit your completed tasks via the submission form shared in your WhatsApp group. "
     "Upload source code to GitHub in a repository named CodeAlpha_ProjectName and post a video on LinkedIn."),
    ("Will I get a certificate after the internship?",
     "Yes! Upon successful completion of at least 2-3 tasks, you will receive a QR-verified completion certificate "
     "along with a unique ID certificate."),
    ("Can I get a letter of recommendation?",
     "Yes, a letter of recommendation is awarded based on your performance during the internship."),
    # AI Tasks
    ("What are the AI tasks in the internship?",
     "The 4 AI tasks are: (1) Language Translation Tool, (2) Chatbot for FAQs, "
     "(3) Music Generation with AI, and (4) Object Detection and Tracking."),
    ("What is Task 1 about?",
     "Task 1 involves building a Language Translation Tool with a UI where users can enter text, "
     "select languages, and get translations via an API like Google Translate."),
    ("What is Task 2 about?",
     "Task 2 involves building an FAQ Chatbot using NLP libraries like NLTK or SpaCy "
     "with cosine similarity or intent matching to find the best answer."),
    ("What is Task 3 about?",
     "Task 3 involves Music Generation using AI — collecting MIDI data, training an LSTM/GAN model, "
     "and generating new music sequences."),
    ("What is Task 4 about?",
     "Task 4 involves Object Detection and Tracking using YOLO or Faster R-CNN with OpenCV "
     "and tracking algorithms like SORT or Deep SORT."),
    # Technical
    ("What programming language is used?",
     "Python is the primary programming language used across all AI internship tasks."),
    ("What libraries are used in the chatbot task?",
     "The chatbot task recommends NLP libraries like NLTK or SpaCy for text preprocessing, "
     "along with cosine similarity or intent matching for question matching."),
    ("Do I need a paid API key for the translation task?",
     "You can use free alternatives like deep-translator (a free Google Translate wrapper) "
     "to avoid needing a paid API key."),
    ("Where should I upload my code?",
     "Upload your complete source code to GitHub in a repository named 'CodeAlpha_ProjectName'. "
     "Then post a LinkedIn video with your GitHub repo link."),
    ("What is cosine similarity?",
     "Cosine similarity is a metric that measures the angle between two text vectors. "
     "A value of 1 means identical, 0 means completely different. It's widely used in NLP for text matching."),
    # Greetings
    ("Hello", "Hello! 👋 I'm the CodeAlpha FAQ Chatbot. How can I help you today?"),
    ("Hi",    "Hi there! 😊 Ask me anything about CodeAlpha or the AI internship."),
    ("Who are you?",
     "I'm an AI-powered FAQ chatbot built for the CodeAlpha AI Internship Task 2. "
     "I answer questions about CodeAlpha and the internship program."),
    ("Thank you", "You're welcome! 😊 Feel free to ask if you have more questions."),
    ("Bye",   "Goodbye! 👋 Best of luck with your internship tasks!"),
]

# ─────────────────────────── NLP Engine ──────────────────────────────
class NLPEngine:
    """Lightweight TF-IDF + cosine similarity without external libraries."""

    def __init__(self, faq_data):
        self.questions = [q for q, _ in faq_data]
        self.answers   = [a for _, a in faq_data]
        self.vocab, self.tfidf_matrix = self._build_tfidf()

    # ── Text helpers ─────────────────────────────────────────────────
    @staticmethod
    def _tokenize(text: str) -> list[str]:
        text = text.lower().translate(str.maketrans("", "", string.punctuation))
        stop = {"a","an","the","is","it","in","on","of","to","do","i","you",
                "we","he","she","they","what","how","where","when","why","are",
                "was","were","be","been","being","have","has","had","this","that",
                "at","by","for","with","about","as","can","will","my","your"}
        return [w for w in text.split() if w not in stop and len(w) > 1]

    def _build_tfidf(self):
        tokenized_docs = [self._tokenize(q) for q in self.questions]
        vocab = sorted({t for doc in tokenized_docs for t in doc})
        idx   = {w: i for i, w in enumerate(vocab)}
        N     = len(tokenized_docs)

        # Document frequency
        df = Counter(w for doc in tokenized_docs for w in set(doc))
        idf = {w: math.log((N + 1) / (df[w] + 1)) + 1 for w in vocab}

        # TF-IDF matrix
        matrix = []
        for doc in tokenized_docs:
            tf = Counter(doc)
            total = len(doc) or 1
            vec = [tf.get(w, 0) / total * idf[w] for w in vocab]
            norm = math.sqrt(sum(v*v for v in vec)) or 1
            matrix.append([v / norm for v in vec])

        return vocab, matrix

    def _vectorize(self, text: str) -> list[float]:
        tokens = self._tokenize(text)
        if not tokens:
            return [0.0] * len(self.vocab)
        idx = {w: i for i, w in enumerate(self.vocab)}
        N   = len(self.tfidf_matrix)
        df  = Counter(w for doc_vec in self.tfidf_matrix for w in [])   # approximate
        idf_approx = {w: math.log((N + 1) / 2) + 1 for w in self.vocab}
        tf  = Counter(tokens)
        total = len(tokens)
        vec = []
        for w in self.vocab:
            val = (tf.get(w, 0) / total) * idf_approx.get(w, 1) if w in idx else 0
            vec.append(val)
        norm = math.sqrt(sum(v*v for v in vec)) or 1
        return [v / norm for v in vec]

    @staticmethod
    def _cosine(a: list[float], b: list[float]) -> float:
        return sum(x*y for x, y in zip(a, b))

    def get_answer(self, user_input: str, threshold: float = 0.15) -> str:
        query_vec = self._vectorize(user_input)
        scores = [self._cosine(query_vec, doc_vec) for doc_vec in self.tfidf_matrix]
        best_idx = max(range(len(scores)), key=lambda i: scores[i])
        if scores[best_idx] < threshold:
            return ("🤔 I'm not sure about that. Please try rephrasing, or contact CodeAlpha at "
                    "services@codealpha.tech for a detailed answer.")
        return self.answers[best_idx]


# ─────────────────────────── Chat UI ─────────────────────────────────
class ChatbotApp:
    def __init__(self, root: tk.Tk):
        self.root = root
        self.root.title("💬 CodeAlpha FAQ Chatbot — Task 2")
        self.root.geometry("720x580")
        self.root.configure(bg="#1e1e2e")
        self.root.resizable(True, True)
        self.engine = NLPEngine(FAQ_DATA)
        self._build_ui()
        self._bot_say("Hello! 👋 I'm the CodeAlpha FAQ Chatbot.\n"
                      "Ask me anything about CodeAlpha or the AI internship program!")

    def _build_ui(self):
        # Title
        tk.Label(self.root, text="💬  CodeAlpha FAQ Chatbot",
                 font=("Segoe UI", 16, "bold"), bg="#181825", fg="#cba6f7").pack(fill="x", pady=8)

        # Chat history
        self.chat_area = scrolledtext.ScrolledText(
            self.root, state="disabled", wrap="word", font=("Segoe UI", 11),
            bg="#313244", fg="#cdd6f4", relief="flat", bd=0, padx=12, pady=8)
        self.chat_area.pack(fill="both", expand=True, padx=16, pady=(0, 8))
        self.chat_area.tag_config("user", foreground="#89b4fa", font=("Segoe UI", 11, "bold"))
        self.chat_area.tag_config("bot",  foreground="#a6e3a1", font=("Segoe UI", 11))
        self.chat_area.tag_config("meta", foreground="#6c7086", font=("Segoe UI", 9))

        # Input row
        input_frame = tk.Frame(self.root, bg="#1e1e2e")
        input_frame.pack(fill="x", padx=16, pady=(0, 14))

        self.entry = tk.Entry(input_frame, font=("Segoe UI", 12),
                              bg="#313244", fg="#cdd6f4", insertbackground="#cdd6f4",
                              relief="flat", bd=0)
        self.entry.pack(side="left", fill="x", expand=True, ipady=8, padx=(0, 10))
        self.entry.bind("<Return>", lambda _: self._send())

        send_btn = tk.Button(input_frame, text="Send ➤", font=("Segoe UI", 11, "bold"),
                             bg="#89b4fa", fg="#1e1e2e", relief="flat",
                             activebackground="#74c7ec", cursor="hand2",
                             command=self._send, padx=14, pady=6)
        send_btn.pack(side="right")

        # Hint
        tk.Label(self.root, text="Try: 'What is CodeAlpha?'  |  'How do I submit tasks?'  |  'What perks do I get?'",
                 font=("Segoe UI", 8), bg="#1e1e2e", fg="#6c7086").pack(pady=(0, 6))

    def _append(self, text: str, tag: str):
        self.chat_area.config(state="normal")
        self.chat_area.insert("end", text + "\n", tag)
        self.chat_area.see("end")
        self.chat_area.config(state="disabled")

    def _bot_say(self, text: str):
        self._append(f"🤖 Bot: {text}\n", "bot")

    def _send(self):
        user_input = self.entry.get().strip()
        if not user_input:
            return
        self.entry.delete(0, "end")
        self._append(f"🧑 You: {user_input}\n", "user")
        threading.Thread(target=self._respond, args=(user_input,), daemon=True).start()

    def _respond(self, user_input: str):
        answer = self.engine.get_answer(user_input)
        self.root.after(300, lambda: self._bot_say(answer))


# ─────────────────────────── Entry point ─────────────────────────────
if __name__ == "__main__":
    root = tk.Tk()
    app = ChatbotApp(root)
    root.mainloop()
