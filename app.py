"""
CodeAlpha AI Internship — Task 1: Language Translation Tool
Author: CodeAlpha Intern
Description: A GUI-based language translation tool using deep_translator (free, no API key needed).
"""

import tkinter as tk
from tkinter import ttk, messagebox
import threading
from deep_translator import GoogleTranslator

# ─────────────────────────── Language map ────────────────────────────
LANGUAGES = {
    "Auto Detect":  "auto",
    "Afrikaans":    "af",  "Albanian":     "sq",  "Arabic":       "ar",
    "Bengali":      "bn",  "Bulgarian":    "bg",  "Chinese (Simplified)": "zh-CN",
    "Chinese (Traditional)": "zh-TW",
    "Croatian":     "hr",  "Czech":        "cs",  "Danish":       "da",
    "Dutch":        "nl",  "English":      "en",  "Finnish":      "fi",
    "French":       "fr",  "German":       "de",  "Greek":        "el",
    "Gujarati":     "gu",  "Hebrew":       "iw",  "Hindi":        "hi",
    "Hungarian":    "hu",  "Indonesian":   "id",  "Italian":      "it",
    "Japanese":     "ja",  "Kannada":      "kn",  "Korean":       "ko",
    "Latvian":      "lv",  "Lithuanian":   "lt",  "Malay":        "ms",
    "Marathi":      "mr",  "Nepali":       "ne",  "Norwegian":    "no",
    "Persian":      "fa",  "Polish":       "pl",  "Portuguese":   "pt",
    "Punjabi":      "pa",  "Romanian":     "ro",  "Russian":      "ru",
    "Serbian":      "sr",  "Sinhala":      "si",  "Slovak":       "sk",
    "Spanish":      "es",  "Swahili":      "sw",  "Swedish":      "sv",
    "Tamil":        "ta",  "Telugu":       "te",  "Thai":         "th",
    "Turkish":      "tr",  "Ukrainian":    "uk",  "Urdu":         "ur",
    "Vietnamese":   "vi",
}

# ─────────────────────────── Main App ────────────────────────────────
class TranslationApp:
    def __init__(self, root: tk.Tk):
        self.root = root
        self.root.title("🌍 Language Translation Tool — CodeAlpha")
        self.root.geometry("900x620")
        self.root.configure(bg="#1e1e2e")
        self.root.resizable(True, True)
        self._build_ui()

    # ── UI Construction ──────────────────────────────────────────────
    def _build_ui(self):
        style = ttk.Style()
        style.theme_use("clam")
        style.configure("TCombobox", fieldbackground="#313244", background="#313244",
                        foreground="#cdd6f4", selectbackground="#89b4fa")
        style.configure("TButton", background="#89b4fa", foreground="#1e1e2e",
                        font=("Segoe UI", 10, "bold"), padding=6)
        style.map("TButton", background=[("active", "#74c7ec")])

        # Title bar
        title_frame = tk.Frame(self.root, bg="#181825", pady=10)
        title_frame.pack(fill="x")
        tk.Label(title_frame, text="🌍  Language Translation Tool",
                 font=("Segoe UI", 18, "bold"), bg="#181825", fg="#cba6f7").pack()
        tk.Label(title_frame, text="CodeAlpha AI Internship — Task 1",
                 font=("Segoe UI", 10), bg="#181825", fg="#6c7086").pack()

        # Language selector row
        lang_frame = tk.Frame(self.root, bg="#1e1e2e", pady=12)
        lang_frame.pack(fill="x", padx=30)

        lang_names = list(LANGUAGES.keys())

        tk.Label(lang_frame, text="Source Language", font=("Segoe UI", 10, "bold"),
                 bg="#1e1e2e", fg="#a6adc8").grid(row=0, column=0, padx=(0, 8))
        self.src_var = tk.StringVar(value="Auto Detect")
        self.src_combo = ttk.Combobox(lang_frame, textvariable=self.src_var,
                                       values=lang_names, width=22, state="readonly")
        self.src_combo.grid(row=0, column=1, padx=(0, 20))

        swap_btn = tk.Button(lang_frame, text="⇄ Swap", font=("Segoe UI", 10, "bold"),
                             bg="#313244", fg="#cdd6f4", relief="flat",
                             activebackground="#45475a", cursor="hand2",
                             command=self._swap_languages)
        swap_btn.grid(row=0, column=2, padx=10)

        tk.Label(lang_frame, text="Target Language", font=("Segoe UI", 10, "bold"),
                 bg="#1e1e2e", fg="#a6adc8").grid(row=0, column=3, padx=(20, 8))
        self.tgt_var = tk.StringVar(value="French")
        self.tgt_combo = ttk.Combobox(lang_frame, textvariable=self.tgt_var,
                                       values=[n for n in lang_names if n != "Auto Detect"],
                                       width=22, state="readonly")
        self.tgt_combo.grid(row=0, column=4)

        # Text areas
        text_frame = tk.Frame(self.root, bg="#1e1e2e")
        text_frame.pack(fill="both", expand=True, padx=30, pady=(0, 10))
        text_frame.columnconfigure(0, weight=1)
        text_frame.columnconfigure(2, weight=1)
        text_frame.rowconfigure(1, weight=1)

        tk.Label(text_frame, text="Enter Text", font=("Segoe UI", 10, "bold"),
                 bg="#1e1e2e", fg="#89b4fa").grid(row=0, column=0, sticky="w", pady=(0, 4))
        tk.Label(text_frame, text="Translation", font=("Segoe UI", 10, "bold"),
                 bg="#1e1e2e", fg="#a6e3a1").grid(row=0, column=2, sticky="w", pady=(0, 4))

        self.input_text = tk.Text(text_frame, font=("Segoe UI", 12), wrap="word",
                                   bg="#313244", fg="#cdd6f4", insertbackground="#cdd6f4",
                                   relief="flat", bd=0, padx=10, pady=10)
        self.input_text.grid(row=1, column=0, sticky="nsew")

        tk.Frame(text_frame, bg="#6c7086", width=2).grid(row=1, column=1, sticky="ns", padx=12)

        self.output_text = tk.Text(text_frame, font=("Segoe UI", 12), wrap="word",
                                    bg="#313244", fg="#a6e3a1", insertbackground="#cdd6f4",
                                    relief="flat", bd=0, padx=10, pady=10, state="disabled")
        self.output_text.grid(row=1, column=2, sticky="nsew")

        # Character counter
        self.char_var = tk.StringVar(value="0 / 5000 characters")
        tk.Label(text_frame, textvariable=self.char_var, font=("Segoe UI", 8),
                 bg="#1e1e2e", fg="#6c7086").grid(row=2, column=0, sticky="w")
        self.input_text.bind("<KeyRelease>", self._update_char_count)

        # Bottom buttons
        btn_frame = tk.Frame(self.root, bg="#1e1e2e", pady=10)
        btn_frame.pack()

        ttk.Button(btn_frame, text="🌐  Translate", command=self._translate_thread).pack(side="left", padx=8)
        ttk.Button(btn_frame, text="📋  Copy Result", command=self._copy_result).pack(side="left", padx=8)
        ttk.Button(btn_frame, text="🗑️  Clear All", command=self._clear_all).pack(side="left", padx=8)

        # Status bar
        self.status_var = tk.StringVar(value="Ready")
        tk.Label(self.root, textvariable=self.status_var, font=("Segoe UI", 9),
                 bg="#181825", fg="#6c7086", anchor="w").pack(fill="x", padx=10, pady=(0, 4))

    # ── Helpers ──────────────────────────────────────────────────────
    def _update_char_count(self, _event=None):
        count = len(self.input_text.get("1.0", "end-1c"))
        self.char_var.set(f"{count} / 5000 characters")

    def _swap_languages(self):
        src, tgt = self.src_var.get(), self.tgt_var.get()
        if src != "Auto Detect":
            self.src_var.set(tgt)
            self.tgt_var.set(src)

    def _clear_all(self):
        self.input_text.delete("1.0", "end")
        self._set_output("")
        self.char_var.set("0 / 5000 characters")
        self.status_var.set("Cleared.")

    def _copy_result(self):
        result = self.output_text.get("1.0", "end-1c").strip()
        if result:
            self.root.clipboard_clear()
            self.root.clipboard_append(result)
            self.status_var.set("✅ Translation copied to clipboard!")
        else:
            self.status_var.set("Nothing to copy yet.")

    def _set_output(self, text: str):
        self.output_text.config(state="normal")
        self.output_text.delete("1.0", "end")
        self.output_text.insert("1.0", text)
        self.output_text.config(state="disabled")

    def _translate_thread(self):
        threading.Thread(target=self._do_translate, daemon=True).start()

    def _do_translate(self):
        text = self.input_text.get("1.0", "end-1c").strip()
        if not text:
            self.status_var.set("⚠️  Please enter text to translate.")
            return
        if len(text) > 5000:
            messagebox.showwarning("Too Long", "Please keep input under 5000 characters.")
            return

        src_code = LANGUAGES[self.src_var.get()]
        tgt_code = LANGUAGES[self.tgt_var.get()]

        self.status_var.set("⏳ Translating…")
        self._set_output("Translating…")

        try:
            translated = GoogleTranslator(source=src_code, target=tgt_code).translate(text)
            self._set_output(translated)
            self.status_var.set(f"✅ Done!  {self.src_var.get()} → {self.tgt_var.get()}")
        except Exception as exc:
            self._set_output("")
            self.status_var.set(f"❌ Error: {exc}")
            messagebox.showerror("Translation Error", str(exc))


# ─────────────────────────── Entry point ─────────────────────────────
if __name__ == "__main__":
    root = tk.Tk()
    app = TranslationApp(root)
    root.mainloop()
