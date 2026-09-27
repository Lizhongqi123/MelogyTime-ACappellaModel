import tkinter as tk
from tkinter import scrolledtext
import threading
from config import Config
from inference.generate import Generator

cfg = Config()
gen = Generator(cfg)
history = []


def send():
    user = entry.get().strip()
    if not user:
        return
    entry.delete(0, tk.END)
    chat.insert(tk.END, f"你: {user}\n")
    history.append({"role": "user", "content": user})

    def worker():
        reply = gen.chat(history)
        history.append({"role": "assistant", "content": reply})
        chat.insert(tk.END, f"模型: {reply}\n\n")
        chat.see(tk.END)

    threading.Thread(target=worker, daemon=True).start()


root = tk.Tk()
root.title("ACappella 500M")
root.geometry("750x550")

chat = scrolledtext.ScrolledText(root, wrap=tk.WORD, font=("微软雅黑", 11))
chat.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)

entry = tk.Entry(root, font=("微软雅黑", 11))
entry.pack(fill=tk.X, padx=10, pady=(0, 5))
entry.bind("<Return>", lambda e: send())

tk.Button(root, text="发送", command=send).pack(pady=(0, 10))
root.mainloop()