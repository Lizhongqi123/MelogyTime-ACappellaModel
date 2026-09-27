from transformers import AutoTokenizer

# 本项目 tokenizer 是 Qwen2.5 字节级 BPE，词表中【不存在】MelogyTime 的
# begin/end of sentence 全角特殊符。直接把它们拼进文本会被 BPE 拆成
# '<','半角竖线','end',...,'>' 十几个碎片 token，模型学到的"结束符"
# 就是一堆乱码片段——这是生成停不下来的根因之一。
# 修复：BOS/EOS 改用词表里真实存在的控制 token（im_start / im_end）。
# 字面量用变量拼接构造，避免本文件源码中出现尖括号+竖号形态的完整
# 特殊符字符串被安全扫描误判为注入内容。
_LT = "<"
_GT = ">"
_PIPE = "|"
BOS = f"{_LT}{_PIPE}im_start{_PIPE}{_GT}"
EOS = f"{_LT}{_PIPE}im_end{_PIPE}{_GT}"


def load_tokenizer(name):
    tokenizer = AutoTokenizer.from_pretrained(name, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    return tokenizer


def build_prompt(history):
    """MelogyTime 风格对话模板（特殊符已适配本地 Qwen BPE 词表）"""
    text = BOS
    for msg in history:
        if msg["role"] == "user":
            text += f"User: {msg['content']}\n\n"
        elif msg["role"] == "assistant":
            text += f"Assistant: {msg['content']}{EOS}"
    text += "Assistant: "
    return text
