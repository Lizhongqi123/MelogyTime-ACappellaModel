class Config:
    # model structure（约 5 亿参数）
    vocab_size = 151936
    dim = 512
    n_layers = 12
    n_heads = 12

    # MLA
    qk_nope_head_dim = 64
    qk_rope_head_dim = 32
    kv_lora_rank = 128

    # MoE
    n_routed_experts = 32
    n_shared_experts = 1
    top_k = 4
    expert_hidden = 1024

    # inference
    max_seq_len = 128
    max_new_tokens = 256
    temperature = 0.5
    top_p = 0.7
    device = "cpu"

    # training
    batch_size = 10
    grad_accum = 4
    lr = 1e-4
    epochs = 5
    data_path = "dialogue.jsonl"
    ckpt_path = "ckpt.pt"
    tokenizer_name = "F:/岁月弦歌/tokenizer"