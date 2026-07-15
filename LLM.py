import torch
import torch.nn as nn
import torch.nn.functional as F

torch.manual_seed(42)

POEMS = """
春眠不觉晓，处处闻啼鸟。
夜来风雨声，花落知多少。
床前明月光，疑是地上霜。
举头望明月，低头思故乡。
白日依山尽，黄河入海流。
欲穷千里目，更上一层楼。
两个黄鹂鸣翠柳，一行白鹭上青天。
窗含西岭千秋雪，门泊东吴万里船。
千山鸟飞绝，万径人踪灭。
孤舟蓑笠翁，独钓寒江雪。
故国三千里，深宫二十年。
一声何满子，双泪落君前。
向晚意不适，驱车登古原。
夕阳无限好，只是近黄昏。
空山新雨后，天气晚来秋。
明月松间照，清泉石上流。
独在异乡为异客，每逢佳节倍思亲。
遥知兄弟登高处，遍插茱萸少一人。
葡萄美酒夜光杯，欲饮琵琶马上催。
醉卧沙场君莫笑，古来征战几人回。
秦时明月汉时关，万里长征人未还。
但使龙城飞将在，不教胡马度阴山。
寒雨连江夜入吴，平明送客楚山孤。
洛阳亲友如相问，一片冰心在玉壶。
渭城朝雨浥轻尘，客舍青青柳色新。
劝君更尽一杯酒，西出阳关无故人。
清明时节雨纷纷，路上行人欲断魂。
借问酒家何处有，牧童遥指杏花村。
远上寒山石径斜，白云生处有人家。
停车坐爱枫林晚，霜叶红于二月花。
"""

BLOCK_SIZE = 16
BATCH_SIZE = 16
N_EMBD = 128
N_HEAD = 4
N_LAYER = 6
TRAIN_STEPS = 6000
LR = 1e-3

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

chars = sorted(set(POEMS))
char_to_idx = {ch: i for i, ch in enumerate(chars)}
idx_to_char = {i: ch for i, ch in enumerate(chars)}

def encode(text):
    return [char_to_idx[c] for c in text]

def decode(indices):
    return "".join(idx_to_char[i] for i in indices)

data = torch.tensor(encode(POEMS), dtype=torch.long, device=device)

def sample_batch():
    starts = torch.randint(len(data) - BLOCK_SIZE, (BATCH_SIZE,), device=device)
    x = torch.stack([data[i:i + BLOCK_SIZE] for i in starts])
    y = torch.stack([data[i + 1:i + BLOCK_SIZE + 1] for i in starts])
    return x, y

class CausalSelfAttention(nn.Module):
    def __init__(self, n_embd, n_head, block_size):
        super().__init__()
        assert n_embd % n_head == 0
        self.n_head = n_head
        self.head_dim = n_embd // n_head
        self.qkv = nn.Linear(n_embd, 3 * n_embd)
        self.proj = nn.Linear(n_embd, n_embd)
        self.register_buffer(
            "causal_mask",
            torch.tril(torch.ones(block_size, block_size)).view(1, 1, block_size, block_size),
        )

    def forward(self, x):
        B, T, C = x.shape
        qkv = self.qkv(x).reshape(B, T, 3, self.n_head, self.head_dim)
        q, k, v = qkv.unbind(2)
        q = q.transpose(1, 2)
        k = k.transpose(1, 2)
        v = v.transpose(1, 2)

        scores = (q @ k.transpose(-2, -1)) / (self.head_dim ** 0.5)
        scores = scores.masked_fill(self.causal_mask[:, :, :T, :T] == 0, float("-inf"))
        attn = F.softmax(scores, dim=-1)
        out = (attn @ v).transpose(1, 2).reshape(B, T, C)
        return self.proj(out)

class FeedForward(nn.Module):
    def __init__(self, n_embd):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(n_embd, 4 * n_embd),
            nn.GELU(),
            nn.Linear(4 * n_embd, n_embd),
        )

    def forward(self, x):
        return self.net(x)

class TransformerBlock(nn.Module):
    def __init__(self, n_embd, n_head, block_size):
        super().__init__()
        self.ln1 = nn.LayerNorm(n_embd)
        self.attn = CausalSelfAttention(n_embd, n_head, block_size)
        self.ln2 = nn.LayerNorm(n_embd)
        self.ff = FeedForward(n_embd)

    def forward(self, x):
        x = x + self.attn(self.ln1(x))
        x = x + self.ff(self.ln2(x))
        return x

class MiniGPT(nn.Module):
    def __init__(self, vocab_size):
        super().__init__()
        self.block_size = BLOCK_SIZE
        self.tok_emb = nn.Embedding(vocab_size, N_EMBD)
        self.pos_emb = nn.Embedding(BLOCK_SIZE, N_EMBD)
        self.blocks = nn.Sequential(
            *[TransformerBlock(N_EMBD, N_HEAD, BLOCK_SIZE) for _ in range(N_LAYER)]
        )
        self.ln_f = nn.LayerNorm(N_EMBD)
        self.head = nn.Linear(N_EMBD, vocab_size)

    def forward(self, idx, targets=None):
        B, T = idx.shape
        pos = torch.arange(T, device=idx.device)
        x = self.tok_emb(idx) + self.pos_emb(pos)
        x = self.blocks(x)
        x = self.ln_f(x)
        logits = self.head(x)

        loss = None
        if targets is not None:
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1))
        return logits, loss

    @torch.no_grad()
    def generate(self, idx, max_new_tokens, temperature=0.8, top_k=5, stop_ids=None):
        for _ in range(max_new_tokens):
            logits, _ = self(idx[:, -self.block_size:])
            logits = logits[:, -1, :] / temperature

            if top_k is not None:
                kth = torch.topk(logits, min(top_k, logits.size(-1))).values[:, -1:]
                logits[logits < kth] = float("-inf")

            probs = F.softmax(logits, dim=-1)
            next_id = torch.multinomial(probs, num_samples=1)
            idx = torch.cat([idx, next_id], dim=1)

            yield next_id.item()

            if stop_ids and next_id.item() in stop_ids:
                break

MODEL_PATH = "minigpt.pt"

def train():
    model = MiniGPT(len(chars)).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=LR)

    model.train()
    for step in range(TRAIN_STEPS):
        x, y = sample_batch()
        _, loss = model(x, y)
        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()

        if step % 500 == 0:
            print(f"step {step:4d}  loss {loss.item():.4f}")

    torch.save(model.state_dict(), MODEL_PATH)
    print(f"模型已保存到 {MODEL_PATH}")
    return model


def load_model():
    model = MiniGPT(len(chars)).to(device)
    model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
    model.eval()
    return model

def chat(model):
    stop_ids = {char_to_idx["，"], char_to_idx["。"]}
    print("模型已就绪，输入文字与它对话（输入q退出）")

    while True:
        user_input = input("\n你：").strip()
        if not user_input or user_input.lower() in ("q", "quit", "exit"):
            break

        try:
            context_ids = [char_to_idx[c] for c in user_input]
        except KeyError as e:
            print(f"字符 {e} 不在词表中，请只用诗歌中的字。")
            continue

        context = torch.tensor([context_ids], dtype=torch.long, device=device)
        if context.size(1) > BLOCK_SIZE:
            context = context[:, -BLOCK_SIZE:]

        print("LLM：", user_input, sep="", end="", flush=True)
        for next_id in model.generate(
            context, max_new_tokens=30, temperature=0.1, top_k=5, stop_ids=stop_ids
        ):
            print(idx_to_char[next_id], end="", flush=True)

def main():
    import os
    if os.path.exists(MODEL_PATH):
        model = load_model()
    else:
        model = train()
    model.eval()
    chat(model)

if __name__ == "__main__":
    main()