import sys, json, random
sys.path.insert(0, '.')

class RWKV_TOKENIZER():   # verbatim from rwkv_v7_demo_fast.py
    def __init__(self, file_name):
        self.idx2token = {}
        sorted = []
        lines = open(file_name, "r", encoding="utf-8").readlines()
        for l in lines:
            idx = int(l[:l.index(' ')])
            x = eval(l[l.index(' '):l.rindex(' ')])
            x = x.encode("utf-8") if isinstance(x, str) else x
            assert isinstance(x, bytes)
            assert len(x) == int(l[l.rindex(' '):])
            sorted += [x]
            self.idx2token[idx] = x
        self.token2idx = {}
        for k, v in self.idx2token.items():
            self.token2idx[v] = int(k)
        self.table = [[[] for j in range(256)] for i in range(256)]
        self.good = [set() for i in range(256)]
        self.wlen = [0 for i in range(256)]
        for i in reversed(range(len(sorted))):
            s = sorted[i]
            if len(s) >= 2:
                s0 = int(s[0]); s1 = int(s[1])
                self.table[s0][s1] += [s]
                self.wlen[s0] = max(self.wlen[s0], len(s))
                self.good[s0].add(s1)
    def encodeBytes(self, src):
        src_len = len(src); tokens = []; i = 0
        while i < src_len:
            s = src[i : i + 1]
            if i < src_len - 1:
                s1 = int(src[i + 1]); s0 = int(src[i])
                if s1 in self.good[s0]:
                    sss = src[i : i + self.wlen[s0]]
                    try: s = next(filter(sss.startswith, self.table[s0][s1]))
                    except: pass
            tokens.append(self.token2idx[s]); i += len(s)
        return tokens
    def encode(self, src): return self.encodeBytes(src.encode("utf-8"))
    def decode(self, tokens): return b''.join(map(lambda i: self.idx2token[i], tokens)).decode('utf-8')

tk = RWKV_TOKENIZER("rwkv_vocab_v20230424.txt")

# real prompts from the repo + adversarial unicode / whitespace / code
cases = [
  "User: simulate SpaceX mars landing using python\n\nAssistant: <think",
  "\nIn a shocking finding, scientist discovered a herd of dragons living in a remote, previously unexplored valley, in Tibet. Even more surprising to the researchers was the fact that the dragons spoke perfect Chinese.",
  "One thing that should be learned from the bitter lesson is the great power of general purpose methods",
  open("RWKV7-G1x-templates.txt", encoding="utf-8").read(),
  "def f(x):\n    return x**2 + 1  # comment\n\tTABBED",
  "日本語のテキストです。中文测试。한국어 텍스트. Ελληνικά. Русский текст.",
  "emoji: 🚀🐍✨ family: 👨‍👩‍👧‍👦 flags: 🇻🇳🇺🇸",
  "        leading spaces and    multiple     internal    runs\n\n\n\nnewlines",
  "".join(chr(i) for i in range(1, 512)),
  "a" * 300 + "ééé" + "-" * 200,
  '{"json": [1,2,3], "nested": {"k": null, "b": true}}',
  "Mixed nbsp​zwsp　ideographic",
]
random.seed(3)
alphabet = "abcXYZ 0123\n\t.,!?'\"\\/{}[]()<>|-_=+*&^%$#@~`日本語🚀é"
for _ in range(300):
    cases.append("".join(random.choice(alphabet) for _ in range(random.randrange(1, 120))))

out = []
for c in cases:
    out.append({"text": c, "ids": tk.encode(c)})
json.dump(out, open(sys.argv[1], "w"))
print(f"python tokenizer: {len(cases)} cases, {sum(len(o['ids']) for o in out)} tokens total")
