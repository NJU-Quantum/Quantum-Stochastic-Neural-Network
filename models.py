# models.py
import torch
import torch.nn as nn
import re
import string
import qsw # liouvillian, lindblad_rhs, liouvillian_mv, evolve_expm, evolve_vec_rk4, evolve_from_operators

try:
    import nltk
    from nltk import pos_tag
    from nltk.corpus import stopwords, wordnet
    from nltk.stem import WordNetLemmatizer
    _NLTK_AVAILABLE = True
except Exception:
    _NLTK_AVAILABLE = False

# 默认使用自适应演化器：小规模用 expm，大规模自动切换 Krylov
# 如需固定算法，可在脚本中临时覆盖 models.evolve。
evolve = qsw.evolve_auto


class QSNNFunction(nn.Module):
    def __init__(self, N_in=10, T=1.0, init_scale=0.05, device="cuda"):
        super().__init__()
        self.N_in = N_in
        self.N = N_in + 1  # last neuron is output
        self.T = T
        self.device = device

        self.H_raw = nn.Parameter(init_scale * torch.randn(self.N, self.N, device=device, dtype=torch.float32))

    def encode(self, x):
        """
        Eq.(7) with n=1: |psi> ∝ Σ_{i=0}^{N_in-1} x^i |i>
        """
        x = x.to(self.device)
        x = x.reshape(-1)  # (B,)
        B = x.shape[0]

        exponents = torch.arange(self.N_in, device=self.device, dtype=x.dtype)
        powers = x.unsqueeze(-1) ** exponents.unsqueeze(0)  # (B, N_in)

        psi = torch.zeros((B, self.N, 1), device=self.device, dtype=torch.complex64)
        psi[:, :self.N_in, 0] = powers.to(torch.complex64)
        psi = psi / torch.linalg.norm(psi, dim=1, keepdim=True).clamp_min(1e-12)
        rho = psi @ psi.mH  # (B, N, N)
        return rho

    def forward(self, x):
        Hf = self.H_raw.to(torch.complex64)
        H = 0.5 * (Hf + Hf.mH)  # Hermitian

        rho_in = self.encode(x)
        rho_out = evolve(rho_in, H, [], self.T)

        yhat = rho_out[:, self.N-1, self.N-1].real.clamp(0.0, 1.0)
        if yhat.numel() == 1:
            return yhat[0], rho_out[0]
        return yhat, rho_out
    

def basis(N, i, device):
    v = torch.zeros((N,1), device=device, dtype=torch.complex64)
    v[i,0] = 1
    return v

class QSNN2D(nn.Module):
    def __init__(self, N_in=12, T_u=1.0, T_d=1.0, init_h=0.1, init_g=0.1, device="cuda", stage2_steps=20):
        super().__init__()
        self.N_in = N_in
        self.N = N_in + 2
        self.T_u, self.T_d = T_u, T_d
        self.device = device
        self.stage2_steps = stage2_steps

        # input-layer Hamiltonian params
        self.Hu_raw = nn.Parameter(init_h * torch.randn(N_in, N_in, device=device, dtype=torch.float32))
        # gammas: 2 outputs x N_in inputs
        self.gamma = nn.Parameter(init_g * torch.randn(2, N_in, device=device, dtype=torch.float32))

    def encode(self, x, y):
        # Eq.(7) with n=2, choose K=N_in/2
        K = self.N_in // 2
        assert 2*K == self.N_in

        x = x.to(self.device).reshape(-1)
        y = y.to(self.device).reshape(-1)
        B = x.shape[0]

        exponents = torch.arange(K, device=self.device, dtype=x.dtype)
        px = x.unsqueeze(-1) ** exponents.unsqueeze(0)
        py = y.unsqueeze(-1) ** exponents.unsqueeze(0)

        psi = torch.zeros((B, self.N, 1), device=self.device, dtype=torch.complex64)
        psi[:, :K, 0] = px.to(torch.complex64)
        psi[:, K:2*K, 0] = py.to(torch.complex64)
        psi = psi / torch.linalg.norm(psi, dim=1, keepdim=True).clamp_min(1e-12)
        return psi @ psi.mH

    def forward(self, xy):
        if isinstance(xy, (list, tuple)):
            x, y = xy[0], xy[1]
        else:
            x, y = xy[..., 0], xy[..., 1]
        N, N_in = self.N, self.N_in

        # Stage 1: unitary on input block
        Hu = self.Hu_raw.to(torch.complex64)
        Hu = 0.5 * (Hu + Hu.mH)
        H = torch.zeros((N,N), device=self.device, dtype=torch.complex64)
        H[:N_in,:N_in] = Hu

        rho0 = self.encode(x, y)
        rho_u = evolve(rho0, H, [], self.T_u)

        # Stage 2: dissipative input -> output（结构化演化器，适配大 N）
        rho_out = qsw.evolve_qsnn2d_stage2_structured(
            rho_u,
            H,
            self.gamma.to(torch.complex64),
            self.T_d,
            N_in,
            steps=self.stage2_steps,
        )

        out0, out1 = N_in, N_in + 1

        p0 = rho_out[:, out0, out0].real
        p1 = rho_out[:, out1, out1].real
        probs = torch.stack([p0, p1], dim=-1).clamp(1e-6, 1.0)
        probs = probs / probs.sum(dim=-1, keepdim=True)  # normalize
        if probs.shape[0] == 1:
            return probs[0], rho_out[0]
        return probs, rho_out


class QSNNText(nn.Module):
    """
    Text-oriented QSNN model that follows the legacy poem pipeline:
    1) preprocess sentence into normalized tokens
    2) quantum sentence encoding by sequential Lindblad evolution
    3) readout on word neurons (N-3)
    4) optimized vectorized binary classifier head (Linear -> Softmax)

    Compared with the legacy script, the classifier is implemented as a single
    vectorized trainable head while preserving the same decision form.
    """

    def __init__(
        self,
        vocab=None,
        t_input=20.0,
        gamma=1.0,
        device="cuda",
        use_nltk=True,
    ):
        super().__init__()
        self.device = device
        self.t_input = float(t_input)
        self.gamma = float(gamma)
        self.use_nltk = bool(use_nltk and _NLTK_AVAILABLE)

        if vocab is None:
            vocab = ["gold", "sun", "dawn", "love", "stay", "day", "go", "noth"]
        self.vocab = list(vocab)
        self.word_to_idx = {w: i for i, w in enumerate(self.vocab)}

        # Legacy convention: N = (N_words) + 3
        self.N_words = len(self.vocab)
        self.N = self.N_words + 3

        # Optimized binary classifier head (equivalent to two linear branches).
        self.classifier = nn.Linear(self.N_words, 2, bias=True, device=device, dtype=torch.float32)
        nn.init.uniform_(self.classifier.weight, a=-1.0, b=1.0)
        nn.init.uniform_(self.classifier.bias, a=-1.0, b=1.0)

        # Cache for repeated word evolution operators keyed by (word_idx, delta_t).
        self._word_l_cache = {}

        if self.use_nltk:
            try:
                self._wnl = WordNetLemmatizer()
                self._stopwords = set(stopwords.words("english"))
            except Exception:
                self.use_nltk = False
                self._wnl = None
                self._stopwords = set()
        else:
            self._wnl = None
            self._stopwords = set()

    def _basis(self, i: int) -> torch.Tensor:
        v = torch.zeros((self.N, 1), device=self.device, dtype=torch.complex64)
        v[i, 0] = 1
        return v

    def _wordnet_pos(self, tag: str):
        if not self.use_nltk:
            return None
        if tag.startswith("J"):
            return wordnet.ADJ
        if tag.startswith("V"):
            return wordnet.VERB
        if tag.startswith("N"):
            return wordnet.NOUN
        if tag.startswith("R"):
            return wordnet.ADV
        return None

    def _simple_tokenize(self, text: str):
        stem_map = {
            "goes": "go",
            "going": "go",
            "went": "go",
            "gone": "go",
            "loves": "love",
            "lovely": "love",
            "stays": "stay",
            "staying": "stay",
            "nothing": "noth",
        }
        toks = re.findall(r"[a-zA-Z']+", text.lower())
        return [stem_map.get(t, t) for t in toks]

    def preprocess_sentence(self, text: str):
        if not self.use_nltk:
            toks = self._simple_tokenize(text)
            return [t for t in toks if t in self.word_to_idx]

        lower = text.lower()
        remove = str.maketrans("", "", string.punctuation)
        no_punc = lower.translate(remove)

        try:
            tokens = nltk.word_tokenize(no_punc)
            tagged = pos_tag(tokens)
        except Exception:
            toks = self._simple_tokenize(no_punc)
            return [t for t in toks if t in self.word_to_idx]

        lemmas = []
        for tok, tag in tagged:
            pos = self._wordnet_pos(tag) or wordnet.NOUN
            lemmas.append(self._wnl.lemmatize(tok, pos=pos))

        filtered = [w for w in lemmas if w not in self._stopwords]
        stemmer = nltk.stem.SnowballStemmer("english")
        stems = [stemmer.stem(w) for w in filtered]
        return [s for s in stems if s in self.word_to_idx]

    def _word_lindblad(self, word_idx: int):
        # Legacy C component used by words is |line><0|, with line in [1, N_words].
        # word_idx is 0-based, line is 1-based in the Hilbert basis.
        line = word_idx + 1
        key = line
        if key in self._word_l_cache:
            return self._word_l_cache[key]

        ket_line = self._basis(line)
        ket_0 = self._basis(0)
        L = (self.gamma * (ket_line @ ket_0.mH)).to(torch.complex64)
        self._word_l_cache[key] = L
        return L

    def sentence_feature(self, sentence: str) -> torch.Tensor:
        tokens = self.preprocess_sentence(sentence)
        if len(tokens) == 0:
            return torch.zeros((self.N_words,), device=self.device, dtype=torch.float32)

        # Legacy timing rule: delta_t = int(t_input / words_in_sent)
        delta_t = max(1, int(self.t_input / max(len(tokens), 1)))
        # Legacy controller used timeline [0, (delta_t-1)/2].
        t_word = max((delta_t - 1) / 2.0, 1e-6)

        rho = self._basis(0) @ self._basis(0).mH
        H = torch.zeros((self.N, self.N), device=self.device, dtype=torch.complex64)

        for tok in tokens:
            idx = self.word_to_idx.get(tok)
            if idx is None:
                continue
            L = self._word_lindblad(idx)
            rho = qsw.evolve_auto(rho, H, [L], t_word)

        # Legacy readout: measure |i><i| for i=1..N_words.
        diag = torch.real(torch.diagonal(rho, dim1=-2, dim2=-1))
        feat = diag[1 : 1 + self.N_words].to(torch.float32)
        return feat

    def encode_sentences(self, sentences):
        feats = [self.sentence_feature(s) for s in sentences]
        return torch.stack(feats, dim=0)

    def forward(self, sentences, labels=None):
        """
        sentences: list[str]
        labels: optional tensor/list with 0/1 labels

        returns dict with:
        - features: (B, N_words)
        - logits: (B, 2)
        - probs: (B, 2)
        - legacy_cost: scalar if labels is provided
        """
        if isinstance(sentences, str):
            sentences = [sentences]

        features = self.encode_sentences(sentences)
        logits = self.classifier(features)
        probs = torch.softmax(logits, dim=-1)

        out = {"features": features, "logits": logits, "probs": probs}

        if labels is not None:
            if not torch.is_tensor(labels):
                labels = torch.tensor(labels, device=self.device, dtype=torch.long)
            else:
                labels = labels.to(self.device, dtype=torch.long)

            p_correct = probs.gather(1, labels.view(-1, 1)).squeeze(1)
            legacy_cost = 1.0 - p_correct.mean()
            out["legacy_cost"] = legacy_cost

        return out