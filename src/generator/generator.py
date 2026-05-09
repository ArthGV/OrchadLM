import torch
import torch.nn.functional as F

class Generator:

    def __init__(self, device, model):
        self.device = device
        self.model = model

    def generate(
        self,
        context_lenght: int,
        tokenized_prompt: list[int],
        max_tokens: int = 200,
        temperature: float = 1.0,
        beam_width: int = 1,
        beam_depth: int = 1,
    ) -> list[int]:
        """
        Generate model answer.

        Args:
            beam_width: 1       → greedy / temperature sampling
                        >1      → lookahead beam search at each token step
            beam_depth: How many steps ahead to score each candidate.
                        Only used when beam_width > 1.
        """
        self.model.eval()
        context = tokenized_prompt[-context_lenght:]
        out = []

        with torch.no_grad():
            for _ in range(max_tokens):
                if beam_width > 1:
                    next_token = self.beam_search(context, beam_width, beam_depth, temperature)
                else:
                    next_token = self.generate_next_token(context, temperature)

                out.append(next_token)
                context = context[1:] + [next_token]

        return out
    
    def generate_next_token(self, context: list[int], temperature: float = 1.0) -> int:
        """
        Generate next token.
        """
        x = torch.tensor([context], dtype=torch.long, device=self.device)
        logits = self.model(x)
        logits = logits[0, -1] / (temperature + 1e-6)
        probs  = F.softmax(logits, dim=-1)
        next_token = torch.multinomial(probs, 1).item()
        return next_token
    
    def beam_search(
        self,
        context: list[int],
        beam_width: int = 5,
        depth: int = 3,
        temperature: float = 1.0,
    ) -> int:
        """
        Pick the next single token using a fixed-depth beam search lookahead.

        For the current context, expands `beam_width` candidate next-tokens,
        then scores each by running a `depth`-step beam search forward.
        Returns the token id whose lookahead branch had the highest
        cumulative log-prob.

        Args:
            context:    Current context window (length <= CONTEXT_LEN).
            beam_width: Number of candidates to explore at each expansion step.
            depth:      How many steps ahead to evaluate each candidate.
            temperature: Scales logits before computing log-probs.

        Returns:
            The single best next token id.
        """
        # Each beam: (cumulative_log_prob, context_window, root_token)
        # root_token tracks which first-step candidate this beam descended from
        x = torch.tensor([context], dtype=torch.long, device=self.device)
        logits = self.model(x)
        logits = logits[0, -1] / (temperature + 1e-6)
        log_probs = F.log_softmax(logits, dim=-1)

        topk_log_probs, topk_tokens = torch.topk(log_probs, beam_width)

        beams: list[tuple[float, list[int], int]] = [
            (lp.item(), context[1:] + [tok.item()], tok.item())
            for lp, tok in zip(topk_log_probs, topk_tokens)
        ]

        # Lookahead: expand `depth - 1` more steps
        for _ in range(depth - 1):
            candidates: list[tuple[float, list[int], int]] = []

            for cum_lp, ctx, root_token in beams:
                x = torch.tensor([ctx], dtype=torch.long, device=self.device)
                logits = self.model(x)
                logits = logits[0, -1] / (temperature + 1e-6)
                lps = F.log_softmax(logits, dim=-1)

                topk_lps, topk_toks = torch.topk(lps, beam_width)

                for lp, tok in zip(topk_lps.tolist(), topk_toks.tolist()):
                    candidates.append((cum_lp + lp, ctx[1:] + [tok], root_token))

            beams = sorted(candidates, key=lambda t: t[0], reverse=True)[:beam_width]

        best = beams[0]
        return best[2]  # root_token
        
