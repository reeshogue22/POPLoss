import torch
import torch.nn.functional as F


class POPLoss(torch.nn.Module):
    """Cross-entropy plus an ORPO-style odds-ratio term at every position.

    chosen = the data token, rejected = a token sampled from the model itself (temperature
    `temperature`, no gradient). The extra term is -beta * log sigmoid(log odds(chosen) -
    log odds(rejected)), so it raises the data token's odds relative to the model's own sample.

    x: logits (N, vocab), y: targets (N,).
    """

    def __init__(self, beta=0.1, temperature=0.7):
        super().__init__()
        self.beta = beta
        self.temperature = temperature

    def forward(self, x, y):
        ce = F.cross_entropy(x, y)

        with torch.no_grad():
            rejected = torch.multinomial(torch.softmax(x / self.temperature, dim=-1), 1).squeeze(-1)

        log_probs = F.log_softmax(x, dim=-1)
        chosen_lp = torch.gather(log_probs, -1, y.unsqueeze(-1)).squeeze(-1)
        rejected_lp = torch.gather(log_probs, -1, rejected.unsqueeze(-1)).squeeze(-1)

        def log_odds(lp):  # log(p / (1 - p)), clamped so p -> 1 stays finite
            lp = lp.clamp(max=-1e-6)
            return lp - torch.log1p(-torch.exp(lp))

        ratio = log_odds(chosen_lp) - log_odds(rejected_lp)
        return ce - self.beta * F.logsigmoid(ratio).mean()
