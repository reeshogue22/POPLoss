import torch

from pop import POPLoss


def _pref_grad(beta):
    """Gradient of the preference term alone w.r.t. the logits."""
    torch.manual_seed(0)
    x = torch.randn(64, 10, requires_grad=True)
    y = torch.randint(0, 10, (64,))
    total = POPLoss(beta=beta)(x, y)
    pref = total - torch.nn.functional.cross_entropy(x, y)
    (g,) = torch.autograd.grad(pref, x)
    return g, y


def test_preference_term_raises_target_logit():
    g, y = _pref_grad(beta=1.0)
    target_grad = g[torch.arange(len(y)), y]
    # gradient descent moves logits along -g: target logits must not be pushed down
    assert (target_grad <= 1e-7).all()
    assert (target_grad < 0).any()


def test_beta_zero_is_cross_entropy():
    torch.manual_seed(0)
    x, y = torch.randn(8, 5), torch.randint(0, 5, (8,))
    assert torch.allclose(POPLoss(beta=0.0)(x, y), torch.nn.functional.cross_entropy(x, y))


def test_finite_when_model_is_certain():
    x = torch.full((4, 6), -50.0)
    y = torch.tensor([0, 1, 2, 3])
    x[torch.arange(4), y] = 50.0
    x.requires_grad_(True)
    loss = POPLoss(beta=0.5)(x, y)
    loss.backward()
    assert torch.isfinite(loss) and torch.isfinite(x.grad).all()
