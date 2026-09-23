# -*- coding: utf-8 -*-
"""
Algorithm 1: Preference-Guided Diffusion Model Reverse Sampling
           (with Interval Rule Constraints)

File name: Algorithm1.py

Dependencies:
    pip install torch numpy
"""

import torch
import torch.nn as nn
from typing import List, Tuple, Callable, Optional


# =====================================================================
# CART Rule Definition
# =====================================================================

class CARTRule:
    """
    CART rule: contains path conditions, output target y_m and weight w_m.
    """

    def __init__(
        self,
        path_conditions: List[Tuple[int, float, str]],
        output: torch.Tensor,
        weight: float = 1.0,
    ):
        """
        Args:
            path_conditions: List of conditions, each as (feature_index, threshold, operator)
                             Operator is '<=' or '>'
            output:          Rule output y_m, scalar or feature-dimensional tensor
            weight:          Rule weight w_m
        """
        self.path_conditions = path_conditions
        self.output = output
        self.weight = weight

    def matches(self, phi: torch.Tensor) -> bool:
        """Check whether feature phi satisfies all path conditions of the rule (discrete check)."""
        for idx, threshold, op in self.path_conditions:
            val = phi[..., idx]
            if op == '<=':
                if not torch.all(val <= threshold):
                    return False
            elif op == '>':
                if not torch.all(val > threshold):
                    return False
            else:
                raise ValueError(f"Unknown operator: {op}")
        return True


# =====================================================================
# Diffusion Model Interface
# =====================================================================

class DiffusionModel:
    """
    Pretrained diffusion model interface.
    In real usage, you need to implement the score and ddim_step methods.
    """

    def score(self, x: torch.Tensor, t: int) -> torch.Tensor:
        """
        Compute diffusion model score ∇_{x_t} log p_θ(x_t).
        Usually derived from the noise prediction network ε_θ:
            score = -ε_θ / sqrt(1 - α_bar_t)
        This is an abstract method and must be implemented by subclasses.
        """
        raise NotImplementedError

    def ddim_step(
        self,
        x: torch.Tensor,
        s_total: torch.Tensor,
        t: int,
        c: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Execute one DDIM denoising update.

        Args:
            x:       Current noisy image x_t
            s_total: Total guidance direction (score + rule gradient)
            t:       Current time step
            c:       Text prompt condition (optional)

        Returns:
            x_{t-1}
        """
        raise NotImplementedError


# =====================================================================
# Main Algorithm: Preference-Guided Diffusion Reverse Sampling
# =====================================================================

def preference_guided_reverse_sampling(
    diffusion_model: DiffusionModel,
    cart_rules: List[CARTRule],
    feature_extractor: Callable[[torch.Tensor], torch.Tensor],
    text_prompt: Optional[torch.Tensor] = None,
    T: int = 1000,
    K: int = 5,
    lam: float = 1.0,
    image_shape: Tuple[int, ...] = (3, 256, 256),
    device: torch.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu'),
) -> torch.Tensor:
    """
    Preference-guided diffusion model reverse sampling (with interval rule constraints).

    Args:
        diffusion_model:  Pretrained diffusion model, must implement score and ddim_step
        cart_rules:       CART rule set R, each rule contains conditions, output y_m and weight w_m
        feature_extractor: Differentiable feature extractor f, input image x, output feature φ
        text_prompt:      Text prompt c (optional)
        T:                Number of sampling steps
        K:                Rule interval step size
        lam:              Constraint strength lambda
        image_shape:      Shape of initial noise
        device:           Computation device

    Returns:
        Generated image x0
    """
    # 1: Initialize pure noise x_T ~ N(0, I)
    x = torch.randn(image_shape, device=device)

    # 2: Initialize previous round rule gradient (for interpolation)
    g_prev = torch.zeros_like(x)

    # 3: Record the step of the last rule gradient computation
    t_last_update = T

    # 5: Reverse sampling loop
    for t in range(T, 0, -1):
        # Step 1: Compute diffusion model score (unconditional direction)
        s_score = diffusion_model.score(x, t)

        # Step 2: Determine whether to update rule gradient
        if (T - t) % K == 0 or t == T:
            # Extract artistic features of the current image
            phi_t = feature_extractor(x)

            # Identify the activated rule subset (if-then condition matching)
            active_rules = [rule for rule in cart_rules if rule.matches(phi_t)]

            if len(active_rules) == 0:
                # No active rules, fall back to unconditional generation
                g_current = torch.zeros_like(x)
            else:
                # Compute weighted aggregated rule distance (WDWA resolution strategy)
                # To compute gradient, we need to rebuild the computation graph on x
                x_in = x.detach().clone().requires_grad_(True)
                phi_in = feature_extractor(x_in)

                numerator = 0.0
                denominator = 0.0
                for rule in active_rules:
                    w_m = rule.weight
                    y_m = rule.output
                    # Make sure y_m matches phi_in dimension
                    if y_m.dim() == 0:
                        y_m = y_m.expand_as(phi_in)
                    diff = phi_in - y_m
                    numerator = numerator + w_m * torch.sum(diff ** 2)
                    denominator = denominator + w_m

                d = numerator / (denominator + 1e-8)

                # Compute rule gradient (back-propagate to input space)
                grad_d = torch.autograd.grad(d, x_in, create_graph=False)[0]
                g_current = -lam * grad_d.detach()

            g_prev = g_current
            t_last_update = t
        else:
            # Interval steps use linear interpolation to approximate the guidance direction
            i = t_last_update - t          # step distance since last update
            g_current = (K - (i % K)) / K * g_prev

        # Step 3: Combine total guidance direction (Formula 12)
        s_total = s_score + g_current

        # Step 4: Execute one denoising update (DDIM sampler)
        x = diffusion_model.ddim_step(x, s_total, t, text_prompt)

    # 38: Return x0
    return x


# =====================================================================
# Mock Example (for demonstrating the algorithm flow; replace with real models in production)
# =====================================================================

class DummyDiffusionModel(DiffusionModel):
    """Mock diffusion model, for demonstration only."""

    def score(self, x, t):
        # Mock score: return a tensor of the same shape as x
        return -0.1 * x + 0.01 * torch.randn_like(x)

    def ddim_step(self, x, s_total, t, c=None):
        # Mock DDIM update: move a small step in the s_total direction
        x_prev = x - 0.01 * s_total
        return x_prev


class DummyFeatureExtractor(nn.Module):
    """Mock differentiable feature extractor, mapping image to 128-D feature."""

    def __init__(self, in_channels=3, feature_dim=128):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, 8, 3, stride=2, padding=1)
        self.pool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(8, feature_dim)

    def forward(self, x):
        # x: (B, C, H, W) or (C, H, W)
        if x.dim() == 3:
            x = x.unsqueeze(0)
        h = self.conv(x)
        h = self.pool(h)
        h = h.view(h.size(0), -1)
        return self.fc(h)


if __name__ == "__main__":
    # Example: create mock components and run the algorithm
    device = torch.device('cpu')
    diffusion_model = DummyDiffusionModel()
    feature_extractor = DummyFeatureExtractor().to(device)

    # Create some CART rules
    rules = [
        CARTRule(path_conditions=[(0, 0.5, '<=')], output=torch.tensor(0.2), weight=1.0),
        CARTRule(path_conditions=[(0, 0.5, '>')], output=torch.tensor(0.8), weight=1.0),
    ]

    # Run reverse sampling
    x0 = preference_guided_reverse_sampling(
        diffusion_model=diffusion_model,
        cart_rules=rules,
        feature_extractor=feature_extractor,
        text_prompt=None,
        T=10,          # use small step count for demonstration
        K=5,
        lam=0.5,
        image_shape=(3, 64, 64),
        device=device,
    )
    print("Generated image shape:", x0.shape)