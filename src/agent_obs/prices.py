"""Sample per-1M-token prices in USD (input, output).

Placeholder figures for demonstration — verify against the provider's
current pricing page before relying on them.
"""

PRICES = {
    "gpt-4o": (2.50, 10.00),
    "gpt-4o-mini": (0.15, 0.60),
    "claude-sonnet-4": (3.00, 15.00),
    "claude-haiku-4": (0.80, 4.00),
    "gemini-2.0-flash": (0.10, 0.40),
}


def estimate_cost(model, input_tokens, output_tokens):
    """Estimate USD cost for token usage on a known model (0.0 if unknown)."""
    if model not in PRICES:
        return 0.0
    pi, po = PRICES[model]
    return input_tokens / 1_000_000 * pi + output_tokens / 1_000_000 * po
