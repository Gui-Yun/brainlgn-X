"""
JAX-native transfer functions to reduce numpy<->JAX overhead in BS backend.

Currently supports ReLU with bias (Max(0, s + b)).
"""

from typing import Optional

try:
    import jax
    import jax.numpy as jnp
except Exception:  # pragma: no cover
    jax = None
    jnp = None


def _require_jax():  # pragma: no cover
    if jnp is None:
        raise ImportError("JAX is required for transfer_jax. Install jax/jaxlib.")


def relu_bias(x, bias: float = 0.0):
    """JAX ReLU with bias: y = max(0, x + b)."""
    _require_jax()
    return jnp.maximum(0.0, x + float(bias))


def maybe_parse_bias_from_scalar_tf(tf_string: str) -> Optional[float]:
    """
    Heuristic parser: try to extract bias from strings like 'Max(0, s + 1.0)'.
    Returns float bias or None if not recognized.
    """
    s = tf_string.replace(' ', '').lower()
    if not (s.startswith('max(0,') and s.endswith(')')):
        return None
    # Expect pattern: max(0,s+<bias>) or max(0,s-<bias>)
    inner = s[len('max(0,'):-1]
    if inner.startswith('s+'):
        try:
            return float(inner[2:])
        except Exception:
            return None
    if inner.startswith('s-'):
        try:
            return -float(inner[2:])
        except Exception:
            return None
    if inner == 's':
        return 0.0
    return None

