"""Competition metric: Tweedie D² (power 1.5), baseline = constant mean."""
import numpy as np

POWER = 1.5


def tweedie_deviance(y, mu, p: float = POWER) -> np.ndarray:
    y, mu = np.asarray(y, float), np.clip(np.asarray(mu, float), 1e-9, None)
    return 2 * (np.power(y, 2 - p) / ((1 - p) * (2 - p)) - y * np.power(mu, 1 - p) / (1 - p) + np.power(mu, 2 - p) / (2 - p))


def d2_tweedie(y, mu, p: float = POWER) -> float:
    y = np.asarray(y, float)
    return 1 - tweedie_deviance(y, mu, p).sum() / tweedie_deviance(y, np.full_like(y, y.mean()), p).sum()
