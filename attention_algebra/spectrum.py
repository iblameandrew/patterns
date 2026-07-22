"""Layer 3 — the Spectrogram Reader.

Maps the Mathematical Schedule from Layer 2 into a cognitive spectrogram:
a time–frequency image where each geometric terminal occupies a carrier band
and schedule dynamics modulate spectral power according to the algebraic
math equivalences.
"""

from __future__ import annotations

import io
import logging
import re
from dataclasses import dataclass

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from .terminals import (
    SYMBOL_TO_TERMINAL,
    TERMINAL_ALT,
    TERMINAL_FREQS,
    TERMINAL_ORDER,
    VOICE_NAME_MAP,
)

log = logging.getLogger(__name__)

VOICE_TERMINAL_RE = re.compile(
    rf"\b({TERMINAL_ALT})\b|"
    r"(Impulse|Anchor|Bifurcate|Return|Eigen|Prune|Harmon|Orth|"
    r"Expand|Bound|Novel|Diffuse)",
    re.IGNORECASE,
)


@dataclass
class SpectrumResult:
    """Output of the spectrogram reader."""

    image: np.ndarray
    report: str
    frequencies: np.ndarray
    times: np.ndarray
    power: np.ndarray


def _resolve_terminal(track: dict) -> str | None:
    """Infer the geometric terminal for a score track."""
    symbol = track.get("symbol", "")
    if symbol in SYMBOL_TO_TERMINAL:
        return SYMBOL_TO_TERMINAL[symbol]

    voice = track.get("voice", "")
    match = VOICE_TERMINAL_RE.search(voice)
    if not match:
        return None
    token = match.group(0)
    if len(token) == 2 and token[0].isupper():
        return token[0] + token[1].lower() if token[1].islower() else token
    # Two-letter codes case-normalised
    if len(token) == 2:
        cand = token[0].upper() + token[1].lower()
        if cand in TERMINAL_FREQS:
            return cand
    return VOICE_NAME_MAP.get(token.lower())


def _normalize_mass(mass: float) -> float:
    return max(float(mass), 0.0) / 10.0


def _envelopes(
    logic: str,
    masses: list[float],
    t: np.ndarray,
    omega: float,
    fold_energy: float | None,
) -> list[np.ndarray]:
    """Return per-track amplitude envelopes for schedule logic."""
    n = len(masses)
    if n == 0:
        return []

    lam = max(omega * 0.1, 0.05)
    envs: list[np.ndarray] = []

    if logic == "Orbital":
        for i, m in enumerate(masses):
            phase = 0.0 if i % 2 == 0 else np.pi / 2
            envs.append(m * (0.5 + 0.5 * np.sin(omega * t + phase)))

    elif logic == "Drag":
        decay = np.exp(-lam * t)
        growth = 1.0 - decay
        for i, m in enumerate(masses):
            envs.append(m * (decay if i == 0 else growth if i == 1 else decay * 0.5))

    elif logic in ("Stochastic Switching", "Domain Switching"):
        for i, m in enumerate(masses):
            duty = 0.5 + 0.1 * np.sin(omega * t + i * np.pi / n)
            envs.append(m * duty)

    elif logic == "Adversarial":
        for i, m in enumerate(masses):
            sign = 1.0 if i == 0 else -0.6
            envs.append(m * sign * (0.7 + 0.3 * np.cos(omega * t)))

    elif logic == "Cooperative Binding":
        stem_scale = 0.5 + 0.5 * np.exp(-lam * t * 0.5)
        for i, m in enumerate(masses):
            role = "stem" if i < 2 else "loop"
            factor = stem_scale if role == "stem" else (1.0 - stem_scale) * 0.4
            envs.append(m * factor)

    elif logic == "Feedback Loop":
        decay = np.exp(-lam * 2 * t)
        for m in masses:
            envs.append(m * decay)

    elif logic == "Partial Adversarial":
        for i, m in enumerate(masses):
            envs.append(m * (1.0 if i == 0 else -0.15))

    elif logic == "Crossing Constraints":
        beat = np.sin(omega * t) * np.sin(omega * 1.5 * t)
        for i, m in enumerate(masses):
            envs.append(m * (0.6 + 0.4 * beat) * (1.0 if i % 2 == 0 else -0.5))

    elif logic == "Softmax Junction":
        logits = np.outer(masses, np.ones_like(t)) + 0.2 * np.sin(omega * t)
        exp_l = np.exp(logits - logits.max(axis=0))
        probs = exp_l / exp_l.sum(axis=0)
        for i, m in enumerate(masses):
            envs.append(m * probs[i])

    elif logic == "Amplified Binding":
        gamma = 1.5
        for m in masses:
            envs.append(m * gamma * (0.8 + 0.2 * np.cos(omega * t)))

    elif logic == "Global Equilibrium":
        scale = 1.0
        if fold_energy is not None:
            scale = np.exp(-fold_energy / 10.0)
        for m in masses:
            envs.append(m * scale * np.ones_like(t))

    elif logic == "Sequential Commitment":
        span = t[-1] if len(t) else 1.0
        for i, m in enumerate(masses):
            start = i / max(n, 1) * span
            envs.append(m * (t >= start).astype(float))

    else:  # Linear and fallback
        for m in masses:
            envs.append(m * np.ones_like(t))

    return envs


class SpectrogramReader:
    """Render a Mathematical Schedule as a cognitive spectrogram."""

    def __init__(
        self,
        sample_rate: int = 2000,
        duration: float = 2.0,
        nfft: int = 256,
    ):
        self.sample_rate = sample_rate
        self.duration = duration
        self.nfft = nfft

    def synthesize(self, composition: dict) -> tuple[np.ndarray, list[str], list[np.ndarray]]:
        """Build mixed signal and per-terminal envelopes."""
        score = composition.get("score", [])
        logic = composition.get("schedule_logic", "Linear")
        omega = float(composition.get("global_frequency", 1.0))
        fold_energy = composition.get("fold_energy")

        n_samples = int(self.sample_rate * self.duration)
        t = np.linspace(0, self.duration, n_samples, endpoint=False)

        tracks: list[tuple[str, float, str]] = []
        for item in score:
            term = _resolve_terminal(item)
            if term is None:
                continue
            mass = _normalize_mass(item.get("mass", 1.0))
            role = item.get("role", "primary")
            tracks.append((term, mass, role))

        if not tracks:
            return np.zeros(n_samples), [], []

        # Re-order envelopes for cooperative binding roles
        masses = [m for _, m, _ in tracks]
        if logic == "Cooperative Binding":
            stems = [m for _, m, r in tracks if r in ("stem", "primary", None)]
            loops = [m for _, m, r in tracks if r == "loop"]
            masses = stems + loops if loops else masses

        envs = _envelopes(logic, masses, t, omega, fold_energy)
        while len(envs) < len(tracks):
            envs.append(np.ones_like(t) * 0.1)

        signal = np.zeros(n_samples)
        labels: list[str] = []
        for (term, mass, role), env in zip(tracks, envs):
            freq = TERMINAL_FREQS[term]
            component = env * np.sin(2 * np.pi * freq * t)
            signal += component
            labels.append(term)

        peak = np.max(np.abs(signal))
        if peak > 0:
            signal /= peak

        return signal, labels, envs

    def compute_spectrogram(self, signal: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Return ``(frequencies, times, power_db)``."""
        from scipy import signal as scipy_signal

        freqs, t, spec = scipy_signal.spectrogram(
            signal,
            fs=self.sample_rate,
            nperseg=self.nfft,
            noverlap=self.nfft // 2,
        )
        power_db = 10 * np.log10(spec + 1e-12)
        return freqs, t, power_db

    def _dominant_bands(
        self, freqs: np.ndarray, power: np.ndarray, labels: list[str]
    ) -> list[str]:
        """Identify which terminal bands carry the most energy."""
        if not labels:
            return []
        band_power: dict[str, float] = {}
        for term in labels:
            f0 = TERMINAL_FREQS[term]
            mask = (freqs >= f0 * 0.9) & (freqs <= f0 * 1.1)
            band_power[term] = float(power[mask].mean()) if mask.any() else 0.0
        ranked = sorted(band_power, key=band_power.get, reverse=True)
        return ranked[:3]

    def render(self, composition: dict) -> SpectrumResult:
        """Produce spectrogram image and textual spectrum report."""
        if isinstance(composition, str):
            return SpectrumResult(
                image=np.zeros((64, 64, 3), dtype=np.uint8),
                report=f"Cannot render spectrogram: {composition}",
                frequencies=np.array([]),
                times=np.array([]),
                power=np.array([[]]),
            )

        expr = composition.get("original_expression", "")
        logic = composition.get("schedule_logic", "Linear")
        omega = composition.get("global_frequency", 1.0)
        fold_energy = composition.get("fold_energy")
        narrative = composition.get("math_narrative", "")

        signal, labels, _ = self.synthesize(composition)
        freqs, times, power = self.compute_spectrogram(signal)
        dominant = self._dominant_bands(freqs, power, labels)

        fig, axes = plt.subplots(2, 1, figsize=(10, 8), gridspec_kw={"height_ratios": [3, 1]})

        im = axes[0].specgram(
            signal,
            Fs=self.sample_rate,
            NFFT=self.nfft,
            noverlap=self.nfft // 2,
            cmap="magma",
        )
        axes[0].set_ylabel("Frequency (Hz)")
        axes[0].set_xlabel("Time (s)")
        axes[0].set_title(f"Cognitive Spectrogram — {logic}")

        for term in TERMINAL_ORDER:
            axes[0].axhline(TERMINAL_FREQS[term], color="white", alpha=0.12, linewidth=0.5)
            axes[0].text(
                0.01,
                TERMINAL_FREQS[term],
                term,
                color="white",
                fontsize=6,
                alpha=0.7,
                transform=axes[0].get_yaxis_transform(),
            )

        fig.colorbar(im[3], ax=axes[0], label="Power (dB)")

        # Band occupancy strip
        band_matrix = np.zeros((len(TERMINAL_ORDER), len(times)))
        for i, term in enumerate(TERMINAL_ORDER):
            f0 = TERMINAL_FREQS[term]
            idx = np.argmin(np.abs(freqs - f0))
            band_matrix[i] = power[idx]

        axes[1].imshow(
            band_matrix,
            aspect="auto",
            origin="lower",
            cmap="viridis",
            extent=[times[0], times[-1], 0, len(TERMINAL_ORDER)],
        )
        axes[1].set_yticks(np.arange(len(TERMINAL_ORDER)) + 0.5)
        axes[1].set_yticklabels(TERMINAL_ORDER, fontsize=7)
        axes[1].set_xlabel("Time (s)")
        axes[1].set_title("Terminal band occupancy")

        fig.tight_layout()

        buf = io.BytesIO()
        fig.savefig(buf, format="png", dpi=120, bbox_inches="tight")
        plt.close(fig)
        buf.seek(0)

        from PIL import Image

        image = np.array(Image.open(buf).convert("RGB"))

        report_lines = [
            "### Cognitive Spectrogram",
            f"**Expression:** `{expr}`",
            f"**Schedule logic:** {logic} (ω={omega})",
        ]
        if fold_energy is not None:
            report_lines.append(f"**Fold energy:** ΔG={fold_energy}")
        if narrative:
            report_lines.append(f"*{narrative}*")
        if dominant:
            report_lines.append(
                f"**Dominant bands:** {' → '.join(dominant)} (by mean spectral power)"
            )
        report_lines.append("")
        report_lines.append("| Terminal | Carrier (Hz) | Objective |")
        report_lines.append("| :--- | ---: | :--- |")
        for item in composition.get("score", []):
            term = _resolve_terminal(item) or "?"
            freq = TERMINAL_FREQS.get(term, 0)
            report_lines.append(
                f"| {term} | {freq:.1f} | `{item.get('symbol', '')}` |"
            )

        report = "\n".join(report_lines)
        log.debug("Rendered spectrogram: %d bands, logic=%s", len(labels), logic)

        return SpectrumResult(
            image=image,
            report=report,
            frequencies=freqs,
            times=times,
            power=power,
        )

    def read(self, composition: dict) -> tuple[np.ndarray, str]:
        """Return ``(image, report)`` for pipeline consumers."""
        result = self.render(composition)
        return result.image, result.report
