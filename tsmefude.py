"""Time-Shift Multiscale Ensemble Fuzzy Dispersion Entropy (TSMEFuDE).

Translation of ``TSMEFuDE.m`` under ``legacy/matlab/base_methods/`` with
the F.2 correction approved by the project owner on 2026-08-19: the
nested ``TSE_beta`` of the original computes only the last phase per
scale (the ``X{beta}=[]`` reset discards every ``m < beta`` phase) and
caps each phase at the largest multiple of ``beta`` <= N instead of N,
so intermediate phases (1 < beta < k, beta not dividing N) lose tail
samples. This module implements the published time-shift multiscale
definition instead: ``X{beta} = S(beta:k:end)`` for beta = 1..k, one
entropy per phase, averaged over the k phases.

The deviation from the legacy output is quantified in
``tests/matlab_reference/tsmefude/entropy_ref_*.csv`` (TSEn_legacy
column); see ``docs/open_questions.md`` (F.2 ruling) and
``docs/discrepancy_log.md`` (D.15).
"""

import numpy as np

from msentropy.core.ensfude import ensfude


def _tse_beta(s: np.ndarray, k: int) -> list[np.ndarray]:
    """Phase subseries of scale k, corrected paper definition (F.2).

    ``X{beta} = S(beta:k:end)`` for beta = 1..k (MATLAB 1-based), i.e.
    ``s[beta-1::k]`` in Python, the full-tail Higuchi-style time-shift
    decomposition. A phase whose start index lies beyond the signal is
    empty, exactly like the original MATLAB cell entries.
    """
    s = np.asarray(s, dtype=np.float64).ravel()
    return [s[b::k] for b in range(k)]


def tsmefude(
    x: np.ndarray,
    m: int,
    nc: int,
    tau: int,
    kmax: int,
) -> np.ndarray:
    """Time-Shift Multiscale Ensemble Fuzzy Dispersion Entropy.

    Python translation of the legacy MATLAB ``TSMEFuDE(X, dim, nc, tau,
    Kmax)`` with the F.2 correction (see the module docstring). Steps:

    1. For each scale k in 1..Kmax, build the k phase subseries
       ``X{beta} = S(beta:k:end)`` (Higuchi-style time shifting).
    2. Compute ``En(beta) = EnsFuDE(X{beta}, m, nc, tau)`` per phase.
    3. ``TSEn(k) = mean(En)``, summed sequentially starting from the
       FIRST element like MATLAB's ``sum`` (no +0 identity), so the
       summation order matches exactly and an all-``-0.0`` ``En`` keeps
       its sign (``np.add.reduce`` would flip it to +0 through its
       +0 identity).

    For k = 1 the phase subseries is the signal itself, so entry 0 is
    plain ``EnsFuDE(x, ...)`` — identical to the legacy output, whose
    beta = 1 phase has no truncation either (probe P2 in
    ``tests/matlab_reference/tsmefude/edge_behavior.txt``).

    Parameters
    ----------
    x : np.ndarray
        Input signal: 1-D (MATLAB row convention) or vector-shaped 2-D
        ``(1, N)`` / ``(N, 1)``. Unlike :func:`mefude`, orientation does
        NOT matter: the MATLAB subseries builder indexes scalar samples
        and always concatenates rows, so row and column inputs give
        identical results (probe P5).
    m : int
        Embedding dimension (MATLAB argument name ``dim``).
    nc : int
        Number of fuzzy classes.
    tau : int
        Delay (in samples).
    kmax : int
        Maximum time scale (MATLAB argument name ``Kmax``).

    Returns
    -------
    np.ndarray
        1-D array of ``kmax`` entropy values; entry k-1 is the mean of
        the k phase entropies of scale k. For ``kmax <= 0`` the array
        is empty: MATLAB's ``zeros(1, Kmax)`` silently treats a
        negative dimension as 0 and the ``1:Kmax`` loop is empty
        (probes P6/P7).

    Raises
    ------
    ValueError
        Non-vector 2-D or higher-dimensional input.
    UnboundLocalError
        Constant signals, or scales where a phase has a single sample
        (std = 0), make the TANSIG/LOGSIG mappings of :func:`ensfude`
        produce NaN class coordinates, reproducing the probed MATLAB
        failure inside ``trapezoidal_MF`` (probes P1/P3).

    Notes
    -----
    - A phase with ``W = L - (m-1)*tau == 0`` gives entropy NaN, which
      then makes ``TSEn(k)`` NaN through the mean, as in MATLAB
      (probe P4).
    - Phases of length 1 exist whenever ``k + beta > N``; ``kmax > N``
      makes EVERY beta <= N phase single-sample, so scale kmax fails at
      its very first phase and the empty phases (beta > N) are never
      reached — empty phases are unreachable in practice for
      m, tau >= 1 (probe P1).
    """
    x = np.asarray(x, dtype=np.float64)
    if x.ndim > 2 or (x.ndim == 2 and 1 not in x.shape):
        raise ValueError(
            "tsmefude: x must be 1-D (MATLAB row convention) or a "
            "vector-shaped 2-D array (1xN or Nx1)"
        )

    # MATLAB zeros(1, Kmax) treats a negative dimension as 0 (probe P7)
    # and the 1:Kmax loop is then empty, so kmax <= 0 -> empty output.
    out = np.empty(max(kmax, 0))
    for k in range(1, kmax + 1):
        phases = _tse_beta(x, k)
        en = np.array([ensfude(p, m, nc, tau)[0] for p in phases])
        # MATLAB sum: sequential accumulation seeded with the first
        # element (see the step-3 note in the docstring).
        total = en[0]
        for val in en[1:]:
            total = total + val
        out[k - 1] = total / k
    return out
