import math
from collections import Counter, deque
from typing import Callable, Iterable, List, Sequence, Union

Label = Union[int, str]

# ---------------------------
# Weighting functions
# d = distance from the window center (non-negative integer)
# ws = context window size (positive integer)
# ---------------------------

def W1_normal(ws: int, d: int) -> float:
    # ceil(ws/2) - d
    return max(0.0, math.ceil(ws / 2) - d)

def W2_normal_inverted(ws: int, d: int) -> float:
    # 1 / (d + 2)
    return 1.0 / (d + 2)

def W3_squared(ws: int, d: int) -> float:
    # (ceil(ws/2) - d)^2
    base = max(0, math.ceil(ws / 2) - d)
    return float(base * base)

def W4_log(ws: int, d: int) -> float:
    # log(ceil(ws/2) - d + 1)
    val = max(1, math.ceil(ws / 2) - d + 1)
    return math.log(val)

def W5_log_inverted(ws: int, d: int) -> float:
    # 1 / log(d + 2)
    return 1.0 / math.log(d + 2)

WEIGHTS: dict[str, Callable[[int, int], float]] = {
    "W1": W1_normal,
    "W2": W2_normal_inverted,
    "W3": W3_squared,
    "W4": W4_log,
    "W5": W5_log_inverted,
}

# --------------------------------------------
# Algorithm 2: Context-based Prediction Correction
# --------------------------------------------

def context_correct(
    W: Sequence[Label],
    ws: int,
    weight_fn: Callable[[int, int], float],
) -> Label:
    """
    Given a context window W (sequence of labels) and a weighting function,
    return the corrected label, following Algorithm 2 with the tie-break rule:
    - choose the label with the maximum weighted score;
    - if scores tie, prefer the label with the higher raw count in W.
    """
    # counts over W
    counts = Counter(W)
    scores = {x: 0.0 for x in counts.keys()}

    # 1-based midpoint per paper: midpoint = floor(|W|/2) + 1
    # we will iterate i in [1..|W|] and use d = |i - midpoint|
    n = len(W)
    midpoint = (n // 2) + 1

    # accumulate weighted scores by position
    # W is 0-based index in Python; convert to 1-based for distance calc
    for i in range(1, n + 1):
        d = abs(i - midpoint)
        w = weight_fn(ws, d)
        lbl = W[i - 1]
        scores[lbl] += w

    # select best by score, then by raw count if tie
    best_label = None
    best_score = -float("inf")
    for x in counts.keys():
        s = scores[x]
        if s > best_score:
            best_score = s
            best_label = x
        elif s == best_score:
            if counts[x] > counts[best_label]:
                best_label = x

    return best_label

# ------------------------------------------------------
# Algorithm 1: Classification Postprocessing Algorithm
# Stream version with latency = ws - 1 future predictions
# ------------------------------------------------------

def postprocess_stream(
    P: Iterable[Label],
    ws: int,
    weight: Union[str, Callable[[int, int], float]] = "W4",
) -> List[Label]:
    """
    Convert a raw prediction stream P into a corrected stream P′ using a sliding
    context window of size ws and the chosen weight function.

    Behavior matches Algorithm 1:
      - Until the window W reaches size ws, pass labels through unchanged.
      - Once |W| == ws, emit context_correct(W) for each new step.
      - Maintain W as a FIFO queue (pop front, push new).

    Notes:
      * This introduces latency proportional to the number of future labels used.
      * Use ws >= 1. For ws == 1, output equals input.
    """
    if ws <= 0:
        raise ValueError("ws must be a positive integer")

    # Resolve weight function
    if isinstance(weight, str):
        if weight not in WEIGHTS:
            raise ValueError(f"Unknown weight '{weight}'. Choose one of {list(WEIGHTS)}")
        weight_fn = WEIGHTS[weight]
    else:
        weight_fn = weight

    P_prime: List[Label] = []
    Wq: deque[Label] = deque()

    for x in P:
        if len(Wq) < ws:
            # Not enough context yet; pass through (paper's line 4-5)
            P_prime.append(x)
        else:
            # Have a full window; correct based on W (lines 6-7)
            P_prime.append(context_correct(list(Wq), ws, weight_fn))
            # slide (lines 8-9)
            Wq.popleft()
        Wq.append(x)

    return P_prime
