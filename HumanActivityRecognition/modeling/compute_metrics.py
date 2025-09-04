from collections import defaultdict
from typing import List, Tuple, Dict

Label = str  # or int

# ----------------------------------------------------------
# Step 1: Convert sequences into event lists
# Each event = (label, start_index, end_index)
# ----------------------------------------------------------

def to_events(seq: List[Label]) -> List[Tuple[Label, int, int]]:
    """
    Collapse consecutive identical labels into events.
    """
    if not seq:
        return []

    events = []
    start = 0
    current = seq[0]

    for i in range(1, len(seq)):
        if seq[i] != current:
            events.append((current, start, i - 1))
            start = i
            current = seq[i]

    # last event
    events.append((current, start, len(seq) - 1))
    return events

# ----------------------------------------------------------
# Step 2: Compute overlap between events
# ----------------------------------------------------------

def overlap(ev1: Tuple[Label, int, int], ev2: Tuple[Label, int, int]) -> int:
    """
    Return length of overlap between two events (label ignored).
    """
    _, s1, e1 = ev1
    _, s2, e2 = ev2
    return max(0, min(e1, e2) - max(s1, s2) + 1)

# ----------------------------------------------------------
# Step 3: Event-level evaluation
# ----------------------------------------------------------

def evaluate_events(
    y_true: List[Label],
    y_pred: List[Label],
) -> Dict[str, int]:
    """
    Compute event-level counts: C, F, M', FM', F', I', D
    following the taxonomy described in the paper.
    """

    gt_events = to_events(y_true)
    pr_events = to_events(y_pred)

    # track matches
    gt_to_pr = defaultdict(list)
    pr_to_gt = defaultdict(list)

    # build overlaps
    for gi, ge in enumerate(gt_events):
        for pi, pe in enumerate(pr_events):
            if ge[0] == pe[0]:  # same label
                if overlap(ge, pe) > 0:
                    gt_to_pr[gi].append(pi)
                    pr_to_gt[pi].append(gi)

    # initialize counts
    C = F = Mp = FMp = Fp = Ip = D = 0

    # classify ground-truth events
    for gi, ge in enumerate(gt_events):
        matched_preds = gt_to_pr.get(gi, [])
        if not matched_preds:
            # deleted event
            D += 1
        elif len(matched_preds) == 1:
            pi = matched_preds[0]
            if len(pr_to_gt[pi]) == 1:
                # perfect one-to-one match
                C += 1
            else:
                # merged into larger pred event
                Mp += 1
        else:
            # fragmented (matched multiple pred events)
            all_multi = all(len(pr_to_gt[pi]) > 1 for pi in matched_preds)
            if all_multi:
                FMp += 1
            else:
                F += 1

    # classify predicted events
    for pi, pe in enumerate(pr_events):
        matched_gts = pr_to_gt.get(pi, [])
        if not matched_gts:
            # false/inserted
            if pe[0] in {g[0] for g in gt_events}:
                Fp += 1
            else:
                Ip += 1
        # else: already accounted for in GT classification

    return dict(C=C, F=F, Mp=Mp, FMp=FMp, Fp=Fp, Ip=Ip, D=D)

# ----------------------------------------------------------
# Step 4: Event-level Precision, Recall, F1
# ----------------------------------------------------------

def event_precision_recall_f1(counts: Dict[str, int]) -> Dict[str, float]:
    """
    Compute event-level precision, recall, F1-score from counts.
    """

    C, F, Mp, FMp, Fp, Ip, D = (
        counts["C"],
        counts["F"],
        counts["Mp"],
        counts["FMp"],
        counts["Fp"],
        counts["Ip"],
        counts["D"],
    )

    TP = C + F + Mp + FMp
    FP = Fp + Ip
    FN = D

    precision = TP / (TP + FP) if (TP + FP) > 0 else 0.0
    recall = TP / (TP + FN) if (TP + FN) > 0 else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0

    return {"precision": precision, "recall": recall, "f1": f1}
