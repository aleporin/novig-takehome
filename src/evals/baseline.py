"""A trivial baseline predictor: majority class, always decline.

No AI. It predicts the most common category and urgency from the training labels
and declines to draft on every ticket. Its only job is to give the harness
something real to measure so we can test the harness before the classifier exists.
"""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass

from triage.schemas import Category, Prediction, Ticket, Urgency

_DECLINE_REASON = "baseline predictor declines all tickets"


@dataclass(frozen=True)
class MajorityBaseline:
    """Predicts fixed majority-class labels and never drafts."""

    category: Category
    urgency: Urgency

    @classmethod
    def from_labels(cls, tickets: list[Ticket]) -> MajorityBaseline:
        """Pick the most common category and urgency from labeled tickets."""
        labeled = [t.label for t in tickets if t.label is not None]
        if not labeled:
            raise ValueError("cannot build a majority baseline from unlabeled tickets")
        category = Counter(label.category for label in labeled).most_common(1)[0][0]
        urgency = Counter(label.urgency for label in labeled).most_common(1)[0][0]
        return cls(category=category, urgency=urgency)

    def predict(self, ticket: Ticket) -> Prediction:
        return Prediction(
            ticket_id=ticket.ticket_id,
            category=self.category,
            urgency=self.urgency,
            should_draft=False,
            no_draft_reason=_DECLINE_REASON,
            draft_response=None,
            confidence=0.0,
        )
