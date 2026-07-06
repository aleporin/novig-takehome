"""Model routing: the T1->T2 escalation cascade and per-call cost accounting.

Routing is subordinate to safety. It chooses which model classifies a ticket; it
never chooses whether the safety gate applies. Escalation can only add flags, so
a stronger model can confirm a concern but never clear one.
"""
