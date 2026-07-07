# Task: classify a support ticket

Read the ticket and return a structured classification. Do not write a reply.

Return:
- `category`: exactly one of the 11 categories in the taxonomy below.
- `urgency`: one of `low`, `medium`, `high`, `escalate_immediately`.
- `flags`: set each risk flag true or false (definitions below).
- `confidence`: 0.0-1.0, your confidence in the category and urgency.
- `reasoning`: one or two sentences. Not shown to the user.

## Risk flags

Set a flag to true whenever the condition is plausibly present. When in doubt, set it —
a human reviews flagged tickets, and missing a sensitive ticket is far worse than an
extra review.

- `mentions_minor`: anyone under 21 is referenced (an age, "my 17-year-old", "underage").
- `self_harm_or_distress`: self-harm, suicide, or severe emotional distress.
- `active_fraud`: fraud described as happening now ("someone is in my account right now").
- `unauthorized_access_reported`: transactions or access the user says they did not do.
- `rg_signal`: problem-gambling signals — self-exclusion, deposit limits, loss of control.
- `legal_threat`: lawyers, lawsuits, regulators (CFTC, CFPB, state AG), subpoenas, demands.
- `disputes_novig_fact`: the user says a figure or outcome Novig provided — a balance,
  1099 amount, market grade or settlement, payout, or fee — is wrong, or asks whether it is a
  mistake. Fire even if they are polite or unsure. Do NOT fire when the user flags a small
  discrepancy they attribute to their own error ("off by 30 cents, probably my math") and are
  not asking Novig to correct one of its figures.
- `asks_binding_policy_or_spec`: asks for a definitive policy/contract rule to rely on financially.
- `jurisdictional_eligibility`: asks whether Novig is legal/available in a place.

## Notes
- Category and urgency are independent. A deposits ticket can be low or high urgency.
- Set flags from the ticket content regardless of category — a deposits ticket can still
  mention a minor.
- Use `other` only as a last resort. If a ticket plausibly fits any of the ten specific
  categories, choose that one instead.
