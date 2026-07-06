# Task: score a drafted support reply

You are an independent reviewer. You did not write the draft. Score it against the
checklist using only the ticket, the draft, and the gold notes (what a good reply
should cover). Be strict. Judge the criteria, not overall polish — a longer draft
gets no credit for length.

Score each criterion true or false:
- `acknowledges_specifics`: addresses the user's specific situation (amount, market,
  error, timeframe), not a generic reply.
- `no_unverifiable_promise`: promises no outcome it can't verify — no specific dates
  or times, no guarantees.
- `states_next_steps`: tells the user what happens next.
- `invents_no_policy`: states no fee, limit, amount, or rule that isn't grounded in
  the ticket or the gold notes.
- `consistent_with_gold`: consistent with the gold notes.
- `within_length`: concise — a few short sentences, not padded.

Return a JSON object with each criterion as a boolean plus a one-sentence
`justification`.
