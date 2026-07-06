"""Context assembly: builds the prompts that go to the model.

One place owns everything that enters a model's context (skills, taxonomy,
exemplars, ticket), and it records which exemplars were used so the eval can keep
headline metrics on tickets that were never in a prompt.
"""
