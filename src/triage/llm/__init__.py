"""LLM access layer: the interface, the disk cache, and the Anthropic client.

The client sits behind the LLMClient interface so tests can swap in a fake and
run without an API key.
"""
