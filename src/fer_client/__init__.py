"""Module 3: client for querying the FER server.

Not implemented yet - scaffolding only. Deliberately has no dependency on
`common` (no torch, no model code) - it only knows how to send an image to
the server's prediction endpoint and parse the response, so it stays usable
by anyone who just wants to query a running server without installing the
full ML stack. Used by `attacks` (black-box queries) and `defenses`
(inference-time defenses that wrap the server). See PLAN.md.
"""
