# GEPA example scripts

These are ordinary, offline GEPA scripts that Autumn can run without API keys.
They use deterministic Python callables in place of task and reflection LMs, so
they are safe for demos and tests of the dashboard flow.

From the repository root, try one of:

```bash
autumn run fixtures/examples/uppercase_words.py --run-dir fixtures/examples/runs/uppercase-words
autumn run fixtures/examples/sentiment_labels.py --run-dir fixtures/examples/runs/sentiment-labels
autumn run fixtures/examples/json_ticket_router.py --run-dir fixtures/examples/runs/json-ticket-router
```

Inside the Autumn command bar, the same examples can be launched with:

```text
gepa fixtures/examples/uppercase_words.py --run-dir fixtures/examples/runs/uppercase-words
gepa fixtures/examples/sentiment_labels.py --run-dir fixtures/examples/runs/sentiment-labels
gepa fixtures/examples/json_ticket_router.py --run-dir fixtures/examples/runs/json-ticket-router
```

The `runs/` subdirectory is ignored by git so demo output can stay local.
