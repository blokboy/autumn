# GEPA example scripts

These are ordinary, offline GEPA scripts that Autumn can run without API keys.
They use deterministic Python callables in place of task and reflection LMs, so
they are safe for demos and tests of the dashboard flow.

From the repository root, run the full example set with:

```bash
autumn run --run-dir fixtures/examples
```

Inside the Autumn command bar, the same directory run is:

```text
gepa --run-dir fixtures/examples
```

Autumn discovers direct child `*.py` files in filename order, skips files like
this `README.md`, launches the first script, and queues the rest.

To run just one example, try one of:

```bash
autumn run fixtures/examples/uppercase_words.py --run-dir fixtures/examples/runs/uppercase-words
autumn run fixtures/examples/sentiment_labels.py --run-dir fixtures/examples/runs/sentiment-labels
autumn run fixtures/examples/json_ticket_router.py --run-dir fixtures/examples/runs/json-ticket-router
autumn run fixtures/examples/priority_triage.py --run-dir fixtures/examples/runs/priority-triage
autumn run fixtures/examples/product_slugs.py --run-dir fixtures/examples/runs/product-slugs
autumn run fixtures/examples/status_csv.py --run-dir fixtures/examples/runs/status-csv
```

Inside the Autumn command bar, individual examples can be launched with:

```text
gepa fixtures/examples/uppercase_words.py --run-dir fixtures/examples/runs/uppercase-words
gepa fixtures/examples/sentiment_labels.py --run-dir fixtures/examples/runs/sentiment-labels
gepa fixtures/examples/json_ticket_router.py --run-dir fixtures/examples/runs/json-ticket-router
gepa fixtures/examples/priority_triage.py --run-dir fixtures/examples/runs/priority-triage
gepa fixtures/examples/product_slugs.py --run-dir fixtures/examples/runs/product-slugs
gepa fixtures/examples/status_csv.py --run-dir fixtures/examples/runs/status-csv
```

The `runs/` subdirectory is ignored by git so demo output can stay local.
