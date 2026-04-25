# API Reference

The public API is grouped into:

- `aibackends.configure(...)`
- `aibackends.tasks.*`
- `aibackends.workflows.*`
- `aibackends.core.runtimes.*`

The codebase is structured to keep task functions framework-agnostic while allowing runtime and workflow internals to evolve independently.
