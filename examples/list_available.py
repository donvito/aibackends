import sys

from aibackends.core.exceptions import AIBackendsError
from aibackends.models import available_models
from aibackends.runtimes import available_runtimes
from aibackends.tasks import available_tasks
from aibackends.workflows import available_workflows


def _print_group(title: str, items: dict[str, type]) -> None:
    print(f"{title} ({len(items)}):")
    for name, cls in items.items():
        print(f"- {name} -> {cls.__name__}")
    print()


def _print_runtime_group() -> None:
    runtimes = available_runtimes()
    print(f"Available runtimes ({len(runtimes)}):")
    for name, spec in runtimes.items():
        print(f"- {name} -> {spec.factory.__name__}")
    print()


def _print_model_group() -> None:
    models = available_models()
    print(f"Available models ({len(models)}):")
    for name in models:
        print(f"- {name}")
    print()


def main() -> None:
    try:
        _print_runtime_group()
        _print_model_group()
        _print_group("Available tasks", available_tasks())
        _print_group("Available workflows", available_workflows())
    except KeyboardInterrupt:
        print("Example cancelled by user.", file=sys.stderr)
        raise SystemExit(130) from None
    except AIBackendsError as exc:
        print(f"Example failed: {exc}", file=sys.stderr)
        raise SystemExit(1) from exc


if __name__ == "__main__":
    main()
