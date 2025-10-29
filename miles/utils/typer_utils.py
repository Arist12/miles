import dataclasses
import inspect
from typing import Any, Callable, Dict, get_type_hints


def patch_typer():
    try:
        import typer

        _ = typer
    except ImportError:
        print("Skip patch_typer since typer is not installed.")
        return

    print("Patching typer...")
    _patch_typer_dataclass_params()
    _patch_typer_run()


def _patch_typer_dataclass_params():
    import typer
    from typer.models import ParamMeta

    def _modified_get_params_from_function(func: Callable[..., Any]) -> Dict[str, ParamMeta]:
        """https://github.com/fastapi/typer/issues/154#issuecomment-810423284"""
        signature = inspect.signature(func)

        type_hints = get_type_hints(func)
        params = {}

        for param in signature.parameters.values():
            annotation = param.annotation
            if param.name == "kwargs":
                continue
            if param.name in type_hints:
                annotation = type_hints[param.name]
            if inspect.isclass(annotation) and dataclasses.is_dataclass(annotation):
                if inspect.isclass(param.default) and issubclass(param.default, inspect.Parameter.empty):
                    dct = dataclasses.asdict(annotation())
                    subtype_hints = get_type_hints(annotation)
                else:
                    dct = dataclasses.asdict(param.default)
                    subtype_hints = get_type_hints(param.default)
                for k, v in dct.items():
                    params[k] = ParamMeta(name=k, default=v, annotation=subtype_hints.get(k, str))
            else:
                params[param.name] = ParamMeta(name=param.name, default=param.default, annotation=annotation)
        return params

    typer.main.get_params_from_function = _modified_get_params_from_function


def _patch_typer_run():
    import typer

    def _modified_run(fn):
        def wrapped_fn(*args, **kwargs):
            TODO

        original_typer_run(wrapped_fn)

    original_typer_run = typer.run
    typer.run = _modified_run
