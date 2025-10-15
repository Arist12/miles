import importlib
import subprocess


def load_function(path):
    """
    Load a function from a module.
    :param path: The path to the function, e.g. "module.submodule.function".
    :return: The function object.
    """
    module_path, _, attr = path.rpartition(".")
    module = importlib.import_module(module_path)
    return getattr(module, attr)


class SingletonMeta(type):
    """
    A metaclass for creating singleton classes.
    """

    _instances = {}

    def __call__(cls, *args, **kwargs):
        if cls not in cls._instances:
            instance = super().__call__(*args, **kwargs)
            cls._instances[cls] = instance
        return cls._instances[cls]


def exec_command(cmd: str, capture_output: bool = False):
    print(f"EXEC: {cmd}", flush=True)
    result = subprocess.run(["bash", "-c", cmd], shell=False, check=True, capture_output=capture_output)
    if capture_output:
        return result.stdout
