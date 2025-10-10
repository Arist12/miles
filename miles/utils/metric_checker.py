from typing import Dict


class MetricChecker:
    def __init__(self, args):
        self.args = args

    def on_eval(self, metrics: Dict[str, float]):
        TODO
