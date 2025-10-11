import logging
import threading

import ray


class RolloutHealthMonitor:
    def __init__(self, args):
        # fault tolerance
        self._health_monitor_thread = None
        self._health_monitor_stop_event = None
        self._health_check_interval = args.rollout_health_check_interval
        self._health_check_timeout = args.rollout_health_check_timeout
        self._health_check_first_wait = args.rollout_health_check_first_wait

    def start(self) -> bool:
        if not self.rollout_engines:
            return False

        assert self._health_monitor_thread is None, "Health monitor thread is already running."

        self._health_monitor_stop_event = threading.Event()
        self._health_monitor_thread = threading.Thread(
            target=self._health_monitor_loop,
            name="RolloutHealthMonitor",
            daemon=True,
        )
        self._health_monitor_thread.start()
        return True

    def stop(self) -> None:
        if not self._health_monitor_thread:
            return

        assert self._health_monitor_stop_event is not None
        self._health_monitor_stop_event.set()
        timeout = self._health_check_timeout + self._health_check_interval + 5
        self._health_monitor_thread.join(timeout=timeout)
        if self._health_monitor_thread.is_alive():
            logging.warning("Rollout health monitor thread did not terminate within %.1fs", timeout)

        self._health_monitor_thread = None
        self._health_monitor_stop_event = None

    def _health_monitor_loop(self) -> None:
        assert self._health_monitor_stop_event is not None
        # TODO: need to be waiting for the large moe to be ready. this is hacky.
        if self._health_monitor_stop_event.wait(self._health_check_first_wait):
            return
        while not self._health_monitor_stop_event.is_set():
            self._run_health_checks()
            if self._health_monitor_stop_event.wait(self._health_check_interval):
                break

    def _run_health_checks(self) -> None:
        for rollout_engine_id, engine in enumerate(self.rollout_engines):
            if self._health_monitor_stop_event is not None and self._health_monitor_stop_event.is_set():
                break
            self._check_engine_health(rollout_engine_id, engine)

    def _check_engine_health(self, rollout_engine_id, engine) -> None:
        if engine is None:
            return

        try:
            ray.get(engine.health_generate.remote(timeout=self._health_check_timeout))
        except Exception as e:
            print(f"Health check timed out for rollout engine {rollout_engine_id} (ray timeout). Killing actor.")
            for i in range(rollout_engine_id * self.nodes_per_engine, (rollout_engine_id + 1) * self.nodes_per_engine):
                engine = self.all_rollout_engines[i]
                try:
                    ray.kill(engine)
                except Exception:
                    pass
                self.all_rollout_engines[i] = None
            self.rollout_engines[rollout_engine_id] = None
