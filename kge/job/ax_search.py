from kge.job import AutoSearchJob
from kge import Config
from ax.service.ax_client import AxClient
from typing import List


class AxSearchJob(AutoSearchJob):
    """Job for hyperparameter search using [ax](https://ax.dev/)."""

    def __init__(self, config: Config, dataset, parent_job=None):
        super().__init__(config, dataset, parent_job)
        self.ax_client = None

    # Overridden such that instances of search job can be pickled to workers
    def __getstate__(self):
        state = super(AxSearchJob, self).__getstate__()
        del state["ax_client"]
        return state

    def init_search(self):
        self.ax_client = AxClient()
        self.ax_client.create_experiment(
            name=self.job_id,
            parameters=self.config.get("ax_search.parameters"),
            objective_name="metric_value",
            minimize=False,
        )

        # By default, ax first uses a Sobol strategy for a certain number of arms,
        # followed by Bayesian Optimization. If we resume this job, some of the Sobol
        # arms may have already been generated. The corresponding arms will be
        # registered later (when this job's run method is executed), but here we already
        # change the generation strategy to take account of these configurations.
        num_generated = len(self.parameters)
        if num_generated > 0:
            old_curr = self.ax_client.generation_strategy._curr
            new_num_arms = max(0, old_curr.num_arms - num_generated)
            new_curr = old_curr._replace(num_arms=new_num_arms)
            self.ax_client.generation_strategy._curr = new_curr
            self.config.log(
                "Reduced number of arms for first generation step of "
                + "ax_client from {} to {} due to prior data.".format(
                    old_curr.num_arms, new_curr.num_arms
                )
            )

    def register_trial(self, parameters=None):
        try:
            if parameters is None:
                parameters, trial_id = self.ax_client.get_next_trial()
            else:
                _, trial_id = self.ax_client.attach_trial(parameters)
            return parameters, trial_id
        except ValueError:
            # error: ax needs more data
            return None, None

    def register_trial_result(self, trial_id, parameters, trace_entry):
        # TODO: std dev shouldn't be fixed to 0.0
        self.ax_client.complete_trial(
            trial_index=trial_id,
            raw_data={"metric_value": (trace_entry["metric_value"], 0.0)},
        )

    def get_best_parameters(self):
        best_parameters, values = self.ax_client.get_best_parameters()
        return best_parameters, float(values[0]["metric_value"])
