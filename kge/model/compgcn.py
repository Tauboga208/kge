from kge import Config, Dataset
from kge.model.kge_model import KgeRgnnModel

import importlib


class CompGCN(KgeRgnnModel):
    """
    This is a base model, which serves to combine an rgnn-encoder with a
    scoring function as decoder.
    """
    def __init__(
        self,
        config: Config,
        dataset: Dataset,
        configuration_key=None,
        init_for_load_only=False,
    ):
        self._init_configuration(config, configuration_key)

        # set the correct scorer
        if self.get_option("decoder.model") == "reciprocal_relations_model":
            decoder_model = importlib.import_module(
                "kge.model." + self.get_option("decoder.reciprocal_relations_model.base_model.type"))  
        else:  
            decoder_model = importlib.import_module("kge.model." + self.get_option("decoder.model"))
        scorer = getattr(decoder_model, self.get_option("decoder.scorer"))

        super().__init__(
            config=config,
            dataset=dataset,
            scorer=scorer,
            configuration_key=self.configuration_key,
            init_for_load_only=init_for_load_only,
        )