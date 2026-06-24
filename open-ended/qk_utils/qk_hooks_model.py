from collections import OrderedDict
from typing import Dict

import torch
from transformers import AutoModelForCausalLM


class QKHooksModel(torch.nn.Module):
    """
    Wraps a HF CausalLM and registers forward hooks on q_proj and k_proj
    for every layer, storing outputs in `selected_out`.
    """
    def __init__(self, model_name: str, device: str, max_memory = None, *args):
        super().__init__(*args)
        self.selected_out: Dict[str, torch.Tensor] = OrderedDict()

        if max_memory is None:
            max_memory = {
                0: "22GB",
                2: "22GB",
                "cpu": "96GB"}

        print(max_memory)

        self.pretrained = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=torch.float16 if "cuda" in device else torch.float32,
            device_map="balanced",
            max_memory=max_memory,
            low_cpu_mem_usage=True,
        )
        self.pretrained.eval()
        self.fhooks = []

        # ILYA
        self.pretrained = self.pretrained.half()

        for i in range(len(self.pretrained.model.layers)):
            self.fhooks.append(
                self.pretrained.model.layers[i].self_attn.q_proj.register_forward_hook(
                    self.forward_hook(f"query_vec_{i}")
                )
            )
            self.fhooks.append(
                self.pretrained.model.layers[i].self_attn.k_proj.register_forward_hook(
                    self.forward_hook(f"key_vec_{i}")
                )
            )

    def forward_hook(self, layer_name: str):
        def hook(_module, _input, output):
            self.selected_out[layer_name] = output.detach().to("cpu")
        return hook

    def clear(self):
        self.selected_out = OrderedDict()

    def forward(self, x):
        self.clear()
        with torch.inference_mode():
            _ = self.pretrained(**x, use_cache=False)

        return None, OrderedDict((k, v) for k, v in self.selected_out.items())


def get_qk_config(model_config) -> Dict[str, int]:
    """
    Derives QK configuration from HF model config.
    """
    qk_config = {}
    qk_config["L"] = model_config.num_hidden_layers
    qk_config["H"] = model_config.num_attention_heads

    if "head_dim" in model_config.__dict__:
        qk_config["H_span"] = model_config.head_dim
    else:
        qk_config["H_span"] = model_config.hidden_size // qk_config["H"]

    if "num_key_value_heads" in model_config.__dict__:
        qk_config["Q_per_K"] = model_config.num_attention_heads // model_config.num_key_value_heads
    else:
        qk_config["Q_per_K"] = 1

    return qk_config
