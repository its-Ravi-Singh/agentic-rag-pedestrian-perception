import torch
from peft import PeftConfig, PeftMixedModel
from peft.utils import set_peft_model_state_dict
from safetensors.torch import load_file
from transformers import AutoImageProcessor, ViTForImageClassification

from apr.config import load_adapters, load_yaml
from apr.utils import get_device


class AdapterBank:
    # one vit with all 8 adapters on it, switch using adapter name

    def __init__(self, groups=("pedestrian_behavior", "scene_context"), device=None):
        self.device = device or get_device()
        base_name = load_yaml("adapters.yaml")["base_model"]
        self.processor = AutoImageProcessor.from_pretrained(base_name)

        base = ViTForImageClassification.from_pretrained(base_name, num_labels=2)
        base.classifier = torch.nn.Identity()

        self.adapters = {}
        for g in groups:
            self.adapters.update(load_adapters(g))

        self.model = None
        self.heads = {}
        for name, info in self.adapters.items():
            cfg = PeftConfig.from_pretrained(info["path"])
            # heads have diffrent number of classes so we load them seperately
            cfg.modules_to_save = None
            cfg.inference_mode = True
            w = load_file(info["path"] / "adapter_model.safetensors")

            head_w = {k.split("classifier.")[1]: v for k, v in w.items() if "classifier" in k}
            head = torch.nn.Linear(head_w["weight"].shape[1], head_w["weight"].shape[0])
            head.load_state_dict(head_w)
            self.heads[name] = head.to(self.device).eval()

            if self.model is None:
                self.model = PeftMixedModel(base, cfg, adapter_name=name)
            else:
                self.model.add_adapter(name, cfg)
            adapter_w = {k: v for k, v in w.items() if "classifier" not in k}
            set_peft_model_state_dict(self.model, adapter_w, adapter_name=name)

        self.model.to(self.device).eval()

    def preprocess(self, images):
        # images is list of rgb numpy arrays
        return self.processor(images=images, return_tensors="pt")["pixel_values"].to(self.device)

    @torch.no_grad()
    def predict(self, name, pixel_values):
        # returns probs and the cls features (used later for router)
        self.model.set_adapter(name)
        feats = self.model(pixel_values=pixel_values).logits
        probs = self.heads[name](feats).softmax(dim=-1)
        return probs, feats

    def labels(self, name):
        return self.adapters[name]["id2label"]
