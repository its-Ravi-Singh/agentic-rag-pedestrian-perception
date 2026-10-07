import torch
from transformers import AutoImageProcessor, ViTForImageClassification
from peft import PeftConfig, PeftMixedModel
from peft.utils import set_peft_model_state_dict
from safetensors.torch import load_file

from apr.config import load_adapters, load_yaml
from apr.utils import get_device

class AdapterBank:
    def __init__(self, groups=["pedestrian_behavior", "scene_context"], device=None):
        if device is None:
            self.device = get_device()
        else:
            self.device = device
            
        cfg_yaml = load_yaml("adapters.yaml")
        base_mdl = cfg_yaml['base_model']
        
        self.processor = AutoImageProcessor.from_pretrained(base_mdl)

        # load base and strip out the original head
        base_vit = ViTForImageClassification.from_pretrained(base_mdl, num_labels=2)
        base_vit.classifier = torch.nn.Identity()

        self.all_adapters = {}
        for g in groups:
            res = load_adapters(g)
            for k, v in res.items():
                self.all_adapters[k] = v

        self.model = None
        self.class_heads = {}
        
        for n, info in self.all_adapters.items():
            conf = PeftConfig.from_pretrained(info['path'])
            
            # fix shape mismatch crash from diff classes
            conf.modules_to_save = None
            conf.inference_mode = True
            
            p = str(info['path']) + "/adapter_model.safetensors"
            weights = load_file(p)

            # get the head manually
            head_w = {}
            for key in list(weights.keys()):
                if "classifier" in key:
                    new_k = key.split("classifier.")[-1]
                    head_w[new_k] = weights[key]
                    
            # shape is (out, in)
            out_sz = head_w["weight"].shape[0]
            in_sz = head_w["weight"].shape[1]
            
            hd = torch.nn.Linear(in_sz, out_sz)
            hd.load_state_dict(head_w)
            hd = hd.to(self.device)
            hd.eval()
            
            self.class_heads[n] = hd

            if self.model is None:
                self.model = PeftMixedModel(base_vit, conf, adapter_name=n)
            else:
                self.model.add_adapter(n, conf)
                
            # now get the rest of the weights to load into peft
            adp_weights = {}
            for k in weights:
                if "classifier" not in k:
                    adp_weights[k] = weights[k]
                    
            set_peft_model_state_dict(self.model, adp_weights, adapter_name=n)

        self.model.to(self.device)
        self.model.eval()

    def preprocess(self, imgs):
        out = self.processor(images=imgs, return_tensors='pt')
        return out['pixel_values'].to(self.device)

    @torch.no_grad()
    def predict(self, name, x):
        self.model.set_adapter(name)
        
        out = self.model(pixel_values=x)
        f = out.logits
        
        # pass thru specific head
        preds = self.class_heads[name](f)
        
        return preds.softmax(dim=-1), f

    def labels(self, name):
        return self.all_adapters[name]['id2label']
