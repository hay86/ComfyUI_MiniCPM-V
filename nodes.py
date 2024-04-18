import os
import torch
import folder_paths

from transformers import AutoTokenizer, AutoModel
from torchvision.transforms.v2 import ToPILImage

class D_MiniCPM_VQA:
    def __init__(self):
        self.model_checkpoint = None
        self.tokenizer = None
        self.model = None
        self.device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
        self.bf16_support = torch.cuda.is_available() and torch.cuda.get_device_capability(self.device)[0] >= 8

    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "image": ("IMAGE",),
                "text": ("STRING", {"default": '', "multiline": True}),
                "model": (["MiniCPM-V", "MiniCPM-V-2"],),
                "temperature": ("FLOAT", {"default": 0.7,})
            },
        }

    RETURN_TYPES = ("STRING",)
    FUNCTION = "inference"
    CATEGORY = "MiniCPM-V"

    def inference(self, image, text, model, temperature):
        model_id = f"openbmb/{model}"
        model_checkpoint = os.path.join(folder_paths.models_dir, 'prompt_generator', os.path.basename(model_id))

        if not os.path.exists(model_checkpoint):
            from huggingface_hub import snapshot_download
            snapshot_download(repo_id=model_id, local_dir=model_checkpoint, local_dir_use_symlinks=False)

        if self.model_checkpoint != model_checkpoint:
            self.model_checkpoint = model_checkpoint
            self.tokenizer = AutoTokenizer.from_pretrained(model_checkpoint, trust_remote_code=True)
            self.model = AutoModel.from_pretrained(model_checkpoint, trust_remote_code=True, torch_dtype=torch.bfloat16)
            self.model = self.model.to(self.device, dtype=torch.bfloat16 if self.bf16_support else torch.float16).eval()

        with torch.no_grad():
            image = ToPILImage()(image.permute([0,3,1,2])[0]).convert("RGB")

            result, context, _ = self.model.chat(
                image=image,
                msgs=[{'role': 'user', 'content': text}],
                context=None,
                tokenizer=self.tokenizer,
                sampling=True,
                temperature=temperature
            )
            return (result,)
        

NODE_CLASS_MAPPINGS = {
    "D_MiniCPM_VQA": D_MiniCPM_VQA,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "D_MiniCPM_VQA": "MiniCPM VQA",
}