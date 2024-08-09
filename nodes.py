import os
import torch
import folder_paths

from transformers import AutoTokenizer, AutoModel
from torchvision.transforms.v2 import ToPILImage

to_image = ToPILImage()

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
                "model": (["MiniCPM-V", "MiniCPM-V-2", "MiniCPM-Llama3-V-2_5", "MiniCPM-Llama3-V-2_5-int4", "MiniCPM-V-2_6", "MiniCPM-V-2_6-int4"],),
                "temperature": ("FLOAT", {"default": 0.7,}),
                "video_max_num_frames": ("INT", {"default": 0,}),
            },
        }

    RETURN_TYPES = ("STRING",)
    FUNCTION = "inference"
    CATEGORY = "MiniCPM-V"

    def inference(self, image, text, model, temperature, video_max_num_frames):
        model_id = f"openbmb/{model}"
        model_checkpoint = os.path.join(folder_paths.models_dir, 'prompt_generator', os.path.basename(model_id))

        if not os.path.exists(model_checkpoint):
            from huggingface_hub import snapshot_download
            snapshot_download(repo_id=model_id, local_dir=model_checkpoint, local_dir_use_symlinks=False)

        if self.model_checkpoint != model_checkpoint:
            self.model_checkpoint = model_checkpoint
            self.tokenizer = AutoTokenizer.from_pretrained(model_checkpoint, trust_remote_code=True)

            if model in ["MiniCPM-V", "MiniCPM-V-2"]:
                self.model = AutoModel.from_pretrained(model_checkpoint, trust_remote_code=True, torch_dtype=torch.bfloat16)
                self.model = self.model.to(self.device, dtype=torch.bfloat16 if self.bf16_support else torch.float16)
            elif model in ["MiniCPM-Llama3-V-2_5"]:
                self.model = AutoModel.from_pretrained(model_checkpoint, trust_remote_code=True, torch_dtype=torch.float16)
                self.model = self.model.to(self.device)
            elif model in ["MiniCPM-V-2_6"]:
                self.model = AutoModel.from_pretrained(model_checkpoint, trust_remote_code=True, attn_implementation='sdpa', torch_dtype=torch.bfloat16)
                self.model = self.model.to(self.device, dtype=torch.bfloat16 if self.bf16_support else torch.float16)
            elif model in ["MiniCPM-Llama3-V-2_5-int4", "MiniCPM-V-2_6-int4"]:
                self.model = AutoModel.from_pretrained(model_checkpoint, trust_remote_code=True)

        self.model.eval()
        with torch.no_grad():
            if model in ["MiniCPM-V", "MiniCPM-V-2"]:
                image = to_image(image.permute([0,3,1,2])[0]).convert("RGB")
                result = self.model.chat(
                    image=image,
                    msgs=[{'role': 'user', 'content': text}],
                    context=None,
                    tokenizer=self.tokenizer,
                    sampling=True,
                    temperature=temperature
                )[0]
            elif model in ["MiniCPM-Llama3-V-2_5", "MiniCPM-Llama3-V-2_5-int4"]:
                image = to_image(image.permute([0,3,1,2])[0]).convert("RGB")
                result = self.model.chat(
                    image=image,
                    msgs=[{'role': 'user', 'content': text}],
                    tokenizer=self.tokenizer,
                    sampling=True,
                    temperature=temperature
                )
            elif model in ["MiniCPM-V-2_6", "MiniCPM-V-2_6-int4"]:
                images = image.permute([0,3,1,2])
                images = [to_image(img).convert("RGB") for img in images]

                params = {"use_image_id": False, "max_slice_nums": 1} if video_max_num_frames > 0 else {}

                def uniform_sample(frames, max_num):
                    if len(frames) <= max_num or max_num <= 0:
                        return frames
                    gap = len(frames) / max_num
                    return [frames[int(i * gap + gap / 2)] for i in range(max_num)]

                sampled_images = uniform_sample(images, video_max_num_frames)

                result = self.model.chat(
                    image=None,
                    msgs=[{'role': 'user', 'content': sampled_images + [text]}],
                    tokenizer=self.tokenizer,
                    **params
                )
            return (result,)
        

NODE_CLASS_MAPPINGS = {
    "D_MiniCPM_VQA": D_MiniCPM_VQA,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "D_MiniCPM_VQA": "MiniCPM VQA",
}