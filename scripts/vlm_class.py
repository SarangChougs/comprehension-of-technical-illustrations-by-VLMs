## Class for loading different models

from app.qwen import load_qwen
from app.internVL import load_internVL
from app.ovis import load_ovis
from app.kimi import load_kimi

class VisionLanguageModel:
    def __init__(self, model_name):
        self.processor = None
        self.model = None
        self.pipe = None
        self.model_name = model_name
        self.response = None
        self.text_tokenizer, self.visual_tokenizer = None, None
        if "Qwen" in model_name:
            self.loading_function, self.inference_function = load_qwen(model_name)
        elif "Intern" in model_name:
            self.loading_function, self.inference_function = load_internVL(model_name)
        elif "Ovis" in model_name:
            self.loading_function, self.inference_function = load_ovis()
        elif "Kimi" in model_name:
            self.loading_function, self.inference_function = load_kimi(model_name)
        else:
            print(f"{self.model_name} Model not found in VLM class")
    
    def load(self):
        if "Qwen" in self.model_name or "Kimi" in self.model_name :
            print(f"Loading {self.model_name}")
            self.model, self.processor = self.loading_function()
        if "Intern" in self.model_name:
            print(f"Loading {self.model_name}")
            self.pipe = self.loading_function(self.model_name)
        if "Ovis" in self.model_name:
            self.model, self.text_tokenizer, self.visual_tokenizer = self.loading_function(self.model_name)

    def run(self, text, image):
        if "Qwen" in self.model_name or "Kimi" in self.model_name:
            self.response = self.inference_function(self.model, self.processor, text, image)
        if "Intern" in self.model_name:
            self.response = self.inference_function(self.pipe, text, image)
        if "Ovis" in self.model_name:
            self.response = self.inference_function(self.model, self.text_tokenizer, self.visual_tokenizer, text, image)
        return self.response