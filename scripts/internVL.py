## OpenGVLab/InternVL2_5-8B-AWQ
## Script to containing functions to load and run Qwen VLM models

from lmdeploy import pipeline, TurbomindEngineConfig, GenerationConfig, ChatTemplateConfig
from lmdeploy.vl import load_image
import nest_asyncio
nest_asyncio.apply()

## Function to load the model
def load_model(model):
    #model = 'OpenGVLab/InternVL2_5-8B-AWQ'
    print(f"*** Loading model : {model} ***")
    if 'AWQ' in model:
        print(f"*** Loading AWQ quantised model ***")
        engine_config = TurbomindEngineConfig(model_format='awq')
        pipe = pipeline(model, backend_config=engine_config, chat_template_config=ChatTemplateConfig(model_name='internvl2_5'))
    else:
        pipe = pipeline(model, chat_template_config=ChatTemplateConfig(model_name='internvl2_5'))

    return pipe

## Function to run model
def run_model(pipe, prompt, image):
    print("*** Running Inference ***")
    response = pipe((prompt, image), gen_config=GenerationConfig(
                    do_sample = False))
    return response.text

## Function to load different model versions 
def load_internVL(model_name):
    loading_function, inference_function = None, None
    
    loading_function = load_model
    inference_function = run_model

    return loading_function, inference_function

if __name__ == "__main__":
    prompt = "Describe this image."
    image = 'https://raw.githubusercontent.com/open-mmlab/mmdeploy/main/tests/data/tiger.jpeg'
    models = {
        '26b-awq': 'OpenGVLab/InternVL2_5-26B-AWQ',
        'v2.5-8b': 'OpenGVLab/InternVL2_5-8B-MPO',
        'v3-14b': 'OpenGVLab/InternVL3-14B',
        'v3-14b-awq': 'OpenGVLab/InternVL3-14B-AWQ',
        'v3-8b': 'OpenGVLab/InternVL3-8B',
        'v3-9b': 'OpenGVLab/InternVL3-9B'
    }
    pipe = load_model(models['v3-9b'])
    output = run_model(pipe, prompt, image)
    print(output)