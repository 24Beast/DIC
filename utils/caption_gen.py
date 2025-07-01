# Importing Libraries
import os
import torch
import pickle
import argparse
from PIL import Image

# Constant Initializaiton
DEVICE = "cuda:0" if torch.cuda.is_available() else "cpu"
VALID_TYPES = ["png", "jpg", "jpeg"]

# Argument Parsing
parser = argparse.ArgumentParser()
parser.add_argument(
    "--img_path", type=str, help="Path of folder containing images.", required=True
)
parser.add_argument(
    "--model_type",
    default="vit_gpt2",
    type=str,
    help="Type of model to be used for captioning.",
)
parser.add_argument("--save_dir", default="./")
args = parser.parse_args()

# Parameter Initialization
BASE_PATH = args.img_path
model_type = args.model_type
save_dir = args.save_dir
img_names = [
    item for item in os.listdir(BASE_PATH) if item.split(".")[-1] in VALID_TYPES
]
num_imgs = len(img_names)

# Model Initialization
if model_type == "vit_gpt2":
    model_name = "nlpconnect/vit-gpt2-image-captioning"
    use_pipeline = True
elif model_type == "blip":
    model_name = "Salesforce/blip-image-captioning-large"
    use_pipeline = True
elif model_type == "git_base":
    model_name = "microsoft/git-base-coco"
    use_pipeline = True
elif model_type == "florence":
    model_name = "microsoft/Florence-2-base"
    use_pipeline = False
elif model_type == "llava":
    model_name = "llava-hf/llava-1.5-7b-hf"
    use_pipeline = False
elif model_type == "bakllava":
    model_name = "llava-hf/bakLlava-v1-hf"
    use_pipeline = False


# Caption Generation
captions = []
if use_pipeline:
    from transformers import pipeline

    image_to_text = pipeline("image-to-text", model=model_name)
    for num, name in enumerate(img_names, 1):
        print(f"\rWorking on img num: {num}/{num_imgs}", end="")
        file_path = BASE_PATH + "/" + name
        img_id = int(name.split(".")[0])
        text = image_to_text(file_path)
        pred = text[0]["generated_text"]
        caption_obj = {"img_id": img_id, "pred": pred}
        captions.append(caption_obj)
else:
    from transformers import AutoProcessor

    if (model_type == "llava") or (model_type == "bakllava"):
        from transformers import LlavaForConditionalGeneration

        prompt = "USER:\n Describe what is going on in image? <image>\nASSISTANT:"
        image_file = "upload/00000230.jpg"

        model = LlavaForConditionalGeneration.from_pretrained(
            model_name,
            torch_dtype=torch.float16,
            # low_cpu_mem_usage=True,
        ).to(DEVICE)

        processor = AutoProcessor.from_pretrained(model_name)

        for num, name in enumerate(img_names, 1):
            print(f"\rWorking on img num: {num}/{num_imgs}", end="")
            file_path = BASE_PATH + "/" + name
            img_id = int(name.split(".")[0])
            img = Image.open(file_path)
            inputs = processor(img, prompt, return_tensors="pt").to(0, torch.float16)
            output = model.generate(**inputs, max_new_tokens=200, do_sample=False)
            pred = processor.decode(output[0], skip_special_tokens=True).split(
                "ASSISTANT: "
            )[1]
            caption_obj = {"img_id": img_id, "pred": pred}
            captions.append(caption_obj)

    elif model_type == "florence":
        from transformers import AutoModelForCausalLM

        model = AutoModelForCausalLM.from_pretrained(
            model_name, torch_dtype=torch.float16, trust_remote_code=True
        ).to(DEVICE)
        processor = AutoProcessor.from_pretrained(model_name, trust_remote_code=True)
        prompt = "<CAPTION>"

        for num, name in enumerate(img_names, 1):
            print(f"\rWorking on img num: {num}/{num_imgs}", end="")
            file_path = BASE_PATH + "/" + name
            img_id = int(name.split(".")[0])
            img = Image.open(file_path)
            inputs = processor(text=prompt, images=img, return_tensors="pt").to(
                DEVICE, torch.float16
            )
            generated_ids = model.generate(
                input_ids=inputs["input_ids"],
                pixel_values=inputs["pixel_values"],
                max_new_tokens=1024,
                num_beams=3,
            )
            generated_text = processor.batch_decode(
                generated_ids, skip_special_tokens=False
            )[0]
            pred = processor.post_process_generation(
                generated_text, task=prompt, image_size=(img.width, img.height)
            )
            caption_obj = {"img_id": img_id, "pred": pred}
            captions.append(caption_obj)


# Saving results
if not (os.path.isdir(save_dir)):
    os.makedirs(save_dir)

out_fname = save_dir + f"/{model_type}.pkl"

with open(out_fname, "wb") as f:
    pickle.dump(captions, f)
