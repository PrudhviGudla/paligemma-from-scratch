from PIL import Image
import torch
import fire # for command line argument parsing
from input_processor import PaliGemmaProcessor
from paligemma import PaliGemmaForConditionalGeneration, PaliGemmaConfig, KVCache, PaliGemmaForConditionalGeneration
from output_processor import PaliGemmaOutputProcessor
from transformers import AutoTokenizer
import json
import glob
from safetensors import safe_open
from typing import Tuple
import os

def load_hf_model(model_path: str, device: str) -> Tuple[PaliGemmaForConditionalGeneration, AutoTokenizer]:
    # Load the tokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_path, padding_side="right")
    assert tokenizer.padding_side == "right"

    # Find all the *.safetensors files
    safetensors_files = glob.glob(os.path.join(model_path, "*.safetensors"))

    # ... and load them one by one in the tensors dictionary
    tensors = {}
    for safetensors_file in safetensors_files:
        with safe_open(safetensors_file, framework="pt", device="cpu") as f:
            for key in f.keys():
                tensors[key] = f.get_tensor(key)

    # Load the model's config
    with open(os.path.join(model_path, "config.json"), "r") as f:
        model_config_file = json.load(f)
        config = PaliGemmaConfig(**model_config_file)

    # Create the model using the configuration
    model = PaliGemmaForConditionalGeneration(config).to(device)

    # Load the state dict of the model
    model.load_state_dict(tensors, strict=False)

    # Tie weights
    model.tie_weights()

    return (model, tokenizer)


def move_inputs_to_device(model_inputs: dict, device: str):
    model_inputs = {k: v.to(device) for k, v in model_inputs.items()}
    return model_inputs


def get_model_inputs(
    processor: PaliGemmaProcessor, prompt: str, image_file_path: str, device: str
):
    image = Image.open(image_file_path)
    images = [image]
    prompts = [prompt]
    model_inputs = processor(text=prompts, images=images)
    model_inputs = move_inputs_to_device(model_inputs, device)
    return model_inputs


def _sample_top_p(probs: torch.Tensor, p: float):
    # (B, vocab_size)
    probs_sort, probs_idx = torch.sort(probs, dim=-1, descending=True) # need probs_idx later when sampling since sorting changes the order
    # probs_idx is a tensor (same shape as probs_sort) where each entry holds the vocabulary index corresponding to that sorted position
    # (B, vocab_size)
    probs_sum = torch.cumsum(probs_sort, dim=-1)
    # (B, vocab_size)
    # (Substracting "probs_sort" shifts the cumulative sum by 1 position to the right before masking)
    # if probs_sort = [0.6, 0.3, 0.1] , Cumsum: [0.6, 0.9, 1.0]
    # Shifted: [0, 0.6, 0.9]
    mask = probs_sum - probs_sort > p # ensures inclusion up to and including the token that pushes the cumulative sum over p
    # Zero out all the probabilities of tokens that are not selected by the Top P
    probs_sort[mask] = 0.0
    # Redistribute the probabilities so that they sum up to 1.
    probs_sort.div_(probs_sort.sum(dim=-1, keepdim=True))
    # Sample a token (its index) from the top p distribution
    # It picks num_samples indices, with probability proportional to the entries in probs_sort
    next_token = torch.multinomial(probs_sort, num_samples=1) # Treats probs_sort as the weights of a multinomial/categorical distribution
    # Get the token position in the vocabulary corresponding to the sampled index
    # it looks up the vocabulary index in probs_idx at position next_token. This is your final token id for your model's output
    next_token = torch.gather(probs_idx, -1, next_token)
    return next_token


def test_inference(
    model: PaliGemmaForConditionalGeneration,
    processor: PaliGemmaProcessor,
    device: str,
    prompt: str,
    image_file_path: str,
    max_tokens_to_generate: int,
    temperature: float,
    top_p: float,
    do_sample: bool,
    output_processor: PaliGemmaOutputProcessor = None,
    output_path: str = None, 
):
    model_inputs = get_model_inputs(processor, prompt, image_file_path, device)
    input_ids = model_inputs["input_ids"]
    attention_mask = model_inputs["attention_mask"]
    pixel_values = model_inputs["pixel_values"]

    kv_cache = KVCache()

    # Generate tokens until you see the stop token
    stop_token = processor.tokenizer.eos_token_id
    generated_tokens = []

    for _ in range(max_tokens_to_generate):
        # Get the model outputs
        # TODO: remove the labels
        outputs = model(
            input_ids=input_ids,
            pixel_values=pixel_values,
            attention_mask=attention_mask,
            kv_cache=kv_cache,
        )
        kv_cache = outputs["kv_cache"]
        next_token_logits = outputs["logits"][:, -1, :]
        # Sample the next token
        if do_sample:
            # Apply temperature
            next_token_logits = torch.softmax(next_token_logits / temperature, dim=-1)
            next_token = _sample_top_p(next_token_logits, top_p)
        else:
            next_token = torch.argmax(next_token_logits, dim=-1, keepdim=True)
        assert next_token.size() == (1, 1)
        next_token = next_token.squeeze(0)  # Remove batch dimension
        generated_tokens.append(next_token)
        # Stop if the stop token has been generated
        if next_token.item() == stop_token:
            break
        # Append the next token to the input
        input_ids = next_token.unsqueeze(-1)
        attention_mask = torch.cat(
            [attention_mask, torch.ones((1, 1), device=input_ids.device)], dim=-1
        )

    generated_tokens = torch.cat(generated_tokens, dim=-1)
    # Decode the generated tokens
    decoded = processor.tokenizer.decode(generated_tokens, skip_special_tokens=False)

    print("\n" + "="*60)
    print("PROMPT:", prompt)
    print("OUTPUT:", decoded)
    print("="*60)

    # ADD THIS: Process output for visualization
    if output_processor:
        image = Image.open(image_file_path)
        output_processor.process_output(prompt, decoded, image, output_path)


def main(
    model_path: str = None,
    prompt: str = None,
    image_file_path: str = None,
    max_tokens_to_generate: int = 100,
    temperature: float = 0.8,
    top_p: float = 0.9,
    do_sample: bool = False,
    only_cpu: bool = False,
    vae_checkpoint: str = "vae-oid.npz",
    output_path: str = None
):
    device = "cpu"

    if not only_cpu:
        if torch.cuda.is_available():
            device = "cuda"
        elif torch.backends.mps.is_available():
            device = "mps"

    print("Device in use: ", device)

    print(f"Loading model")
    model, tokenizer = load_hf_model(model_path, device)
    model = model.to(device).eval()

    num_image_tokens = model.config.vision_config.num_image_tokens
    image_size = model.config.vision_config.image_size
    input_processor = PaliGemmaProcessor(tokenizer, num_image_tokens, image_size)

    output_processor = None
    if 'detect' in prompt.lower() or 'segment' in prompt.lower():
        print(f"\nInitializing output processor...")
        output_processor = PaliGemmaOutputProcessor(
            vae_checkpoint_path=vae_checkpoint if os.path.exists(vae_checkpoint) else None,
            device=device
        )

    print("Running inference")
    with torch.no_grad():
        test_inference(
            model,
            input_processor,
            device,
            prompt,
            image_file_path,
            max_tokens_to_generate,
            temperature,
            top_p,
            do_sample,
            output_processor,
            output_path, 
        )


if __name__ == "__main__":
    fire.Fire(main)