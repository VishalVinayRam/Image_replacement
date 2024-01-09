import torch
from diffusers import StableDiffusionXLControlNetPipeline, ControlNetModel, AutoencoderKL, UniPCMultistepScheduler
from diffusers import LCMLora, LoRA

model_id = "stabilityai/stable-diffusion-xl-base-1.0"
adapter_id = "latent-consistency/lcm-lora-sdxl"

pipe = None

def init():
    global pipe

    print("Initializing depth ControlNet...")

    depth_controlnet = ControlNetModel.from_pretrained(
        "diffusers/controlnet-depth-sdxl-1.0",
        use_safetensors=True,
        torch_dtype=torch.float16
    ).to("cuda")

    print("Initializing autoencoder...")

    vae = AutoencoderKL.from_pretrained(
        "madebyollin/sdxl-vae-fp16-fix",
        torch_dtype=torch.float16,
    ).to("cuda")

    print("Initializing LCM LoRA...")

    lcm_lora = LCMLora.from_pretrained(adapter_id).to("cuda")

    print("Initializing SDXL pipeline...")

    pipe = StableDiffusionXLControlNetPipeline.from_pretrained(
        model_id,
        controlnet=[depth_controlnet],
        vae=vae,
        variant="fp16",
        use_safetensors=True,
        torch_dtype=torch.float16
    ).to("cuda")

    # Add LoRA and LCM to the pipeline
    pipe.add_residual_adapter("lcm_lora", LoRA.from_pretrained(adapter_id), adapter_weight=1.0)
    
    pipe.enable_model_cpu_offload()
    pipe.scheduler = UniPCMultistepScheduler.from_config(pipe.scheduler.config)
    pipe.enable_xformers_memory_efficient_attention()

    # Load and fuse LoRA weights
    pipe.load_lora_weights(adapter_id)
    pipe.fuse_lora()

def run_pipeline(image, positive_prompt, negative_prompt, seed):
    if seed == -1:
        print("Using random seed")
        generator = None
    else:
        print("Using seed:", seed)
        generator = torch.manual_seed(seed)

    images = pipe(
        prompt=positive_prompt,
        negative_prompt=negative_prompt,
        num_inference_steps=30,
        num_images_per_prompt=4,
        controlnet_conditioning_scale=0.65,
        guidance_scale=10.0,
        generator=generator,
        image=image
    ).images

    return images

prompt = "Self-portrait oil painting, a beautiful cyborg with golden hair, 8k"
image = pipe(prompt=prompt, num_inference_steps=4, guidance_scale=0).images[0]
