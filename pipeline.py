import torch
from diffusers import StableDiffusionXLControlNetPipeline, ControlNetModel, AutoencoderKL, UniPCMultistepScheduler
from diffusers import LCMLora, LoRA
from diffusers import StableDiffusionXLPipeline, UNet2DConditionModel  # Import SDXL Turbo components

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

    print("Initializing SDXL Turbo pipeline...")

    # Use StableDiffusionXLPipeline for SDXL Turbo
    pipe = StableDiffusionXLPipeline.from_pretrained(model_id, torch_dtype=torch.float16, variant="fp16", use_safetensors=True).to("cuda")

    # Load and fuse LoRA weights
    pipe.load_lora_weights(adapter_id)
    pipe.fuse_lora()

    # Add UNet2DConditionModel to the pipeline
    unet_id = "mhdang/dpo-sdxl-text2image-v1"
    unet = UNet2DConditionModel.from_pretrained(unet_id, subfolder="unet", torch_dtype=torch.float16)
    pipe.unet = unet

    # Add ControlNet to the pipeline
    pipe.add_controlnet("depth_controlnet", depth_controlnet)

    # Add LoRA and LCM to the pipeline
    pipe.add_residual_adapter("lcm_lora", LoRA.from_pretrained(adapter_id), adapter_weight=1.0)

    pipe.enable_model_cpu_offload()
    pipe.scheduler = UniPCMultistepScheduler.from_config(pipe.scheduler.config)
    pipe.enable_xformers_memory_efficient_attention()

def run_pipeline(image, positive_prompt, negative_prompt, seed):
    init()
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
image = run_pipeline(image=prompt, positive_prompt=prompt, negative_prompt="", seed=42)[0]
image = image.resize((512, 512))
image.save("output_image.png")
