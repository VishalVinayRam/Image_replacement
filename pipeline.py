import torch
from diffusers import StableDiffusionXLControlNetPipeline, ControlNetModel, AutoencoderKL, UniPCMultistepScheduler
from diffusers import LCMScheduler, AutoPipelineForText2Image, LCMLora, LoRA

model_id = "stabilityai/stable-diffusion-xl-base-1.0"
adapter_id = "latent-consistency/lcm-lora-sdxl"

# Initializing AutoPipelineForText2Image with LCMScheduler and LCM LoRA
auto_pipe = AutoPipelineForText2Image.from_pretrained(model_id, torch_dtype=torch.float16, variant="fp16")
auto_pipe.scheduler = LCMScheduler.from_config(auto_pipe.scheduler.config)
auto_pipe.to("cuda")
auto_pipe.load_lora_weights(adapter_id)
auto_pipe.fuse_lora()

# Initializing StableDiffusionControlNetPipeline
controlnet_pipe = None

def init_controlnet_pipeline():
    global controlnet_pipe

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

    print("Initializing SDXL ControlNet pipeline...")

    controlnet_pipe = StableDiffusionControlNetPipeline.from_pretrained(
        model_id,
        controlnet=[depth_controlnet],
        vae=vae,
        variant="fp16",
        use_safetensors=True,
        torch_dtype=torch.float16
    ).to("cuda")

    controlnet_pipe.enable_model_cpu_offload()
    controlnet_pipe.scheduler = UniPCMultistepScheduler.from_config(controlnet_pipe.scheduler.config)
    controlnet_pipe.enable_xformers_memory_efficient_attention()

# Combined initialization function
def init():
    global auto_pipe, controlnet_pipe

    # Initializing AutoPipelineForText2Image
    print("Initializing AutoPipelineForText2Image...")

    auto_pipe = AutoPipelineForText2Image.from_pretrained(model_id, torch_dtype=torch.float16, variant="fp16")
    auto_pipe.scheduler = LCMScheduler.from_config(auto_pipe.scheduler.config)
    auto_pipe.to("cuda")

    auto_pipe.load_lora_weights(adapter_id)
    auto_pipe.fuse_lora()

    # Initializing StableDiffusionControlNetPipeline
    init_controlnet_pipeline()

def run_combined_pipeline(image, positive_prompt, negative_prompt, seed):
    global auto_pipe, controlnet_pipe

    init()

    if seed == -1:
        print("Using random seed")
        generator = None
    else:
        print("Using seed:", seed)
        generator = torch.manual_seed(seed)

    # Running AutoPipelineForText2Image
    auto_images = auto_pipe(
        prompt=positive_prompt,
        negative_prompt=negative_prompt,
        num_inference_steps=30,
        num_images_per_prompt=4,
        controlnet_conditioning_scale=0.65,
        guidance_scale=10.0,
        generator=generator,
        image=image
    ).images

    # Running StableDiffusionControlNetPipeline
    controlnet_images = controlnet_pipe(
        prompt=positive_prompt,
        negative_prompt=negative_prompt,
        num_inference_steps=30,
        num_images_per_prompt=4,
        controlnet_conditioning_scale=0.65,
        guidance_scale=10.0,
        generator=generator,
        image=image
    ).images

    return auto_images, controlnet_images

prompt = "Self-portrait oil painting, a beautiful cyborg with golden hair, 8k"
auto_images, controlnet_images = run_combined_pipeline(image=prompt, positive_prompt=prompt, negative_prompt="", seed=42)

# Access generated images from AutoPipelineForText2Image and StableDiffusionControlNetPipeline
auto_generated_image = auto_images[0].resize((512, 512))
controlnet_generated_image = controlnet_images[0].resize((512, 512))

# Save images
auto_generated_image.save("auto_generated_image.png")
controlnet_generated_image.save("controlnet_generated_image.png")
