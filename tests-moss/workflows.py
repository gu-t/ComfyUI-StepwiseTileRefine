"""API-format workflow builders for the MOSS tests."""

MOSS_UNET = "MOSS\\moss_image_sft_int8_convrot.safetensors"
MOSS_TE = "qwen3.5_9b_w4a8_gptq.safetensors"
MOSS_VAE = "flux2-vae.safetensors"

NEGATIVE = "nsfw, worst quality, bad anatomy, watermark, jpeg artifact"


class Graph:
    def __init__(self):
        self.nodes = {}
        self._n = 0

    def add(self, class_type, **inputs):
        self._n += 1
        nid = str(self._n)
        self.nodes[nid] = {"class_type": class_type, "inputs": inputs}
        return nid

    @staticmethod
    def out(nid, slot=0):
        return [nid, slot]


def moss_loaders(g, unet=MOSS_UNET):
    model = g.add("UNETLoader", unet_name=unet, weight_dtype="default")
    clip = g.add("CLIPLoader", clip_name=MOSS_TE, type="moss_image", device="default")
    vae = g.add("VAELoader", vae_name=MOSS_VAE)
    return model, clip, vae


def encode(g, clip, text):
    return g.add("CLIPTextEncode", clip=[clip, 0], text=text)


def moss_t2i(prompt, width, height, seed, steps=25, cfg=3.0, sampler="er_sde", scheduler="simple", prefix="baseline/t2i"):
    g = Graph()
    model, clip, vae = moss_loaders(g)
    pos = encode(g, clip, prompt)
    neg = encode(g, clip, NEGATIVE)
    lat = g.add("EmptyFlux2LatentImage", width=width, height=height, batch_size=1)
    ks = g.add("KSampler", model=[model, 0], positive=[pos, 0], negative=[neg, 0], latent_image=[lat, 0],
               seed=seed, steps=steps, cfg=cfg, sampler_name=sampler, scheduler=scheduler, denoise=1.0)
    dec = g.add("VAEDecode", samples=[ks, 0], vae=[vae, 0])
    g.add("SaveImage", images=[dec, 0], filename_prefix=prefix)
    return g.nodes


def _sampling(g, model, clip, prompt, cfg, sampler, seed):
    pos = encode(g, clip, prompt)
    neg = encode(g, clip, NEGATIVE)
    guider = g.add("CFGGuider", model=[model, 0], positive=[pos, 0], negative=[neg, 0], cfg=cfg)
    samp = g.add("KSamplerSelect", sampler_name=sampler)
    noise = g.add("RandomNoise", noise_seed=seed)
    return pos, neg, guider, samp, noise


def stepwise(image_name, prompt, upscale_by, upscale_steps=0, seed=1, steps=25, cfg=3.0, sampler="er_sde",
             scheduler="simple", denoise_first=0.45, denoise_last=0.22, anchor=32, overlap=32, max_tile=0,
             denoise_mode="denoise", steps_mode="steps per pass", output_size="exact, crop edges",
             auto_step_factor=1.5, tile_mode="tile by tile", tile_size="always full",
             plan_only=False, upscale_model=None, prefix="stepwise/out", save_steps=True,
             oversample=1.0, oversample_mode="constant", upscale_method="lanczos", downscale_method="area"):
    # The defaults keep the settings the measured results were made with (25 steps per pass).
    g = Graph()
    model, clip, vae = moss_loaders(g)
    _, _, guider, samp, noise = _sampling(g, model, clip, prompt, cfg, sampler, seed)
    img = g.add("LoadImage", image=image_name)
    inputs = dict(image=[img, 0], guider=[guider, 0], sampler=[samp, 0], noise=[noise, 0], vae=[vae, 0],
                  scheduler=scheduler, steps=steps, steps_mode=steps_mode, upscale_by=upscale_by,
                  output_size=output_size, upscale_steps=upscale_steps, auto_step_factor=auto_step_factor,
                  max_tile_width=max_tile, max_tile_height=max_tile, tile_padding=anchor, tile_overlap=overlap,
                  tile_mode=tile_mode, tile_size=tile_size,
                  denoise_first=denoise_first, denoise_last=denoise_last, denoise_mode=denoise_mode,
                  plan_only=plan_only, oversample=oversample,
                  oversample_mode=oversample_mode, upscale_method=upscale_method, downscale_method=downscale_method)
    if upscale_model:
        um = g.add("UpscaleModelLoader", model_name=upscale_model)
        inputs["upscale_model"] = [um, 0]
    node = g.add("StepwiseTileUpscale", **inputs)
    g.add("PreviewAny", source=[node, 2])
    if not plan_only:
        g.add("SaveImage", images=[node, 0], filename_prefix=prefix)
        if save_steps:
            g.add("SaveImage", images=[node, 1], filename_prefix=prefix + "_step")
    return g.nodes


def usdu(image_name, prompt, upscale_by, seed=1, steps=25, cfg=3.0, sampler="er_sde", scheduler="simple",
         denoise=0.35, tile=1024, padding=64, prefix="usdu/out", passes=1, denoises=None):
    # One or more chained Ultimate SD Upscale (custom sample, lanczos) passes, upscale_by split evenly.
    g = Graph()
    model, clip, vae = moss_loaders(g)
    pos = encode(g, clip, prompt)
    neg = encode(g, clip, NEGATIVE)
    cur = g.add("LoadImage", image=image_name)
    per = upscale_by ** (1.0 / passes)
    denoises = denoises or [denoise] * passes
    for i in range(passes):
        cur = g.add("UltimateSDUpscaleCustomSample", image=[cur, 0], model=[model, 0], positive=[pos, 0],
                    negative=[neg, 0], vae=[vae, 0], upscale_by=per, seed=seed + i, steps=steps, cfg=cfg,
                    sampler_name=sampler, scheduler=scheduler, denoise=denoises[i], mode_type="Linear",
                    tile_width=tile, tile_height=tile, mask_blur=8, tile_padding=padding,
                    seam_fix_mode="None", seam_fix_denoise=1.0, seam_fix_width=64, seam_fix_mask_blur=8,
                    seam_fix_padding=16, force_uniform_tiles=True, tiled_decode=False, batch_size=1)
    g.add("SaveImage", images=[cur, 0], filename_prefix=prefix)
    return g.nodes


def vae_roundtrip(image_name, factor, prefix="probe/roundtrip"):
    # Lanczos enlargement, then VAE encode + decode with no sampling: isolates the VAE's own grid.
    g = Graph()
    vae = g.add("VAELoader", vae_name=MOSS_VAE)
    img = g.add("LoadImage", image=image_name)
    up = g.add("ImageScaleBy", image=[img, 0], upscale_method="lanczos", scale_by=factor)
    enc = g.add("VAEEncode", pixels=[up, 0], vae=[vae, 0])
    dec = g.add("VAEDecode", samples=[enc, 0], vae=[vae, 0])
    g.add("SaveImage", images=[dec, 0], filename_prefix=prefix)
    g.add("SaveImage", images=[up, 0], filename_prefix=prefix + "_lanczos")
    return g.nodes


def whole_image_refine(image_name, prompt, factor, denoise, seed=1, steps=25, cfg=3.0, prefix="probe/whole"):
    # Lanczos enlargement and ONE untiled MOSS img2img pass over the whole image (no tiling).
    g = Graph()
    model, clip, vae = moss_loaders(g)
    pos = encode(g, clip, prompt)
    neg = encode(g, clip, NEGATIVE)
    img = g.add("LoadImage", image=image_name)
    up = g.add("ImageScaleBy", image=[img, 0], upscale_method="lanczos", scale_by=factor)
    enc = g.add("VAEEncode", pixels=[up, 0], vae=[vae, 0])
    ks = g.add("KSampler", model=[model, 0], positive=[pos, 0], negative=[neg, 0], latent_image=[enc, 0],
               seed=seed, steps=steps, cfg=cfg, sampler_name="er_sde", scheduler="simple", denoise=denoise)
    dec = g.add("VAEDecode", samples=[ks, 0], vae=[vae, 0])
    g.add("SaveImage", images=[dec, 0], filename_prefix=prefix)
    return g.nodes
