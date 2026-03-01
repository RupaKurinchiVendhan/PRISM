
# Simple mapping from natural language keywords to distortion types
PROMPT_TO_DISTORTION = {
    # Cloud and lighting combinations
    "remove clouds and brighten": "cloud_low",
    "remove clouds brighten aerial": "cloud_low", 
    "cloud removal brightness": "cloud_low",
    "clouds dark aerial": "cloud_low",
    
    # Geometric distortions
    "remove blur": "blur",
    "deblur": "blur", 
    "blurry": "blur",
    "motion blur": "blur",
    "remove motion blur": "blur",
    "fix blur": "blur",
    
    "out of focus": "defocus",
    "defocus": "defocus",
    "defocus blur": "defocus",
    "unfocused": "defocus",
    "fix focus": "defocus",
    
    "warping": "warp",
    "dewarp": "warp",
    "remove warping": "warp",
    "fix distortion": "warp",
    "geometric distortion": "warp",
    
    "refraction": "refract",
    "derefract": "refract",
    "remove refraction": "refract",
    "water distortion": "refract",
    "underwater": "refract",
    "unrefract": "refract",
    
    # Photometric degradations
    "contrast": "contrast",
    "fix contrast": "contrast",
    "low contrast": "contrast",
    "improve contrast": "contrast",
    "enhance contrast": "contrast",
    
    "color shifts": "color",
    "color": "color",
    "fix colors": "color",
    "color correction": "color",
    "white balance": "color",
    "color cast": "color",
    
    "brightness": "brightness",
    "fix brightness": "brightness",
    "adjust brightness": "brightness",
    "too bright": "brightness",
    "too dark": "brightness",
    
    "brighten": "low",
    "low light": "low",
    "lowlight": "low",
    "dark": "low",
    "underexposed": "low",
    "enhance lighting": "low",
    "improve lighting": "low",
    
    # Occlusions
    "remove haze": "haze",
    "dehaze": "haze",
    "remove fog": "haze",
    "foggy": "haze",
    "hazy": "haze",
    "clear haze": "haze",
    
    "remove rain": "rain",
    "derain": "rain",
    "rain streaks": "rain",
    "rainy": "rain",
    "rain drops": "rain",
    "rainfall": "rain",
    
    "remove snow": "snow",
    "desnow": "snow",
    "snowy": "snow",
    "winter": "snow",
    "snow removal": "snow",
    
    "remove clouds": "clouds",
    "decloud": "clouds",
    "cloudy sky": "clouds",
    "cloud removal": "clouds",
    "clouds": "clouds",
    
    # Noise and compression
    "remove noise": "noise",
    "denoise": "noise",
    "noisy": "noise",
    "grainy": "noise",
    "additive noise": "noise",
    "noise reduction": "noise",
    
    "compression": "compress",
    "decompress": "compress",
    "jpeg artifacts": "compress",
    "compression artifacts": "compress",
    "blocky": "compress",
    "fix compression": "compress",
    "resolution": "compress",
    "superresolve": "compress",
    "super-resolve": "compress",
    
    # Other
    "enhance face": "face",
    "face": "face",
    "portrait": "face",
    
    "correct exposure": "highlight",
    "overexposed": "highlight",
    "highlight": "highlight",
    
    # Compound tasks available in unified checkpoint
    "remove haze and snow": "haze_snow",
    "dehaze and desnow": "haze_snow",
    "haze and snow": "haze_snow",
    "fog and snow": "haze_snow",

    "low contrast color": "low_contrast_color",
    "faded image": "low_contrast_color",
    "enhance the colors": "low_contrast_color",
    "enhance this underwater image": "low_contrast_color",
    
    "geometric distortion underwater": "unwarp_unrefract",
    "unwarp unrefract": "unwarp_unrefract",
    "underwater geometric": "unwarp_unrefract",
}


def embed(prompt):
    """
    Map natural language prompt to distortion type.
    Uses simple keyword matching with priority for compound tasks.
    """
    prompt_lower = prompt.lower().strip()
    words = prompt_lower.split()
    
    # Priority order for compound tasks (check these first)
    # Check for brighten/lowlight + haze + snow combination
    has_brighten = any(word in prompt_lower for word in ["brighten", "bright", "dark", "low light", "underexposed", "lowlight"])
    has_haze = any(word in prompt_lower for word in ["haze", "dehaze", "fog", "foggy"])
    has_snow = any(word in prompt_lower for word in ["snow", "desnow", "snowy"])
    
    if has_brighten and has_haze and has_snow:
        return "low_haze_snow"
    
    if ("cloud" in words or "clouds" in words) and ("brighten" in words or "bright" in words or "dark" in words or "low light" in prompt_lower):
        return "cloud_low"
    
    if "low contrast" in prompt_lower or "faded" in words:
        return "low_contrast_color"
        
    if "underwater" in words and ("distortion" in words or "geometric" in words or "warp" in words):
        return "unwarp_unrefract"
        
    if ("haze" in words or "dehaze" in words) and ("snow" in words or "desnow" in words):
        return "haze_snow"
    
    if ("blur" in words or "deblur" in words) and ("contrast" in words or "low" in words):
        return "deblur_contrast_low"
        
    if ("noise" in words or "denoise" in words) and ("contrast" in words or "low" in words):
        return "denoise_contrast_low"
        
    if ("superresolve" in words or "super resolution" in prompt_lower) and "noise" in words:
        return "compress_noise"
    
    # Try exact phrase matches
    for keywords, distortion in PROMPT_TO_DISTORTION.items():
        if keywords in prompt_lower:
            return distortion
    
    # Individual task matching (fallback) - check for single distortion types
    # Geometric distortions
    if "motion blur" in prompt_lower or ("blur" in words and "motion" in words):
        return "blur"
    elif "defocus" in words or ("out" in words and "focus" in words):
        return "focus"
    elif "warp" in words or "warping" in words or "dewarp" in words:
        return "warp"
    elif "refract" in words or "derefract" in words:
        return "refract"
    elif "blur" in words or "deblur" in words or "blurry" in words:
        return "blur"
    
    # Photometric degradations
    elif "contrast" in words:
        return "contrast"
    elif "color" in words and any(w in words for w in ["shift", "cast", "correction", "fix"]):
        return "color"
    elif "brightness" in words:
        return "brightness"
    elif any(w in words for w in ["brighten", "lowlight", "underexposed"]) or "low light" in prompt_lower:
        return "lowlight"
    
    # Occlusions
    elif "haze" in words or "fog" in words or "dehaze" in words or "foggy" in words or "hazy" in words:
        return "haze"
    elif "rain" in words or "derain" in words or "rainy" in words:
        return "rain"
    elif "snow" in words or "desnow" in words or "snowy" in words:
        return "snow"
    elif "cloud" in words or "clouds" in words or "decloud" in words:
        return "clouds"
    
    # Noise and compression
    elif "noise" in words or "grain" in words or "denoise" in words or "noisy" in words or "grainy" in words:
        return "denoise"
    elif "compress" in words or "decompress" in words or "jpeg" in words or "artifact" in words or "blocky" in words:
        return "comppress"
    
    # Other
    elif "underwater" in words:
        return "refract"
    
    return None