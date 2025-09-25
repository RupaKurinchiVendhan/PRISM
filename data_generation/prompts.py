
# Simple mapping from natural language keywords to distortion types
PROMPT_TO_DISTORTION = {
    # Cloud and lighting combinations
    "remove clouds and brighten": "cloud_low",
    "remove clouds brighten aerial": "cloud_low", 
    "cloud removal brightness": "cloud_low",
    "clouds dark aerial": "cloud_low",
    
    # Individual distortions
    "remove haze": "dehaze",
    "dehaze": "dehaze",
    "remove fog": "dehaze",
    "foggy": "dehaze",
    
    "remove blur": "deblur",
    "deblur": "deblur", 
    "blurry": "deblur",
    "motion blur": "deblur",
    "out of focus": "deblur",
    
    "remove rain": "derain",
    "derain": "derain",
    "rain streaks": "derain",
    "rainy": "derain",
    
    "remove snow": "desnow",
    "desnow": "desnow",
    "snowy": "desnow",
    "winter": "desnow",
    
    "brighten": "lowlight",
    "low light": "lowlight", 
    "dark": "lowlight",
    "underexposed": "lowlight",
    
    "remove clouds": "decloud",
    "decloud": "decloud",
    "cloudy sky": "decloud",
    
    "remove noise": "denoise",
    "denoise": "denoise",
    "noisy": "denoise",
    "grainy": "denoise",
    
    "remove moire": "demoire",
    "moire pattern": "demoire",
    "demoire": "demoire",
    
    "enhance face": "face",
    "face": "face",
    "portrait": "face",
    
    "correct exposure": "highlight",
    "overexposed": "highlight",
    "highlight": "highlight",
    
    "underwater": "unrefract",
    "unrefract": "unrefract",
    "water distortion": "unrefract",
    
    # Compound tasks available in unified checkpoint
    "remove haze and snow": "dehaze_desnow",
    "dehaze and desnow": "dehaze_desnow",
    "haze and snow": "dehaze_desnow",
    "fog and snow": "dehaze_desnow",

    "low contrast color": "low_contrast_color",
    "faded image": "low_contrast_color",
    "enhance the colors": "low_contrast_color",
    "enhance this underwater image": "low_contrast_color",
    
    "geometric distortion underwater": "unwarp_unrefract",
    "unwarp unrefract": "unwarp_unrefract",
    "underwater geometric": "unwarp_unrefract",
}


def map_prompt_to_distortion(prompt):
    """
    Map natural language prompt to distortion type.
    Uses simple keyword matching with priority for compound tasks.
    """
    prompt_lower = prompt.lower().strip()
    words = prompt_lower.split()
    
    # Priority order for compound tasks (check these first)
    if ("cloud" in words or "clouds" in words) and ("brighten" in words or "bright" in words or "dark" in words or "low light" in prompt_lower):
        return "cloud_low"
    
    if "low contrast" in prompt_lower or "faded" in words:
        return "low_contrast_color"
        
    if "underwater" in words and ("distortion" in words or "geometric" in words or "warp" in words):
        return "unwarp_unrefract"
        
    if ("haze" in words or "dehaze" in words) and ("snow" in words or "desnow" in words):
        return "dehaze_desnow"
    
    if ("blur" in words or "deblur" in words) and ("contrast" in words or "low" in words):
        return "deblur_contrast_low"
        
    if ("noise" in words or "denoise" in words) and ("contrast" in words or "low" in words):
        return "denoise_contrast_low"
        
    if ("superresolve" in words or "super resolution" in prompt_lower) and "noise" in words:
        return "superresolve_denoise"
    
    # Try exact phrase matches
    for keywords, distortion in PROMPT_TO_DISTORTION.items():
        if keywords in prompt_lower:
            return distortion
    
    # Individual task matching (fallback)
    if "haze" in words or "fog" in words or "dehaze" in words:
        return "dehaze_desnow"  # Use compound version if available
    elif "blur" in words or "deblur" in words:
        return "deblur_contrast_low"  # Use compound version
    elif "noise" in words or "grain" in words or "denoise" in words:
        return "denoise"
    elif "cloud" in words or "clouds" in words:
        return "decloud"
    elif "underwater" in words or "unrefract" in words:
        return "unrefract"
    elif "unwarp" in words or "warp" in words:
        return "unwarp"
    elif "defocus" in words or "focus" in words:
        return "defocus"
    elif "decolor" in words or "color" in words:
        return "decolor"
    
    return None