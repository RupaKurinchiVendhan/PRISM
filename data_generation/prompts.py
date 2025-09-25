
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
