"""
Natural Language Task Selector for Image Restoration

This module provides functionality to parse natural language prompts and map them
to specific restoration tasks based on keyword matching and semantic understanding.
"""

import re
import json
from typing import Dict, List, Set, Tuple, Optional
from dataclasses import dataclass
from pathlib import Path

@dataclass
class TaskMapping:
    """Data structure for task mapping information"""
    task_name: str
    keywords: Set[str]
    checkpoint_path: str
    description: str
    priority: int = 1  # Higher priority tasks are selected first in conflicts

class TaskSelector:
    """
    Natural language task selector for image restoration.
    
    Maps natural language prompts to specific restoration tasks based on
    keyword matching and semantic understanding.
    """
    
    def __init__(self, config_path: Optional[str] = None):
        """
        Initialize the TaskSelector.
        
        Args:
            config_path: Path to task configuration JSON file
        """
        self.task_mappings: Dict[str, TaskMapping] = {}
        self.compound_tasks: Dict[str, List[str]] = {}
        
        if config_path and Path(config_path).exists():
            self.load_config(config_path)
        else:
            self._initialize_default_mappings()
    
    def _initialize_default_mappings(self):
        """Initialize default task mappings"""
        
        # Define individual restoration tasks based on the available checkpoints
        default_tasks = [
            TaskMapping(
                task_name="deblur",
                keywords={
                    "blur", "blurry", "blurred", "sharp", "sharpen", "sharpening", 
                    "focus", "focusing", "unfocus", "defocus", "motion blur",
                    "gaussian blur", "out of focus", "soft", "clarity", "clear up"
                },
                checkpoint_path="pre-trained/deblur",
                description="Remove blur and improve image sharpness",
                priority=2
            ),
            TaskMapping(
                task_name="dehaze",
                keywords={
                    "haze", "hazy", "fog", "foggy", "mist", "misty", "smog", 
                    "atmospheric", "visibility", "clear air", "remove haze",
                    "defog", "defogging", "visibility enhancement", "atmospheric scattering",
                    "clouds", "cloudy", "overcast"
                },
                checkpoint_path="pre-trained/dehaze",
                description="Remove atmospheric haze and fog",
                priority=2
            ),
            TaskMapping(
                task_name="derain",
                keywords={
                    "rain", "rainy", "raindrops", "water drops", "precipitation",
                    "wet", "remove rain", "rain removal", "weather", "storm",
                    "drizzle", "downpour", "rain streaks", "water streaks"
                },
                checkpoint_path="pre-trained/derain",
                description="Remove rain effects and water artifacts",
                priority=2
            ),
            TaskMapping(
                task_name="desnow",
                keywords={
                    "snow", "snowy", "snowflakes", "winter", "snowfall",
                    "remove snow", "snow removal", "blizzard", "snow storm"
                },
                checkpoint_path="pre-trained/desnow",
                description="Remove snow effects from images",
                priority=2
            ),
            TaskMapping(
                task_name="demoire",
                keywords={
                    "moire", "moiré", "pattern", "interference", "grid",
                    "screen", "digital artifacts", "remove moire", "moire removal"
                },
                checkpoint_path="pre-trained/demoire",
                description="Remove moire patterns and interference",
                priority=2
            ),
            TaskMapping(
                task_name="lowlight",
                keywords={
                    "dark", "dim", "low light", "underexposed", "bright", "brighten",
                    "illuminate", "lighting", "exposure", "shadow", "shadows",
                    "enhance lighting", "light enhancement", "brightening", "night",
                    "nighttime", "enhance brightness"
                },
                checkpoint_path="pre-trained/lowlight",
                description="Enhance low-light and dark images",
                priority=2
            ),
            TaskMapping(
                task_name="highlight",
                keywords={
                    "highlight", "highlights", "overexposed", "bright spots",
                    "blown out", "white balance", "exposure correction",
                    "reduce highlights", "tone down", "balance exposure"
                },
                checkpoint_path="pre-trained/highlight",
                description="Correct overexposure and bright highlights",
                priority=2
            ),
            TaskMapping(
                task_name="decloud",
                keywords={
                    "cloud", "clouds", "cloudy", "overcast", "sky",
                    "remove clouds", "cloud removal", "clear sky"
                },
                checkpoint_path="pre-trained/decloud",
                description="Remove clouds from images",
                priority=2
            ),
            TaskMapping(
                task_name="unrefract",
                keywords={
                    "refraction", "underwater", "water distortion", "optical distortion",
                    "remove refraction", "correct distortion", "underwater correction",
                    "water effects", "submerged"
                },
                checkpoint_path="pre-trained/unrefract",
                description="Correct underwater refraction effects",
                priority=2
            ),
            TaskMapping(
                task_name="face",
                keywords={
                    "face", "facial", "portrait", "person", "people",
                    "face enhancement", "facial restoration", "skin",
                    "face repair", "portrait enhancement"
                },
                checkpoint_path="pre-trained/face",
                description="Enhance facial features and portraits",
                priority=2
            ),
            TaskMapping(
                task_name="general_restoration",
                keywords={
                    "restore", "restoration", "repair", "fix", "enhance",
                    "improve", "quality", "general", "overall", "complete",
                    "clean up", "make better", "enhance quality"
                },
                checkpoint_path="pre-trained/general",
                description="General image restoration and enhancement",
                priority=1
            )
        ]
        
        # Store task mappings
        for task in default_tasks:
            self.task_mappings[task.task_name] = task
        
        # Define compound tasks (combinations)
        self.compound_tasks = {
            "dehaze_derain": ["dehaze", "derain"],
            "deblur_enhance": ["deblur", "general_restoration"],
            "weather_restoration": ["dehaze", "derain", "desnow"],
            "complete_restoration": ["deblur", "dehaze", "derain", "general_restoration"],
            "outdoor_restoration": ["dehaze", "decloud", "derain"],
            "indoor_restoration": ["lowlight", "deblur", "general_restoration"]
        }
    
    def add_task(self, task_mapping: TaskMapping):
        """Add a new task mapping"""
        self.task_mappings[task_mapping.task_name] = task_mapping
    
    def add_compound_task(self, compound_name: str, task_list: List[str]):
        """Add a compound task definition"""
        self.compound_tasks[compound_name] = task_list
    
    def preprocess_prompt(self, prompt: str) -> str:
        """Preprocess the input prompt for better matching"""
        # Convert to lowercase
        prompt = prompt.lower()
        
        # Remove punctuation and extra spaces
        prompt = re.sub(r'[^\w\s]', ' ', prompt)
        prompt = re.sub(r'\s+', ' ', prompt)
        
        # Handle common variations and stop words
        replacements = {
            "i want to": "",
            "please": "",
            "can you": "",
            "help me": "",
            "from this image": "",
            "in this photo": "",
            "this picture": "",
            "the image": "",
            "my image": "",
            "my photo": "",
            "this image": "",
            "the photo": ""
        }
        
        for old, new in replacements.items():
            prompt = prompt.replace(old, new)
        
        return prompt.strip()
    
    def extract_tasks_from_prompt(self, prompt: str) -> Dict[str, float]:
        """
        Extract restoration tasks from natural language prompt.
        
        Args:
            prompt: Natural language description of desired restoration
            
        Returns:
            Dictionary mapping task names to confidence scores
        """
        processed_prompt = self.preprocess_prompt(prompt)
        words = set(processed_prompt.split())
        
        task_scores = {}
        
        # Check each task for keyword matches
        for task_name, task_mapping in self.task_mappings.items():
            matches = words.intersection(task_mapping.keywords)
            if matches:
                # Calculate confidence based on number of matches and priority
                base_score = len(matches) / len(task_mapping.keywords)
                priority_bonus = task_mapping.priority * 0.1
                task_scores[task_name] = min(1.0, base_score + priority_bonus)
        
        # Check for compound task indicators
        compound_indicators = {
            "and", "also", "plus", "with", "including", "both", "all", 
            "multiple", "several", "various", "different", "combined",
            "as well as", "together with"
        }
        
        has_compound_indicator = bool(words.intersection(compound_indicators))
        
        # If multiple tasks detected and compound indicators present, boost scores
        if len(task_scores) > 1 and has_compound_indicator:
            for task in task_scores:
                task_scores[task] *= 1.2  # Boost compound tasks
        
        return task_scores
    
    def select_tasks(self, prompt: str, threshold: float = 0.1, max_tasks: int = 3) -> List[Tuple[str, float]]:
        """
        Select restoration tasks based on natural language prompt.
        
        Args:
            prompt: Natural language description
            threshold: Minimum confidence threshold for task selection
            max_tasks: Maximum number of tasks to select
            
        Returns:
            List of (task_name, confidence) tuples, sorted by confidence
        """
        task_scores = self.extract_tasks_from_prompt(prompt)
        
        # Filter by threshold and sort by confidence
        selected_tasks = [
            (task, score) for task, score in task_scores.items() 
            if score >= threshold
        ]
        selected_tasks.sort(key=lambda x: x[1], reverse=True)
        
        # Limit to max_tasks
        return selected_tasks[:max_tasks]
    
    def get_checkpoint_paths(self, selected_tasks: List[Tuple[str, float]]) -> List[str]:
        """Get checkpoint paths for selected tasks"""
        paths = []
        for task_name, _ in selected_tasks:
            if task_name in self.task_mappings:
                paths.append(self.task_mappings[task_name].checkpoint_path)
        return paths
    
    def suggest_compound_task(self, selected_tasks: List[str]) -> Optional[str]:
        """Suggest a compound task name if applicable"""
        selected_set = set(selected_tasks)
        
        # Check if selected tasks match any predefined compound tasks
        for compound_name, compound_tasks in self.compound_tasks.items():
            if set(compound_tasks) == selected_set:
                return compound_name
        
        # Generate automatic compound name
        if len(selected_tasks) > 1:
            return "_".join(sorted(selected_tasks))
        
        return None
    
    def explain_selection(self, prompt: str) -> Dict:
        """
        Provide detailed explanation of task selection process.
        
        Returns:
            Dictionary with selection details and reasoning
        """
        processed_prompt = self.preprocess_prompt(prompt)
        task_scores = self.extract_tasks_from_prompt(prompt)
        selected_tasks = self.select_tasks(prompt)
        
        explanation = {
            "original_prompt": prompt,
            "processed_prompt": processed_prompt,
            "detected_keywords": {},
            "task_scores": task_scores,
            "selected_tasks": selected_tasks,
            "checkpoint_paths": self.get_checkpoint_paths(selected_tasks),
            "compound_task": self.suggest_compound_task([t[0] for t in selected_tasks])
        }
        
        # Add keyword detection details
        words = set(processed_prompt.split())
        for task_name, task_mapping in self.task_mappings.items():
            matches = words.intersection(task_mapping.keywords)
            if matches:
                explanation["detected_keywords"][task_name] = list(matches)
        
        return explanation
    
    def save_config(self, config_path: str):
        """Save current task mappings to JSON file"""
        config_data = {
            "tasks": {},
            "compound_tasks": self.compound_tasks
        }
        
        for task_name, mapping in self.task_mappings.items():
            config_data["tasks"][task_name] = {
                "keywords": list(mapping.keywords),
                "checkpoint_path": mapping.checkpoint_path,
                "description": mapping.description,
                "priority": mapping.priority
            }
        
        with open(config_path, 'w') as f:
            json.dump(config_data, f, indent=2)
    
    def load_config(self, config_path: str):
        """Load task mappings from JSON file"""
        with open(config_path, 'r') as f:
            config_data = json.load(f)
        
        self.task_mappings = {}
        # Support both "tasks" and "task_mappings" keys for backward compatibility
        task_data_dict = config_data.get("task_mappings", config_data.get("tasks", {}))
        
        for task_name, task_data in task_data_dict.items():
            self.task_mappings[task_name] = TaskMapping(
                task_name=task_name,
                keywords=set(task_data["keywords"]),
                checkpoint_path=task_data["checkpoint_path"],
                description=task_data["description"],
                priority=task_data.get("priority", 1)
            )
        
        self.compound_tasks = config_data.get("compound_tasks", {})

# Convenience function for quick task selection
def select_restoration_tasks(prompt: str, config_path: Optional[str] = None) -> List[Tuple[str, float]]:
    """
    Quick function to select restoration tasks from natural language prompt.
    
    Args:
        prompt: Natural language description of restoration needs
        config_path: Optional path to custom task configuration
        
    Returns:
        List of (task_name, confidence_score) tuples
    """
    selector = TaskSelector(config_path)
    return selector.select_tasks(prompt)