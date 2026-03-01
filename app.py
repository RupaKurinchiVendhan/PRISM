#!/usr/bin/env python3
"""
Gradio Web App for PRISM Image Restoration
Provides an interactive UI for natural language image restoration
"""

import gradio as gr
import subprocess
import os
import tempfile
import shutil
from PIL import Image
import time
from pathlib import Path

# Default paths
DEFAULT_PRISM_CHECKPOINT = "pre-trained/prism_model.pt"
DEFAULT_COMBINED_WEIGHTS = "pre-trained/combined_weights.pt"
# DEFAULT_CLIP_PATH = "pre-trained/ca_clip.pt"
# DEFAULT_DEGRADATION_ENCODER = "pre-trained/best_model.pt"
SAVE_ROOT = "demo_results"

# Create results directory if it doesn't exist
os.makedirs(SAVE_ROOT, exist_ok=True)

# Check which weights are available
USE_COMBINED_WEIGHTS = os.path.exists(DEFAULT_COMBINED_WEIGHTS)


def restore_image(input_image, prompt, num_inference_steps, seed, selected_distortions):
    """
    Run image restoration using the demo.py script
    
    Args:
        input_image: PIL Image or numpy array
        prompt: Natural language prompt describing restoration task (optional)
        num_inference_steps: Number of diffusion steps
        seed: Random seed for reproducibility
        selected_distortions: List of selected distortion types
    
    Returns:
        restored_image: PIL Image
        status_message: String with execution details
    """
    try:
        if input_image is None:
            return None, "Please upload an image first!"
        
        # Initialize status message
        status_msg = ""
        
        # Validate required files exist
        if not os.path.exists('modules/demo.py'):
            return None, "Error: modules/demo.py not found. Please ensure you're running from the correct directory."
        
        if not os.path.exists(DEFAULT_PRISM_CHECKPOINT):
            return None, f"Error: PRISM checkpoint not found at {DEFAULT_PRISM_CHECKPOINT}"
        
        # Create a temporary file to save the input image
        input_path = None
        try:
            with tempfile.NamedTemporaryFile(delete=False, suffix=".png") as tmp_input:
                input_path = tmp_input.name
                if isinstance(input_image, Image.Image):
                    input_image.save(input_path)
                else:
                    # Convert numpy array to PIL Image
                    Image.fromarray(input_image).save(input_path)
            
            # Build command
            cmd = [
                'python', '-m', 'modules.demo',
                '--img_path', input_path,
                '--save_root', SAVE_ROOT,
                '--num_inference_steps', str(num_inference_steps),
                '--seed', str(seed),
                '--prism_checkpoint_path', DEFAULT_PRISM_CHECKPOINT,
            ]
            
            # Use combined weights if available, otherwise use separate files
            if USE_COMBINED_WEIGHTS:
                cmd.extend(['--combined_weights_path', DEFAULT_COMBINED_WEIGHTS])
                status_msg += "Using combined weights file\n"
            else:
                cmd.extend([
                    '--clip_path', DEFAULT_CLIP_PATH,
                    '--degradation_encoder_checkpoint', DEFAULT_DEGRADATION_ENCODER
                ])
                status_msg += "Using separate weight files\n"
            
            # Determine which mode to use: buttons, manual prompt, or auto-detect
            if selected_distortions:
                # Use selected distortions from buttons
                distortion_prompt = " ".join(selected_distortions)
                cmd.extend(['--prompt', distortion_prompt])
                status_msg = f"Removing selected distortions: {', '.join(selected_distortions)}\n"
            elif prompt and prompt.strip() != "":
                # Use manual text prompt
                cmd.extend(['--prompt', prompt])
                status_msg += f"Processing with prompt: '{prompt}'\n"
            else:
                # Auto-detect if no prompt or buttons provided
                if USE_COMBINED_WEIGHTS:
                    # Combined weights already includes degradation encoder
                    pass
                else:
                    cmd.extend(['--degradation_encoder_checkpoint', DEFAULT_DEGRADATION_ENCODER])
                status_msg += "Auto-detecting distortions...\n"
            
            # Run the restoration
            status_msg += f"Running restoration with {num_inference_steps} steps...\n"
            status_msg += f"Command: {' '.join(cmd)}\n\n"
            
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=300)
            
            # Check for subprocess errors
            if result.returncode != 0:
                status_msg += f"\n--- Error: Command failed with exit code {result.returncode} ---\n"
            
            status_msg += "\n--- Output ---\n"
            status_msg += result.stdout if result.stdout else "(no output)"
            
            if result.stderr:
                status_msg += "\n--- Warnings/Errors ---\n"
                status_msg += result.stderr
            
            # If command failed, return early with error
            if result.returncode != 0:
                return None, status_msg
            
            # Find the most recent output file
            output_files = list(Path(SAVE_ROOT).glob("*.png")) + list(Path(SAVE_ROOT).glob("*.jpg"))
            if output_files:
                latest_output = max(output_files, key=os.path.getctime)
                restored_image = Image.open(latest_output)
                status_msg += f"\n\nSuccess! Restored image saved to: {latest_output}"
                return restored_image, status_msg
            else:
                status_msg += "\n\nError: No output image was generated."
                return None, status_msg
                
        except subprocess.TimeoutExpired:
            return None, "Error: Restoration timed out (>5 minutes). Try reducing inference steps."
        finally:
            # Clean up temporary input file
            if input_path and os.path.exists(input_path):
                os.remove(input_path)
    
    except Exception as e:
        import traceback
        error_msg = f"Unexpected error during restoration:\n{str(e)}\n\n{traceback.format_exc()}"
        return None, error_msg


def create_app():
    """Create and configure the Gradio interface"""
    
    with gr.Blocks(title="PRISM Image Restoration", analytics_enabled=False) as app:
        # Add custom CSS for button styling
        gr.HTML("""
        <style>
        .gradio-container {
            font-family: 'Arial', sans-serif;
        }
        /* Category color coding - light backgrounds with dark text */
        button.distortion-btn-geometric { 
            background: linear-gradient(135deg, #e3f2fd 0%, #bbdefb 100%) !important;
            border: 2px solid #90caf9 !important;
            color: #2196f3 !important;
            transition: all 0.2s ease !important;
        }
        button.distortion-btn-photometric { 
            background: linear-gradient(135deg, #fff3e0 0%, #ffe0b2 100%) !important;
            border: 2px solid #ffb74d !important;
            color: #f57c00 !important;
            transition: all 0.2s ease !important;
        }
        button.distortion-btn-occlusion { 
            background: linear-gradient(135deg, #f3e5f5 0%, #e1bee7 100%) !important;
            border: 2px solid #ba68c8 !important;
            color: #9c28b0 !important;
            transition: all 0.2s ease !important;
        }
        button.distortion-btn-noise { 
            background: linear-gradient(135deg, #ffebee 0%, #ffcdd2 100%) !important;
            border: 2px solid #ef9a9a !important;
            color: #f44336 !important;
            transition: all 0.2s ease !important;
        }
        /* Selected state - thick colored border and bold text */
        button.distortion-btn-geometric.selected { 
            border: 5px solid #2196f3 !important;
            box-shadow: 0 0 15px rgba(33, 150, 243, 0.6) !important;
            font-weight: 900 !important;
            transform: scale(1.05) !important;
        }
        button.distortion-btn-photometric.selected { 
            border: 5px solid #ffc107 !important;
            box-shadow: 0 0 15px rgba(255, 193, 7, 0.6) !important;
            font-weight: 900 !important;
            transform: scale(1.05) !important;
        }
        button.distortion-btn-occlusion.selected { 
            border: 5px solid #9c28b0 !important;
            box-shadow: 0 0 15px rgba(156, 40, 176, 0.6) !important;
            font-weight: 900 !important;
            transform: scale(1.05) !important;
        }
        button.distortion-btn-noise.selected { 
            border: 5px solid #f44336 !important;
            box-shadow: 0 0 15px rgba(244, 67, 54, 0.6) !important;
            font-weight: 900 !important;
            transform: scale(1.05) !important;
        }
        </style>
        <script>
        // Function to toggle button selection - more aggressive approach
        function setupDistortionButtons() {
            const buttons = document.querySelectorAll('button[class*="distortion-btn-"]');
            console.log('Found buttons:', buttons.length);
            
            buttons.forEach((button, index) => {
                // Check if already has listener
                if (button.hasAttribute('data-listener-added')) return;
                button.setAttribute('data-listener-added', 'true');
                
                // Add click listener
                button.addEventListener('click', function(e) {
                    console.log('Button clicked:', this.textContent);
                    // Toggle selected class
                    if (this.classList.contains('selected')) {
                        this.classList.remove('selected');
                        console.log('Removed selected class');
                    } else {
                        this.classList.add('selected');
                        console.log('Added selected class');
                    }
                }, true); // Use capture phase
            });
        }
        
        // Run setup multiple times with delays
        function runSetup() {
            console.log('Running setupDistortionButtons');
            setupDistortionButtons();
        }
        
        // Initial run
        runSetup();
        
        // Run after delays
        setTimeout(runSetup, 100);
        setTimeout(runSetup, 500);
        setTimeout(runSetup, 1000);
        setTimeout(runSetup, 2000);
        setTimeout(runSetup, 3000);
        setTimeout(runSetup, 5000);
        
        // Watch for DOM changes
        const observer = new MutationObserver(function(mutations) {
            runSetup();
        });
        
        // Start observing
        setTimeout(function() {
            const container = document.body;
            if (container) {
                observer.observe(container, { 
                    childList: true, 
                    subtree: true,
                    attributes: true,
                    attributeFilter: ['class']
                });
                console.log('Observer started');
            }
        }, 100);
        
        // Also listen for Gradio's load event
        window.addEventListener('load', runSetup);
        document.addEventListener('DOMContentLoaded', runSetup);
        </script>
        """)
        gr.Markdown(
            """
            # PRISM Demo: Controllable & Compositional Image Restoration
            
            Upload an image and describe how you want to restore it. 
            Leave the prompt empty to automatically detect and remove distortions.
            """
        )
        
        with gr.Row():
            with gr.Column(scale=1):
                gr.Markdown("### Input")
                input_image = gr.Image(
                    label="Upload Image",
                    type="pil"
                )
                
                prompt = gr.Textbox(
                    label="Optional Restoration Prompt (ex: `Dehaze this image.`, `Remove the rain from this photo.`)",
                    placeholder="Describe what you want to fix, or leave empty for auto-detection",
                    lines=2,
                    interactive=True
                )
                
                # State to track selected distortions
                selected_distortions = gr.State([])
                
                with gr.Accordion("Advanced Settings", open=False):
                    num_steps = gr.Slider(
                        minimum=5,
                        maximum=100,
                        value=20,
                        step=1,
                        label="Number of Inference Steps",
                        info="More steps = better quality but slower"
                    )
                    seed = gr.Number(
                        value=42,
                        label="Random Seed",
                        precision=0,
                        info="For reproducible results"
                    )
                
                restore_btn = gr.Button("Restore Image", variant="primary", size="lg")
                
                gr.Markdown("---")
                gr.Markdown("### Alternate Restoration Method")
                
                gr.Markdown("**Select Distortions to Remove (up to 3)** - *Click buttons to toggle selection*")
                
                # Display for selected distortions - inline, no box
                selected_display = gr.Markdown("**Selected:** None")

                # Geometric distortions (blue background)
                gr.Markdown("**Geometric**")
                with gr.Row():
                    btn_motion_blur = gr.Button("Motion Blur", variant="secondary", size="sm", elem_classes=["distortion-btn-geometric"], elem_id="btn-motion-blur")
                    btn_warping = gr.Button("Warping", variant="secondary", size="sm", elem_classes=["distortion-btn-geometric"], elem_id="btn-warping")
                    btn_refraction = gr.Button("Refraction", variant="secondary", size="sm", elem_classes=["distortion-btn-geometric"], elem_id="btn-refraction")
                    btn_defocus = gr.Button("Defocus Blur", variant="secondary", size="sm", elem_classes=["distortion-btn-geometric"], elem_id="btn-defocus")
                
                # Photometric degradations (orange background)
                gr.Markdown("**Photometric**")
                with gr.Row():
                    btn_contrast = gr.Button("Contrast", variant="secondary", size="sm", elem_classes=["distortion-btn-photometric"], elem_id="btn-contrast")
                    btn_color = gr.Button("Color Shifts", variant="secondary", size="sm", elem_classes=["distortion-btn-photometric"], elem_id="btn-color")
                    btn_brightness = gr.Button("Brightness", variant="secondary", size="sm", elem_classes=["distortion-btn-photometric"], elem_id="btn-brightness")
                    btn_lowlight = gr.Button("Low Light", variant="secondary", size="sm", elem_classes=["distortion-btn-photometric"], elem_id="btn-lowlight")
                
                # Occlusions (purple background)
                gr.Markdown("**Occlusions**")
                with gr.Row():
                    btn_clouds = gr.Button("Clouds", variant="secondary", size="sm", elem_classes=["distortion-btn-occlusion"], elem_id="btn-clouds")
                    btn_haze = gr.Button("Haze", variant="secondary", size="sm", elem_classes=["distortion-btn-occlusion"], elem_id="btn-haze")
                    btn_rain = gr.Button("Rain", variant="secondary", size="sm", elem_classes=["distortion-btn-occlusion"], elem_id="btn-rain")
                    btn_snow = gr.Button("Snow", variant="secondary", size="sm", elem_classes=["distortion-btn-occlusion"], elem_id="btn-snow")
                
                # Noise-based effects (green background)
                gr.Markdown("**Noise & Resolution**")
                with gr.Row():
                    btn_noise = gr.Button("Additive Noise", variant="secondary", size="sm", elem_classes=["distortion-btn-noise"], elem_id="btn-noise")
                    btn_compression = gr.Button("Compression", variant="secondary", size="sm", elem_classes=["distortion-btn-noise"], elem_id="btn-compression")
                
            with gr.Column(scale=1):
                gr.Markdown("### Output")
                output_image = gr.Image(
                    label="Restored Image",
                    type="pil"
                )
                
                status_text = gr.Textbox(
                    label="Status & Details",
                    lines=10,
                    max_lines=20,
                    interactive=False
                )
        
        # Set up the event handler
        # Mapping from internal distortion names to display names
        DISTORTION_DISPLAY_NAMES = {
            "blur": "Motion Blur",
            "warp": "Warping",
            "refract": "Refraction",
            "defocus": "Defocus Blur",
            "contrast": "Contrast",
            "color": "Color Shifts",
            "brightness": "Brightness",
            "low": "Low Light",
            "clouds": "Clouds",
            "haze": "Haze",
            "rain": "Rain",
            "snow": "Snow",
            "noise": "Additive Noise",
            "compress": "Compression"
        }
        
        def toggle_distortion(current_list, distortion_name):
            """Toggle a distortion in the selected list (max 3 selections)"""
            current_list = list(current_list) if current_list else []
            if distortion_name in current_list:
                # Deselect
                current_list.remove(distortion_name)
            else:
                # Select (if under limit)
                if len(current_list) < 3:
                    current_list.append(distortion_name)
            
            # Create display string with human-readable names
            if not current_list:
                display = "**Selected:** None"
            else:
                display_names = [DISTORTION_DISPLAY_NAMES.get(d, d) for d in current_list]
                display = f"**Selected ({len(current_list)}/3):** {', '.join(display_names)}"
            
            return current_list, display
        
        # Button click handlers for each distortion - now return both state and display
        btn_motion_blur.click(lambda x: toggle_distortion(x, "blur"), inputs=[selected_distortions], outputs=[selected_distortions, selected_display], _js="() => { document.getElementById('btn-motion-blur')?.classList.toggle('selected'); }")
        btn_warping.click(lambda x: toggle_distortion(x, "warp"), inputs=[selected_distortions], outputs=[selected_distortions, selected_display], _js="() => { document.getElementById('btn-warping')?.classList.toggle('selected'); }")
        btn_refraction.click(lambda x: toggle_distortion(x, "refract"), inputs=[selected_distortions], outputs=[selected_distortions, selected_display], _js="() => { document.getElementById('btn-refraction')?.classList.toggle('selected'); }")
        btn_defocus.click(lambda x: toggle_distortion(x, "defocus"), inputs=[selected_distortions], outputs=[selected_distortions, selected_display], _js="() => { document.getElementById('btn-defocus')?.classList.toggle('selected'); }")
        
        btn_contrast.click(lambda x: toggle_distortion(x, "contrast"), inputs=[selected_distortions], outputs=[selected_distortions, selected_display], _js="() => { document.getElementById('btn-contrast')?.classList.toggle('selected'); }")
        btn_color.click(lambda x: toggle_distortion(x, "color"), inputs=[selected_distortions], outputs=[selected_distortions, selected_display], _js="() => { document.getElementById('btn-color')?.classList.toggle('selected'); }")
        btn_brightness.click(lambda x: toggle_distortion(x, "brightness"), inputs=[selected_distortions], outputs=[selected_distortions, selected_display], _js="() => { document.getElementById('btn-brightness')?.classList.toggle('selected'); }")
        btn_lowlight.click(lambda x: toggle_distortion(x, "low"), inputs=[selected_distortions], outputs=[selected_distortions, selected_display], _js="() => { document.getElementById('btn-lowlight')?.classList.toggle('selected'); }")
        
        btn_clouds.click(lambda x: toggle_distortion(x, "decloud"), inputs=[selected_distortions], outputs=[selected_distortions, selected_display], _js="() => { document.getElementById('btn-clouds')?.classList.toggle('selected'); }")
        btn_haze.click(lambda x: toggle_distortion(x, "haze"), inputs=[selected_distortions], outputs=[selected_distortions, selected_display], _js="() => { document.getElementById('btn-haze')?.classList.toggle('selected'); }")
        btn_rain.click(lambda x: toggle_distortion(x, "rain"), inputs=[selected_distortions], outputs=[selected_distortions, selected_display], _js="() => { document.getElementById('btn-rain')?.classList.toggle('selected'); }")
        btn_snow.click(lambda x: toggle_distortion(x, "snow"), inputs=[selected_distortions], outputs=[selected_distortions, selected_display], _js="() => { document.getElementById('btn-snow')?.classList.toggle('selected'); }")
        
        btn_noise.click(lambda x: toggle_distortion(x, "noise"), inputs=[selected_distortions], outputs=[selected_distortions, selected_display], _js="() => { document.getElementById('btn-noise')?.classList.toggle('selected'); }")
        btn_compression.click(lambda x: toggle_distortion(x, "compress"), inputs=[selected_distortions], outputs=[selected_distortions, selected_display], _js="() => { document.getElementById('btn-compression')?.classList.toggle('selected'); }")
        
        restore_btn.click(
            fn=restore_image,
            inputs=[input_image, prompt, num_steps, seed, selected_distortions],
            outputs=[output_image, status_text],
            api_name="restore"
        )
        
        # Example images (if they exist)
        example_dir = "data/real_demo"
        if os.path.exists(example_dir):
            example_files = [
                os.path.join(example_dir, f) for f in os.listdir(example_dir)
                if f.endswith(('.png', '.jpg', '.jpeg'))
            ][:5]  # Limit to 5 examples
            
            if example_files:
                gr.Markdown("### Example Images")
                gr.Examples(
                    examples=[
                        [f, "Remove the haze", 20, 42] 
                        for f in example_files
                    ],
                    inputs=[input_image, prompt, num_steps, seed],
                    outputs=[output_image, status_text],
                    fn=restore_image,
                    cache_examples=False
                )
    
    return app


if __name__ == "__main__":
    import argparse
    import socket
    
    def find_free_port(start_port=7860, max_port=7960):
        """Find an available port starting from start_port"""
        for port in range(start_port, max_port):
            try:
                with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
                    s.bind(('', port))
                    return port
            except OSError:
                continue
        return None
    
    parser = argparse.ArgumentParser(description="PRISM Gradio Web App")
    parser.add_argument("--share", action="store_true", help="Create a public shareable link")
    parser.add_argument("--port", type=int, default=None, help="Port to run the app on (auto-detect if not specified)")
    parser.add_argument("--server_name", type=str, default="127.0.0.1", help="Server name/IP")
    args = parser.parse_args()
    
    # Find available port if not specified
    if args.port is None:
        args.port = find_free_port()
        if args.port is None:
            print("Error: Could not find an available port in range 7860-7960")
            exit(1)
        print(f"Auto-detected available port: {args.port}")
    
    print("Starting PRISM Image Restoration Web App...")
    print(f"Server will run on: http://{args.server_name}:{args.port}")
    if args.share:
        print("Creating public shareable link...")
    
    app = create_app()
    app.launch(
        share=args.share,
        server_name=args.server_name,
        server_port=args.port,
        show_error=True,
        quiet=False,
        max_threads=10
    )
