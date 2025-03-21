import torch
import gradio as gr
from PIL import Image, ImageDraw
import numpy as np
from torchvision import transforms
from torchvision.models import vit_b_16, ViT_B_16_Weights


print("Loading Vision Transformer model...")
weights = ViT_B_16_Weights.DEFAULT
model = vit_b_16(weights=weights)
model.eval()


class_names = weights.meta["categories"]

transform = weights.transforms()


def visualize_patches(img):
    """Create a visualization of the image patches"""

    if isinstance(img, np.ndarray):
        img = Image.fromarray(img.astype('uint8'))

    # Resize to 224x224 to match ViT input
    img = img.resize((224, 224))

    draw_img = img.copy()
    draw = ImageDraw.Draw(draw_img)

    # ViT patch size (16x16 for 224x224 image)
    patch_size = 14

    for i in range(1, 16):
        draw.line([(0, i * patch_size), (224, i * patch_size)],
                  fill=(255, 0, 255), width=1)
        draw.line([(i * patch_size, 0), (i * patch_size, 224)],
                  fill=(255, 0, 255), width=1)

    return draw_img


def predict_image(img):
    """Process image and return predictions and visualization"""
    if img is None:
        return "Please upload an image", None

    if isinstance(img, np.ndarray):
        input_img = Image.fromarray(img.astype('uint8'))
    else:
        input_img = img

    patch_viz = visualize_patches(input_img)

    img_tensor = transform(input_img).unsqueeze(0)

    with torch.no_grad():
        output = model(img_tensor)

    # Get top 5 predictions
    probabilities = torch.nn.functional.softmax(output[0], dim=0)
    top5_prob, top5_indices = torch.topk(probabilities, 5)

    results = []
    for i, (idx, prob) in enumerate(zip(top5_indices, top5_prob)):
        results.append(f"{class_names[idx]}: {prob.item()*100:.2f}%")
    return "\n".join(results), patch_viz


# Create a simple Gradio interface
with gr.Blocks() as demo:
    gr.Markdown("# Vision Transformer (ViT) Demo")
    gr.Markdown(
        "Upload an image to see how Vision Transformer processes and classifies it.")

    with gr.Row():
        with gr.Column(scale=1):
            input_image = gr.Image(type="pil", label="Upload Image")
            classify_btn = gr.Button("Classify Image")

        with gr.Column(scale=1):
            patch_viz = gr.Image(type="pil", label="Image with Patches")
            prediction = gr.Textbox(label="Top 5 Predictions")

    # Set up the button click
    classify_btn.click(
        fn=predict_image,
        inputs=input_image,
        outputs=[prediction, patch_viz]
    )

    # Add demo explanation
    with gr.Accordion("How Vision Transformer Works", open=False):
        gr.Markdown("""
        ## How Vision Transformer Works
        
        1. **Patch Extraction**: The image is divided into 16x16 equal patches (shown by the purple grid)
        2. **Position Encoding**: Each patch is given position information so the model knows where they are in the image
        3. **Attention Mechanism**: Each patch "attends to" all other patches, evaluating their relationships
        4. **Feature Processing**: Each patch is processed independently by the transformer
        5. **Classification**: The model uses the processed patch information to identify the image content
        
        This demo uses `vit_b_16`, a standard Vision Transformer pre-trained on ImageNet, capable of recognizing 1,000 different object classes.
        """)

# Run the app
if __name__ == "__main__":
    demo.launch()
