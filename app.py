import streamlit as st
from PIL import Image
import torch
import torchvision.models as models
import torchvision.transforms as transforms
import urllib.request
from ui_components import render_header, render_footer

render_header()

LABELS_URL = "https://raw.githubusercontent.com/pytorch/hub/master/imagenet_classes.txt"
response = urllib.request.urlopen(LABELS_URL)
labels = [line.strip().decode("utf-8") for line in response.readlines()]

model = models.resnet50(pretrained=True)
model.eval()

transform = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225])
])

st.markdown("### 📤 Upload your image:")
uploaded_file = st.file_uploader("", type=["jpg", "png", "jpeg"])

if uploaded_file:
    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, caption="🖼 Uploaded Image", use_column_width=True)

    input_tensor = transform(image).unsqueeze(0)
    with torch.no_grad():
        output = model(input_tensor)
        probs = torch.nn.functional.softmax(output[0], dim=0)
        top_probs, top_idxs = torch.topk(probs, 3)

    st.markdown("### 🔍 Top Predictions:")
    for i in range(3):
        st.success(f"**{labels[top_idxs[i]]}** — {top_probs[i].item():.2%}")
else:
    st.info("👈 Upload an image to begin classification.")

render_footer()
