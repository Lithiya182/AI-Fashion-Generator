import streamlit as st
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
import io
import time

# -----------------------------------
# PAGE CONFIG
# -----------------------------------
st.set_page_config(
    page_title="AI Fashion Generator",
    layout="wide"
)

# -----------------------------------
# CUSTOM CSS
# -----------------------------------
st.markdown("""
<style>

/* HIDE STREAMLIT DEFAULTS */
header {
    visibility: hidden;
}

footer {
    visibility: hidden;
}

/* APP BACKGROUND */
.stApp {

    background:
    linear-gradient(
        135deg,
        #020617 0%,
        #0f172a 50%,
        #111827 100%
    );

    color: white;

    font-family: 'Inter', sans-serif;
}

/* MAIN CONTAINER */
.block-container {

    max-width: 1450px;

    padding-top: 3rem;

    padding-bottom: 3rem;
}

/* TITLE */
.main-title {

    text-align: center;

    font-size: 68px;

    font-weight: 800;

    color: white;

    margin-bottom: 10px;
}

/* SUBTITLE */
.sub-title {

    text-align: center;

    color: #94a3b8;

    font-size: 20px;

    margin-bottom: 60px;
}

/* GLASS CARD */
[data-testid="stVerticalBlockBorderWrapper"] {

    background:
    rgba(15,23,42,0.72);

    backdrop-filter: blur(14px);

    border-radius: 24px;

    border: 1px solid rgba(255,255,255,0.08);

    padding: 28px;

    box-shadow:
    0 8px 30px rgba(0,0,0,0.35);

    transition: 0.3s ease;
}

/* HOVER EFFECT */
[data-testid="stVerticalBlockBorderWrapper"]:hover {

    border: 1px solid rgba(139,92,246,0.35);
}

/* LABELS */
label {

    color: #e2e8f0 !important;

    font-weight: 600 !important;

    font-size: 15px !important;
}

/* SELECTBOX */
.stSelectbox > div > div {

    background: #1e293b !important;

    border: 1px solid #334155 !important;

    border-radius: 14px !important;

    min-height: 52px !important;

    color: white !important;
}

/* SELECTED TEXT */
.stSelectbox div[data-baseweb="select"] > div {
    color: white !important;
}

/* ALL DROPDOWN TEXT */
.stSelectbox * {
    color: white !important;
}

/* SLIDER */
.stSlider [data-baseweb="slider"] div {
    color: #8b5cf6 !important;
}

/* BUTTON */
.stButton {
    margin-top: 22px;
}

.stDownloadButton {
    margin-top: -8px;
}
.stButton > button {

    width: 100%;

    background:
    linear-gradient(
        90deg,
        #7c3aed,
        #9333ea
    );

    color: white;

    border: none;

    border-radius: 16px;

    padding: 13px;

    font-size: 17px;

    font-weight: 700;

    transition: 0.3s ease;

    box-shadow:
    0 10px 25px rgba(124,58,237,0.3);
}

.stButton > button:hover {

    transform: translateY(-2px);

    box-shadow:
    0 16px 30px rgba(124,58,237,0.45);
}

/* SECTION TITLE */
.section-title {

    font-size: 34px;

    font-weight: 800;

    color: white;

    margin-bottom: 25px;
}

/* IMAGE CARD */
.image-box {

    background:
    rgba(255,255,255,0.03);

    border-radius: 18px;

    padding: 12px;

    border:
    1px solid rgba(255,255,255,0.06);

    transition: 0.3s ease;
}

/* IMAGE CARD HOVER */
.image-box:hover {

    transform: translateY(-5px);

    border:
    1px solid rgba(139,92,246,0.35);
}

/* IMAGE */
img {

    border-radius: 14px;

    transition: 0.3s ease;

    box-shadow:
    0 4px 15px rgba(0,0,0,0.25);
}

img:hover {
    transform: scale(1.02);
}

/* METRICS */
.metric {

    color: #94a3b8;

    font-size: 14px;

    margin-top: 8px;
}

/* IMAGE FADE ANIMATION */
[data-testid="stImage"] {

    animation: fadeIn 0.5s ease;
}

@keyframes fadeIn {

    from {

        opacity: 0;

        transform: translateY(8px);
    }

    to {

        opacity: 1;

        transform: translateY(0px);
    }
}

/* FOOTER */
.footer {

    text-align: center;

    margin-top: 35px;

    color: #64748b;

    font-size: 14px;
}

</style>
""", unsafe_allow_html=True)

# -----------------------------------
# GENERATOR MODEL
# -----------------------------------
class Generator(nn.Module):

    def __init__(self):

        super().__init__()

        self.label_emb = nn.Embedding(10, 10)

        self.model = nn.Sequential(

            nn.Linear(110, 256),
            nn.ReLU(),

            nn.Linear(256, 512),
            nn.ReLU(),

            nn.Linear(512, 784),
            nn.Tanh()
        )

    def forward(self, noise, labels):

        label_input = self.label_emb(labels)

        x = torch.cat([noise, label_input], dim=1)

        return self.model(x).view(-1, 1, 28, 28)

# -----------------------------------
# LOAD MODEL
# -----------------------------------
@st.cache_resource
def load_model():

    model = Generator()

    model.load_state_dict(
        torch.load("generator.pth", map_location="cpu")
    )

    model.eval()

    return model

generator = load_model()

# -----------------------------------
# TITLE
# -----------------------------------
st.markdown("""
<div class="main-title">
AI Fashion Generator
</div>
""", unsafe_allow_html=True)

st.markdown("""
<div class="sub-title">
Powered by Conditional GANs
</div>
""", unsafe_allow_html=True)

# -----------------------------------
# LAYOUT
# -----------------------------------
left, right = st.columns([1, 1.9], gap="large")

# -----------------------------------
# LEFT PANEL
# -----------------------------------
with left:

    with st.container(border=True):

        st.markdown("## 🎛 Generate Fashion")

        classes = {
            0: "👕 T-shirt",
            1: "👖 Trouser",
            2: "🧥 Pullover",
            3: "👗 Dress",
            4: "🧥 Coat",
            5: "👡 Sandal",
            6: "👔 Shirt",
            7: "👟 Sneaker",
            8: "👜 Bag",
            9: "🥾 Ankle Boot"
        }

        option = st.selectbox(
            "Select Category",
            list(classes.values())
        )

        label = list(classes.keys())[
            list(classes.values()).index(option)
        ]

        num_images = st.slider(
            "Number of Images",
            1,
            10,
            5
        )
        

        generate = st.button(
            "✨ Generate Images"
        )

        st.markdown("---")

        st.markdown(
            f'<div class="metric">Model: Conditional GAN</div>',
            unsafe_allow_html=True
        )

        st.markdown(
            f'<div class="metric">Category: {option}</div>',
            unsafe_allow_html=True
        )

        st.markdown(
            f'<div class="metric">Images: {num_images}</div>',
            unsafe_allow_html=True
        )

# -----------------------------------
# RIGHT PANEL
# -----------------------------------
with right:

    st.markdown("""
    <div class="section-title">
    Generated Images
    </div>
    """, unsafe_allow_html=True)

    if generate:

        with st.spinner("✨ Generating AI fashion concepts..."):

            time.sleep(1)


            noise = torch.randn(num_images, 100)

            labels = torch.tensor([label] * num_images)

            with torch.no_grad():

                images = generator(noise, labels)

            cols = st.columns(num_images)

            for i in range(num_images):
                # ORIGINAL IMAGE
                img_tensor = images[i].unsqueeze(0)

                # UPSCALE
                img_tensor = F.interpolate(
                    img_tensor,
                    size=(128,128),
                    mode="bicubic",
                    align_corners=False
                )

                # CONVERT TO NUMPY
                img = img_tensor.squeeze().numpy()

                # NORMALIZE
                img = (img + 1) / 2

                img = img.clip(0, 1)

                # PIL IMAGE
                pil_img = Image.fromarray(
                    (img * 255).astype("uint8")
                )

                # SAVE BUFFER
                buf = io.BytesIO()

                pil_img.save(buf, format="PNG")

                with cols[i]:
                    st.image(
                        img,
                        use_container_width=True
                    )

                    st.download_button(
                        "⬇ Download",
                        data=buf.getvalue(),
                        file_name=f"fashion_{i}.png",
                        mime="image/png",
                        use_container_width=True
                    )

    else:

        st.info(
            "✨ Your AI-generated fashion images will appear here."
        )

# -----------------------------------
# FOOTER
# -----------------------------------
st.markdown("""
<div class="footer">
Built with PyTorch • Streamlit • Conditional GANs
</div>
""", unsafe_allow_html=True)