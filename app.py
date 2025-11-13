import gradio as gr
from main import predict_iris

import gradio.themes as themes

# Create the Gradio interface
custom_theme = themes.Default(
    primary_hue="red",
    secondary_hue="pink",
    neutral_hue="slate"
)

interface = gr.Interface(
    fn=predict_iris,
    inputs=[
        gr.Number(label="Sepal length"),
        gr.Number(label="Sepal width"),
        gr.Number(label="petal length"),
        gr.Number(label="petal width"),
    ],
    outputs=[
        gr.Textbox(label="Predicted Flower"),
        gr.Image(type="filepath", label="Flower Image")

    ],
    theme=custom_theme
)

interface.launch()