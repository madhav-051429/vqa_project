import uvicorn
from fastapi import FastAPI
import gradio as gr
from app.api import app as api_app
from app.gradio_interface import create_gradio_interface

demo = create_gradio_interface()
app = gr.mount_gradio_app(api_app, demo, path="/")

if __name__ == "__main__":
    demo.launch(share=True)  # This creates a public link

