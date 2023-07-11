#!/usr/bin/python
import re

import fire
import gradio as gr

from modules.settings import Settings
from modules.template import Template
from modules.model import Model


def ui(model : Model, settings : Settings, template : Template):
    def query(query : str, temp : float = 0.05, top_p: float = 0.9, top_k : int = 50):
        full = ""
        request = template.apply(instruction=query, output="")
        for r in model.generate(request, limit=256, temp=temp, top_p=top_p, top_k=top_k):
            print(r, end="")
            full += r
            yield full.strip()

    with gr.Blocks() as app:
        if settings.ui.title:
            gr.Markdown(value=f"**{settings.ui.title}**")
        input = gr.Textbox(label="Input")
        output = gr.Textbox(label="Output")
        with gr.Accordion("Settings", open=False):
            temp = gr.Slider(minimum=0, maximum=1, value=0.1, label="Temperature")
            top_p = gr.Slider(minimum=0, maximum=1, value=0.75, label="Top p")
            top_k = gr.Slider(minimum=0, maximum=100, step=1, value=40, label="Top k")

        input.submit(query, inputs=[input, temp, top_p, top_k], outputs=output)

    app.queue().launch()   


def main(s : str = None, q : str = None):
    assert s, "Must provide settings file name"
    settings = Settings(s)
    settings.print()

    model = Model(settings, trainable=False)
    templatePath = settings.ui.templatePath
    if not templatePath:
        templatePath = settings.templatePath
    template = Template(templatePath)
    
    if q:
        request = template.apply(input=q, output="")
        result = model.generate(request)
        for r in result:
            print(r, end="", flush=True)
        print()

    else:
        ui(model, settings, template)


if __name__ == "__main__":
    fire.Fire(main)
