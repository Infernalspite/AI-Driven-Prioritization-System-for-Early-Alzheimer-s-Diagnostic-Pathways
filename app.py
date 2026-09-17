import gradio as gr 
def analyze_patient(age, score): 
    if score < 24: return "High Priority - Recommend Escalation to Blood Biomarkers / MRI" 
    return "Low Priority - Maintain Longitudinal Monitoring" 
with gr.Blocks(theme=gr.themes.Soft()) as demo: 
    gr.Markdown("# AI-Driven Prioritization System for Early Alzheimer's Diagnostics") 
    with gr.Row(): 
        with gr.Column(): 
            age = gr.Number(label="Patient Age", value=70) 
            score = gr.Slider(label="Cognitive Screening Score (e.g. MMSE)", minimum=0, maximum=30, value=25) 
            btn = gr.Button("Evaluate Priority Level") 
        with gr.Column(): 
            output = gr.Textbox(label="System Recommendation") 
    btn.click(fn=analyze_patient, inputs=[age, score], outputs=output) 
demo.launch() 
