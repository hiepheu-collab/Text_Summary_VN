import gradio as gr
from extractive import extractive_summary
from abstractive import abstractive_summary
from rouge_score import rouge_scorer
import matplotlib.pyplot as plt
import numpy as np
from io import BytesIO
from PIL import Image
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction

smooth = SmoothingFunction().method1

def generate_extractive(article, ext_state):
    """Generate extractive summary and update state.
    If no article provided, return a helpful message and keep previous state.
    """
    # Check for empty input and avoid generating
    if not (article and article.strip()):
        msg = "Không thể sinh tóm tắt Extractive: vui lòng nhập văn bản để tóm tắt."
        return msg, ext_state

    try:
        summary = extractive_summary(article)
    except Exception as e:
        summary = f"Error (extractive): {e}"
    return summary, summary

def generate_abstractive(article, abs_state):
    """Generate abstractive summary and update state.
    If no article provided, return a helpful message and keep previous state.
    """
    # Check for empty input and avoid generating
    if not (article and article.strip()):
        msg = "Không thể sinh tóm tắt Abstractive: vui lòng nhập văn bản để tóm tắt."
        return msg, abs_state

    try:
        summary = abstractive_summary(article)
    except Exception as e:
        summary = f"Error (abstractive): {e}"
    return summary, summary

def evaluate_using_stored(reference, ext_summary, abs_summary):
    """
    Use already-generated summaries (ext_summary, abs_summary).
    Compute BLEU & ROUGE for both and produce a bar chart image.
    Returns: chart_image (PIL)
    """
    ext = ext_summary or ""
    abs_ = abs_summary or ""
    ref = reference or ""

    # BLEU helper
    def safe_tokens(text):
        return [t for t in text.split() if t.strip()]

    # BLEU scores (we compute but won't return them as JSON)
    try:
        ref_tokens = safe_tokens(ref)
        if ref_tokens and ext:
            bleu1_ext = sentence_bleu([ref_tokens], safe_tokens(ext), weights=(1,0,0,0), smoothing_function=smooth)
            bleu4_ext = sentence_bleu([ref_tokens], safe_tokens(ext), weights=(0.25,0.25,0.25,0.25), smoothing_function=smooth)
        else:
            bleu1_ext = bleu4_ext = 0.0

        if ref_tokens and abs_:
            bleu1_abs = sentence_bleu([ref_tokens], safe_tokens(abs_), weights=(1,0,0,0), smoothing_function=smooth)
            bleu4_abs = sentence_bleu([ref_tokens], safe_tokens(abs_), weights=(0.25,0.25,0.25,0.25), smoothing_function=smooth)
        else:
            bleu1_abs = bleu4_abs = 0.0
    except Exception:
        bleu1_ext = bleu4_ext = bleu1_abs = bleu4_abs = 0.0

    # ROUGE scores (compute but keep internal)
    rouge = rouge_scorer.RougeScorer(['rouge1','rouge2','rougeL'], use_stemmer=True)
    try:
        if ref and ext:
            ext_r = rouge.score(ref, ext)
        else:
            ext_r = {'rouge1': None, 'rouge2': None, 'rougeL': None}
        if ref and abs_:
            abs_r = rouge.score(ref, abs_)
        else:
            abs_r = {'rouge1': None, 'rouge2': None, 'rougeL': None}
    except Exception:
        ext_r = abs_r = {'rouge1': None, 'rouge2': None, 'rougeL': None}

    # Prepare numeric values for plotting (use 0.0 when unavailable)
    metrics = ["BLEU-1", "BLEU-4", "ROUGE-1", "ROUGE-2", "ROUGE-L"]
    ext_vals = [
        float(bleu1_ext or 0.0),
        float(bleu4_ext or 0.0),
        float(ext_r['rouge1'].fmeasure) if ext_r['rouge1'] is not None else 0.0,
        float(ext_r['rouge2'].fmeasure) if ext_r['rouge2'] is not None else 0.0,
        float(ext_r['rougeL'].fmeasure) if ext_r['rougeL'] is not None else 0.0,
    ]
    abs_vals = [
        float(bleu1_abs or 0.0),
        float(bleu4_abs or 0.0),
        float(abs_r['rouge1'].fmeasure) if abs_r['rouge1'] is not None else 0.0,
        float(abs_r['rouge2'].fmeasure) if abs_r['rouge2'] is not None else 0.0,
        float(abs_r['rougeL'].fmeasure) if abs_r['rougeL'] is not None else 0.0,
    ]

    # Plot
    x = np.arange(len(metrics))
    width = 0.35
    fig, ax = plt.subplots(figsize=(9,4))
    ax.bar(x - width/2, ext_vals, width, label='Extractive', color="#66B2FF")
    ax.bar(x + width/2, abs_vals, width, label='Abstractive', color="#FF9999")
    ax.set_ylabel("Score")
    ax.set_title("Comparison: Extractive vs Abstractive")
    ax.set_xticks(x)
    ax.set_xticklabels(metrics)
    ax.set_ylim(0, 1)
    ax.legend(loc='upper right')
    for i, v in enumerate(ext_vals):
        ax.text(i - width/2, v + 0.02, f"{v:.3f}", ha='center', fontsize=9)
    for i, v in enumerate(abs_vals):
        ax.text(i + width/2, v + 0.02, f"{v:.3f}", ha='center', fontsize=9)
    plt.tight_layout()

    # Convert plot to PIL Image
    buf = BytesIO()
    fig.savefig(buf, format="png")
    plt.close(fig)
    buf.seek(0)
    img = Image.open(buf).convert("RGB")
    return img


# Build Gradio app
with gr.Blocks() as demo:
    gr.Markdown("## Text Summarization (Extractive / Abstractive) — Reuse generated summaries for evaluation")

    # Shared article box
    article_box = gr.Textbox(label="Văn bản gốc (article) — dùng cho cả 2 phương pháp", lines=8, placeholder="Dán bài báo ở đây...")

    with gr.Row():
        with gr.Column():
            gr.Markdown("### Extractive")
            ex_btn = gr.Button("Sinh tóm tắt Extractive")
            ex_out = gr.Textbox(label="Kết quả Extractive", interactive=False, lines=6)
            ext_state = gr.State("")  # hold generated extractive summary
        with gr.Column():
            gr.Markdown("### Abstractive")
            ab_btn = gr.Button("Sinh tóm tắt Abstractive")
            ab_out = gr.Textbox(label="Kết quả Abstractive", interactive=False, lines=6)
            abs_state = gr.State("")  # hold generated abstractive summary

    # connect buttons to generation functions and update states
    ex_btn.click(fn=generate_extractive,
                 inputs=[article_box, ext_state],
                 outputs=[ex_out, ext_state])
    ab_btn.click(fn=generate_abstractive,
                 inputs=[article_box, abs_state],
                 outputs=[ab_out, abs_state])

    gr.Markdown("## Đánh giá & So sánh (sử dụng các tóm tắt đã sinh)")
    ref_box = gr.Textbox(label="Tóm tắt gốc (reference) — để trống nếu không có", lines=4)
    eval_plot = gr.Image(label="Biểu đồ so sánh", type="pil")

    # Evaluate using stored states (ext_state, abs_state) — only return the image
    eval_btn = gr.Button("Đánh giá & Vẽ biểu đồ")
    eval_btn.click(fn=evaluate_using_stored,
                   inputs=[ref_box, ext_state, abs_state],
                   outputs=[eval_plot])

    gr.Markdown("Ghi chú: Sinh tóm tắt trước rồi nhấn 'Đánh giá & Vẽ biểu đồ' để dùng các tóm tắt đã tạo. Nếu chưa sinh tóm tắt, biểu đồ sẽ hiển thị 0 cho các chỉ số.")

if __name__ == "__main__":
    demo.launch()
