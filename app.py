"""
RetailMind — Self-Healing LLM for Store Intelligence

Gradio application showcasing real-time semantic drift detection,
autonomous prompt adaptation, and hybrid RAG retrieval.
"""

import logging
import sys

# ── HF Hub Compatibility Monkeypatch ───────────────────────────────────────
# Fix for: ImportError: cannot import name 'HfFolder' from 'huggingface_hub'
# This happens in HF Spaces with newer hub versions and older Gradio versions.
try:
    import huggingface_hub
    if not hasattr(huggingface_hub, "HfFolder"):
        class MockHfFolder:
            @staticmethod
            def get_token(): return None
            @staticmethod
            def save_token(token): pass
            @staticmethod
            def delete_token(): pass
        huggingface_hub.HfFolder = MockHfFolder
except ImportError:
    pass

import gradio as gr
import plotly.graph_objects as go
from modules.data_simulation import generate_catalog, get_scenarios
from modules.retrieval import HybridRetriever
from modules.drift import DriftDetector
from modules.adaptation import Adapter
from modules.llm import generate_response

# ── Logging ────────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s │ %(name)-24s │ %(levelname)-5s │ %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("retailmind")

# ── Initialize components ─────────────────────────────────────────────────
logger.info("Bootstrapping RetailMind…")
catalog = generate_catalog()
retriever = HybridRetriever(catalog)
detector = DriftDetector()
adapter = Adapter()
scenarios = get_scenarios()
logger.info("Ready — %d products indexed.", len(catalog))


# ── Helper: Image mapping ─────────────────────────────────────────────────
IMAGE_MAP = {
    "Parka": "https://images.unsplash.com/photo-1544923246-77307dd270b5?w=400&h=300&fit=crop",
    "Sweater": "https://images.unsplash.com/photo-1610652492500-dea0624af6ee?w=400&h=300&fit=crop",
    "Gloves": "https://images.unsplash.com/photo-1551538827-9c037cb4f32a?w=400&h=300&fit=crop",
    "Boots": "https://images.unsplash.com/photo-1608256246200-53e635b5b65f?w=400&h=300&fit=crop",
    "Beanie": "https://images.unsplash.com/photo-1576871337622-98d48d1cf531?w=400&h=300&fit=crop",
    "Fleece": "https://images.unsplash.com/photo-1591047139829-d91aecb6caea?w=400&h=300&fit=crop",
    "Base Layer": "https://images.unsplash.com/photo-1489987707025-afc232f7ea0f?w=400&h=300&fit=crop",
    "Vest": "https://images.unsplash.com/photo-1591047139829-d91aecb6caea?w=400&h=300&fit=crop",
    "Sneakers": "https://images.unsplash.com/photo-1542291026-7eec264c27ff?w=400&h=300&fit=crop",
    "Shorts": "https://images.unsplash.com/photo-1591195853828-11db59a44f6b?w=400&h=300&fit=crop",
    "Sunglasses": "https://images.unsplash.com/photo-1511499767150-a48a237f0083?w=400&h=300&fit=crop",
    "Linen": "https://images.unsplash.com/photo-1596755094514-f87e34085b2c?w=400&h=300&fit=crop",
    "Sandals": "https://images.unsplash.com/photo-1603487742131-4160ec999306?w=400&h=300&fit=crop",
    "Tank": "https://images.unsplash.com/photo-1521572163474-6864f9cf17ab?w=400&h=300&fit=crop",
    "Hat": "https://images.unsplash.com/photo-1521369909029-2afed882baee?w=400&h=300&fit=crop",
    "Water Shoes": "https://images.unsplash.com/photo-1542291026-7eec264c27ff?w=400&h=300&fit=crop",
    "Backpack": "https://images.unsplash.com/photo-1553062407-98eeb64c6a62?w=400&h=300&fit=crop",
    "Bottle": "https://images.unsplash.com/photo-1602143407151-7111542de6e8?w=400&h=300&fit=crop",
    "Tee": "https://images.unsplash.com/photo-1521572163474-6864f9cf17ab?w=400&h=300&fit=crop",
    "Tote": "https://images.unsplash.com/photo-1622560480605-d83c853bc5c3?w=400&h=300&fit=crop",
    "Shoes": "https://images.unsplash.com/photo-1542291026-7eec264c27ff?w=400&h=300&fit=crop",
    "Jacket": "https://images.unsplash.com/photo-1551028719-00167b16eac5?w=400&h=300&fit=crop",
    "Watch": "https://images.unsplash.com/photo-1523275335684-37898b6baf30?w=400&h=300&fit=crop",
    "Mat": "https://images.unsplash.com/photo-1553062407-98eeb64c6a62?w=400&h=300&fit=crop", # reusing backpack as mat placeholder
    "Tights": "https://images.unsplash.com/photo-1556821840-3a63f95609a7?w=400&h=300&fit=crop", # reusing hoodie as apparel placeholder
    "Pack": "https://images.unsplash.com/photo-1553062407-98eeb64c6a62?w=400&h=300&fit=crop",
    "Headphones": "https://images.unsplash.com/photo-1505740420928-5e560c06d30e?w=400&h=300&fit=crop",
    "Tracker": "https://images.unsplash.com/photo-1557438159-51eec7a6c9e8?w=400&h=300&fit=crop",
    "Earbuds": "https://images.unsplash.com/photo-1590658268037-6bf12f032f55?w=400&h=300&fit=crop",
    "Charger": "https://images.unsplash.com/photo-1609091839311-d5365f9ff1c5?w=400&h=300&fit=crop",
    "Speaker": "https://images.unsplash.com/photo-1608043152269-423dbba4e7e1?w=400&h=300&fit=crop",
    "Lamp": "https://images.unsplash.com/photo-1507473885765-e6ed057ab6fe?w=400&h=300&fit=crop",
    "Power Bank": "https://images.unsplash.com/photo-1609091839311-d5365f9ff1c5?w=400&h=300&fit=crop",
    "Mug": "https://images.unsplash.com/photo-1514228742587-6b1558fcca3d?w=400&h=300&fit=crop",
    "Weekender": "https://images.unsplash.com/photo-1590874103328-eac38a683ce7?w=400&h=300&fit=crop",
    "Overcoat": "https://images.unsplash.com/photo-1544923246-77307dd270b5?w=400&h=300&fit=crop",
    "Wallet": "https://images.unsplash.com/photo-1627123424574-724758594e93?w=400&h=300&fit=crop",
    "Belt": "https://images.unsplash.com/photo-1553062407-98eeb64c6a62?w=400&h=300&fit=crop",
    "Candle": "https://images.unsplash.com/photo-1602607616777-b8fbdc2cd8a9?w=400&h=300&fit=crop",
    "Blanket": "https://images.unsplash.com/photo-1555041469-a586c61ea9bc?w=400&h=300&fit=crop",
    "Clock": "https://images.unsplash.com/photo-1563861826100-9cb868fdbe1c?w=400&h=300&fit=crop",
    "Sunscreen": "https://images.unsplash.com/photo-1556228578-83b6329731eb?w=400&h=300&fit=crop",
    "Lipstick": "https://images.unsplash.com/photo-1586495777744-4413f21062fa?w=400&h=300&fit=crop",
    "Serum": "https://images.unsplash.com/photo-1620916566398-39f1143ab7be?w=400&h=300&fit=crop",
    "Lip Balm": "https://images.unsplash.com/photo-1629813359670-357ff8ca8e21?w=400&h=300&fit=crop",
    "Towel": "https://images.unsplash.com/photo-1583845112203-29329902332e?w=400&h=300&fit=crop",
    "Hoodie": "https://images.unsplash.com/photo-1556821840-3a63f95609a7?w=400&h=300&fit=crop",
    "Chino": "https://images.unsplash.com/photo-1473966968600-fa801b869a1a?w=400&h=300&fit=crop",
    "Crossbody": "https://images.unsplash.com/photo-1590874103328-eac38a683ce7?w=400&h=300&fit=crop",
    "Socks": "https://images.unsplash.com/photo-1586350977771-b3b0abd50c82?w=400&h=300&fit=crop",
    "Basketball": "https://images.unsplash.com/photo-1546519638-68e109498ffc?w=400&h=300&fit=crop",
    "Jersey": "https://images.unsplash.com/photo-1565299624946-b28f40a0ae38?w=400&h=300&fit=crop",
    "Cushion": "https://images.unsplash.com/photo-1555041469-a586c61ea9bc?w=400&h=300&fit=crop",
    "Planter": "https://images.unsplash.com/photo-1459411552884-841db9b3cc2a?w=400&h=300&fit=crop",
    "Organizer": "https://images.unsplash.com/photo-1507473885765-e6ed057ab6fe?w=400&h=300&fit=crop",
    "Pour-Over": "https://images.unsplash.com/photo-1495474472287-4d71bcdd2085?w=400&h=300&fit=crop",
}

DEFAULT_IMG = "https://images.unsplash.com/photo-1542291026-7eec264c27ff?w=400&h=300&fit=crop" # use shoes as fallback so it never 404s


def _get_product_image(title: str) -> str:
    """Map product title → curated Unsplash photo."""
    for key, url in IMAGE_MAP.items():
        if key.lower() in title.lower():
            return url
    return DEFAULT_IMG


# ── Plotly drift chart ────────────────────────────────────────────────────

def _plot_drift() -> go.Figure:
    series = detector.get_history_series()
    ewma = detector.get_ewma_scores()
    fig = go.Figure()

    colors = {"price_sensitive": "#f59e0b", "summer_shift": "#06b6d4", "eco_trend": "#10b981"}
    labels = {"price_sensitive": "Price Sensitivity", "summer_shift": "Summer Shift", "eco_trend": "Eco Trend"}

    for concept in series:
        data = series[concept][-30:]  # last 30 data points
        fig.add_trace(go.Scatter(
            y=data,
            mode="lines",
            name=labels.get(concept, concept),
            line=dict(color=colors.get(concept, "#fff"), width=2.5, shape="spline"),
            fill="tozeroy",
            fillcolor=colors.get(concept, "#fff").replace(")", ", 0.08)").replace("rgb", "rgba") if "rgb" in colors.get(concept, "") else f"rgba(255,255,255,0.05)",
        ))

    # Threshold line
    fig.add_hline(y=0.38, line_dash="dot", line_color="rgba(255,255,255,0.3)",
                  annotation_text="Threshold", annotation_font_color="rgba(255,255,255,0.4)")

    fig.update_layout(
        height=240,
        margin=dict(l=0, r=0, t=10, b=0),
        plot_bgcolor="rgba(0,0,0,0)",
        paper_bgcolor="rgba(0,0,0,0)",
        font=dict(color="#94a3b8", size=11),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="center", x=0.5,
                    font=dict(size=10)),
        xaxis=dict(showgrid=False, showticklabels=False),
        yaxis=dict(showgrid=True, gridwidth=1, gridcolor="rgba(255,255,255,0.06)",
                   range=[0, 0.8]),
    )
    return fig


# ── Product cards HTML ────────────────────────────────────────────────────

def _build_product_html(retrieved: list[dict]) -> str:
    if not retrieved:
        return _empty_catalog_html()

    cards = []
    for r in retrieved:
        p = r["product"]
        score = r["score"]
        img = _get_product_image(p["title"])
        stars_full = int(p.get("rating", 4))
        stars_html = "★" * stars_full + "☆" * (5 - stars_full)
        reviews = p.get("reviews", 0)
        score_pct = int(score * 100)
        tags_html = "".join(
            f"<span style='background:rgba(99,102,241,0.15); color:#818cf8; padding:2px 8px; "
            f"border-radius:20px; font-size:10px; margin-right:4px;'>{t}</span>"
            for t in p.get("tags", [])[:3]
        )

        cards.append(f"""
        <div style='background:rgba(255,255,255,0.03); border:1px solid rgba(255,255,255,0.08);
                     border-radius:16px; overflow:hidden; transition:all 0.3s ease;
                     box-shadow:0 4px 20px rgba(0,0,0,0.3);'>
            <div style='position:relative;'>
                <img src='{img}' style='width:100%; height:150px; object-fit:cover;
                     border-bottom:1px solid rgba(255,255,255,0.06);'
                     onerror="this.src='{DEFAULT_IMG}'" />
                <div style='position:absolute; top:8px; right:8px; background:rgba(0,0,0,0.75);
                     color:#f8fafc; padding:3px 10px; border-radius:20px; font-size:13px;
                     font-weight:700; backdrop-filter:blur(8px);
                     border:1px solid rgba(255,255,255,0.15);'>
                    ${p['price']:.2f}
                </div>
                <div style='position:absolute; top:8px; left:8px; background:rgba(99,102,241,0.85);
                     color:white; padding:2px 8px; border-radius:20px; font-size:10px;
                     font-weight:600; letter-spacing:0.5px;'>
                    {score_pct}% match
                </div>
            </div>
            <div style='padding:14px;'>
                <div style='color:#f1f5f9; font-size:14px; font-weight:600;
                     margin-bottom:4px; line-height:1.3;'>{p['title']}</div>
                <div style='display:flex; align-items:center; gap:6px; margin-bottom:6px;'>
                    <span style='color:#fbbf24; font-size:12px; letter-spacing:1px;'>{stars_html}</span>
                    <span style='color:#64748b; font-size:11px;'>({reviews:,})</span>
                </div>
                <div style='margin-bottom:8px;'>{tags_html}</div>
                <p style='color:#94a3b8; font-size:12px; line-height:1.4; margin:0;'>
                    {p['desc'][:100]}…
                </p>
            </div>
        </div>
        """)

    return f"""
    <div style='display:grid; grid-template-columns:1fr 1fr; gap:16px; padding:8px;'>
        {''.join(cards)}
    </div>
    """


def _empty_catalog_html() -> str:
    return """
    <div style='padding:60px 30px; text-align:center; color:#475569;
                border:2px dashed rgba(255,255,255,0.08); border-radius:20px; margin:16px;'>
        <div style='font-size:2.5rem; margin-bottom:12px;'>🛍️</div>
        <div style='font-size:1.1rem; font-weight:500; color:#64748b;'>Awaiting your query…</div>
        <div style='font-size:0.85rem; color:#475569; margin-top:6px;'>
            Try a scenario below or type your own question
        </div>
    </div>
    """


# ── Main query handler ────────────────────────────────────────────────────

def process_query(query: str, history: list):
    if not query or not query.strip():
        return "", history, _plot_drift(), "", "—", _empty_catalog_html()

    logger.info("Processing query: %r", query)

    # 1. Measure drift
    drift_state, scores = detector.analyze_drift(query)

    # 2. Retrieve products (hybrid: price-filter + semantic)
    retrieved = retriever.search(query, top_k=4)

    # 3. Adapt system prompt
    system_prompt = adapter.adapt_prompt(drift_state)
    explanation = adapter.get_explanation(drift_state)
    label = adapter.get_label(drift_state)

    # 4. Generate LLM response
    response = generate_response(system_prompt, query, retrieved)

    history = history or []
    history.append({"role": "user", "content": query})
    history.append({"role": "assistant", "content": response})

    return "", history, _plot_drift(), explanation, label, _build_product_html(retrieved)


def reset_chat():
    global detector, adapter
    detector = DriftDetector()
    adapter = Adapter()
    return (
        "",
        [],
        _plot_drift(),
        ("📊 System Status: Normal\n"
         "━━━━━━━━━━━━━━━━━━━━━━━━━━\n"
         "No significant drift detected.\n"
         "System prompt: Default balanced mode.\n"
         "All EWMA concept scores below threshold (0.38)."),
        "⚖️ Balanced Mode",
        _empty_catalog_html()
    )


def load_example(example_text: str) -> str:
    return example_text


# ══════════════════════════════════════════════════════════════════════════
# UI Definition
# ══════════════════════════════════════════════════════════════════════════

css = """
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800&display=swap');

body, .gradio-container {
    font-family: 'Inter', system-ui, -apple-system, sans-serif !important;
    background: #0a0f1a !important;
}

/* Header */
.hero-header {
    text-align: center;
    padding: 2.5rem 2rem 1.5rem;
    background: linear-gradient(135deg, rgba(15,23,42,0.95) 0%, rgba(30,41,59,0.6) 50%, rgba(15,23,42,0.95) 100%);
    border-radius: 24px;
    border: 1px solid rgba(255,255,255,0.06);
    box-shadow: 0 25px 60px rgba(0,0,0,0.5);
    position: relative;
    overflow: hidden;
    margin-bottom: 1.5rem;
}
.hero-header::before {
    content: '';
    position: absolute;
    top: -50%;
    left: -50%;
    width: 200%;
    height: 200%;
    background: radial-gradient(circle at 30% 50%, rgba(99,102,241,0.08) 0%, transparent 50%),
                radial-gradient(circle at 70% 50%, rgba(6,182,212,0.06) 0%, transparent 50%);
    animation: aurora 8s ease-in-out infinite alternate;
}
@keyframes aurora {
    0% { transform: translate(0, 0) rotate(0deg); }
    100% { transform: translate(-5%, 5%) rotate(3deg); }
}
.hero-title {
    font-size: 2.8rem;
    font-weight: 800;
    background: linear-gradient(135deg, #818cf8 0%, #06b6d4 50%, #10b981 100%);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    margin: 0;
    position: relative;
    letter-spacing: -0.5px;
}
.hero-sub {
    color: #64748b;
    font-size: 0.95rem;
    letter-spacing: 3px;
    text-transform: uppercase;
    font-weight: 500;
    margin-top: 0.5rem;
    position: relative;
}
.hero-badges {
    display: flex;
    justify-content: center;
    gap: 12px;
    margin-top: 1rem;
    position: relative;
    flex-wrap: wrap;
}
.hero-badge {
    background: rgba(255,255,255,0.04);
    border: 1px solid rgba(255,255,255,0.08);
    color: #94a3b8;
    padding: 4px 14px;
    border-radius: 20px;
    font-size: 0.75rem;
    font-weight: 500;
    letter-spacing: 0.5px;
}

/* Panels */
.glass-panel {
    background: rgba(15, 23, 42, 0.6) !important;
    border: 1px solid rgba(255,255,255,0.06) !important;
    border-radius: 20px !important;
    backdrop-filter: blur(12px) !important;
}

/* Scenario pills */
.scenario-row { display: flex; gap: 8px; flex-wrap: wrap; margin-top: 8px; }

/* Section headers */
.panel-header {
    color: #e2e8f0;
    font-size: 1rem;
    font-weight: 600;
    padding: 14px 16px 8px;
    display: flex;
    align-items: center;
    gap: 8px;
}

/* Info box */
.info-callout {
    background: rgba(99,102,241,0.08);
    border: 1px solid rgba(99,102,241,0.2);
    border-radius: 12px;
    padding: 12px 16px;
    color: #a5b4fc;
    font-size: 0.8rem;
    line-height: 1.5;
    margin: 8px 12px;
}

/* Hide Gradio footer */
footer { display: none !important; }
"""

with gr.Blocks(title="RetailMind — Self-Healing AI", css=css, theme=gr.themes.Base()) as app:

    # ── Header ────────────────────────────────────────────────────
    gr.HTML("""
    <div class="hero-header">
        <h1 class="hero-title">RetailMind</h1>
        <p class="hero-sub">Self-Healing LLM · Store Intelligence</p>
        <div class="hero-badges">
            <span class="hero-badge">🧠 Semantic Drift Detection</span>
            <span class="hero-badge">🔄 Autonomous Prompt Healing</span>
            <span class="hero-badge">🔍 Hybrid RAG Retrieval</span>
            <span class="hero-badge">📊 Real-Time Telemetry</span>
        </div>
    </div>
    """)

    with gr.Row():
        # ── LEFT: Chat Panel ─────────────────────────────────────
        with gr.Column(scale=4, elem_classes=["glass-panel"]):
            gr.HTML("<div class='panel-header'>💬 AI Shopping Assistant</div>")
            gr.HTML("""
            <div style="padding: 10px 14px; margin-bottom: 12px; background: rgba(59, 130, 246, 0.1); border: 1px solid rgba(59, 130, 246, 0.2); border-radius: 8px; font-size: 0.85em; color: #93c5fd; line-height: 1.4;">
                <b>🏷️ In Stock:</b> Outerwear & Apparel <span>·</span> Footwear <span>·</span> Tech Accessories <span>·</span> Home & Lifestyle <span>·</span> Health & Beauty
            </div>
            """)
            chatbot = gr.Chatbot(
                height=420,
                container=False,
                type="messages",
                placeholder="Ask me about products, deals, or seasonal picks…",
            )
            with gr.Row():
                msg = gr.Textbox(
                    placeholder="e.g. Find me eco-friendly running shoes under $120…",
                    show_label=False,
                    container=False,
                    scale=8,
                )
                submit = gr.Button("Search", variant="primary", scale=2)
                reset_btn = gr.Button("🔄 Reset", variant="secondary", scale=1)

            gr.HTML("""
            <div class='info-callout'>
                💡 <b>Demo tip:</b> Click the scenario buttons below in order
                (Phase 1 → 4) to watch the system detect intent drift and
                autonomously heal its behavior in real time.
            </div>
            """)

            for scenario_name, queries in scenarios.items():
                with gr.Accordion(scenario_name, open=False):
                    for q in queries:
                        btn = gr.Button(q, size="sm", variant="secondary")
                        btn.click(fn=load_example, inputs=btn, outputs=msg)

        # ── MIDDLE: Product Feed ─────────────────────────────────
        with gr.Column(scale=4, elem_classes=["glass-panel"]):
            gr.HTML("<div class='panel-header'>🛍️ Retrieved Products</div>")
            retrieved_box = gr.HTML(value=_empty_catalog_html())

        # ── RIGHT: MLOps Telemetry ───────────────────────────────
        with gr.Column(scale=3, elem_classes=["glass-panel"]):
            gr.HTML("<div class='panel-header'>⚡ MLOps Telemetry</div>")

            current_phase = gr.Textbox(
                label="Active Semantic State",
                value="⚖️ Balanced Mode",
                interactive=False,
            )

            drift_plot = gr.Plot(value=_plot_drift())

            gr.HTML("""
            <div class='info-callout'>
                📈 The chart above tracks <b>EWMA-smoothed</b> semantic
                similarity between user queries and concept anchors
                (price, season, eco). When a line crosses the dotted
                threshold, the system <b>autonomously rewrites</b> its
                own instructions.
            </div>
            """)

            gr.HTML("<div class='panel-header'>🧠 Self-Healing Log</div>")
            explanation_box = gr.Textbox(
                label="Adaptation Status",
                interactive=False,
                lines=6,
                value=(
                    "📊 System Status: Normal\n"
                    "━━━━━━━━━━━━━━━━━━━━━━━━━━\n"
                    "No significant drift detected.\n"
                    "System prompt: Default balanced mode.\n"
                    "All EWMA concept scores below threshold (0.38)."
                ),
            )

    # ── Event wiring ──────────────────────────────────────────────
    submit.click(
        process_query,
        inputs=[msg, chatbot],
        outputs=[msg, chatbot, drift_plot, explanation_box, current_phase, retrieved_box],
    )
    msg.submit(
        process_query,
        inputs=[msg, chatbot],
        outputs=[msg, chatbot, drift_plot, explanation_box, current_phase, retrieved_box],
    )
    reset_btn.click(
        reset_chat,
        inputs=None,
        outputs=[msg, chatbot, drift_plot, explanation_box, current_phase, retrieved_box],
    )


if __name__ == "__main__":
    app.launch(server_name="0.0.0.0", share=True)
