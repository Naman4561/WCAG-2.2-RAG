import os
from functools import lru_cache

import dash
from dash import html, dcc, Input, Output, State
import chromadb
from sentence_transformers import SentenceTransformer

PERSIST_DIR = os.path.join("data", "vectorstore", "chroma_wcag22")
COLLECTION_NAME = "wcag22_spec"
EMBED_MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"


@lru_cache(maxsize=1)
def get_model():
    return SentenceTransformer(EMBED_MODEL_NAME)


@lru_cache(maxsize=1)
def get_collection():
    client = chromadb.PersistentClient(path=PERSIST_DIR)
    return client.get_collection(COLLECTION_NAME)


def run_search(query: str, k: int = 5):
    model = get_model()
    col = get_collection()
    q_emb = model.encode([query], normalize_embeddings=True)[0].tolist()

    return col.query(
        query_embeddings=[q_emb],
        n_results=k,
        include=["documents", "metadatas", "distances"]
    )


def result_card(i, meta, dist, doc):
    title = f"{meta.get('sc_id')} — {meta.get('sc_title')}"
    level = meta.get("level") or "Unknown"
    url = meta.get("url") or ""
    snippet = doc[:1200] + ("…" if len(doc) > 1200 else "")

    return html.Details(
        open=(i == 0),
        children=[
            html.Summary(
                [
                    html.Span(f"{i+1}. {title}", style={"fontWeight": 600}),
                    html.Span(f"  |  Level: {level}", style={"marginLeft": "12px"}),
                    html.Span(f"  |  dist={dist:.4f}", style={"marginLeft": "12px", "color": "#666"}),
                ]
            ),
            html.Div(
                [
                    html.Div(
                        [
                            html.Span("Source: "),
                            html.A(url, href=url, target="_blank", rel="noreferrer"),
                        ],
                        style={"marginTop": "8px"}
                    ),
                    html.Pre(
                        snippet,
                        style={
                            "whiteSpace": "pre-wrap",
                            "background": "#f7f7f7",
                            "padding": "12px",
                            "borderRadius": "8px",
                            "marginTop": "8px",
                            "border": "1px solid #eee",
                        },
                    ),
                ],
                style={"padding": "8px 0"},
            ),
        ],
        style={
            "border": "1px solid #e5e5e5",
            "borderRadius": "10px",
            "padding": "10px 12px",
            "marginBottom": "10px",
            "background": "white",
        },
    )


app = dash.Dash(__name__)
app.title = "WCAG 2.2 Retrieval Demo (Spec-only)"

app.layout = html.Div(
    style={"maxWidth": "980px", "margin": "24px auto", "fontFamily": "system-ui, Arial"},
    children=[
        html.H2("WCAG 2.2 Retrieval Demo (Spec-only)"),
        html.Div(
            "Retrieves normative WCAG 2.2 Success Criterion text from a local vector index (Chroma).",
            style={"color": "#555", "marginBottom": "16px"},
        ),

        html.Div(
            style={"display": "flex", "gap": "12px", "alignItems": "flex-end"},
            children=[
                html.Div(
                    style={"flex": 1},
                    children=[
                        html.Label("Question"),
                        dcc.Input(
                            id="query",
                            type="text",
                            value="Is focus allowed to be obscured?",
                            style={"width": "100%", "padding": "10px", "borderRadius": "8px", "border": "1px solid #ccc"},
                        ),
                    ],
                ),
                html.Div(
                    style={"width": "240px"},
                    children=[
                        html.Label("Top-k"),
                        dcc.Slider(
                            id="topk",
                            min=3, max=10, step=1, value=5,
                            marks={i: str(i) for i in range(3, 11)},
                            tooltip={"placement": "bottom", "always_visible": False},
                        ),
                    ],
                ),
                html.Button(
                    "Search",
                    id="search_btn",
                    n_clicks=0,
                    style={"padding": "10px 16px", "borderRadius": "10px", "border": "1px solid #333", "cursor": "pointer"},
                ),
            ],
        ),

        html.Div(id="status", style={"marginTop": "14px"}),
        html.Hr(style={"margin": "18px 0"}),

        html.Div(id="results"),
    ],
)


@app.callback(
    Output("status", "children"),
    Output("results", "children"),
    Input("search_btn", "n_clicks"),
    State("query", "value"),
    State("topk", "value"),
)
def on_search(n_clicks, query, topk):
    if not n_clicks:
        return html.Div("Enter a question and click Search."), []

    if not query or not query.strip():
        return html.Div("Please enter a question.", style={"color": "crimson"}), []

    # Basic index existence check
    if not os.path.exists(PERSIST_DIR):
        return html.Div(
            f"Vector index not found at {PERSIST_DIR}. Run: python retrieval/build_index.py",
            style={"color": "crimson"},
        ), []

    try:
        out = run_search(query.strip(), int(topk))
    except Exception as e:
        return html.Div(f"Search failed: {e}", style={"color": "crimson"}), []

    ids = out.get("ids", [[]])[0]
    if not ids:
        return html.Div("No results returned."), []

    cards = []
    for i in range(len(ids)):
        meta = out["metadatas"][0][i]
        dist = out["distances"][0][i]
        doc = out["documents"][0][i]
        cards.append(result_card(i, meta, dist, doc))

    return html.Div(f"Retrieved {len(ids)} chunks."), cards


if __name__ == "__main__":
    # Tip: set debug=False once stable
    app.run_server(debug=True)
