import os

import dash
from dash import html, dcc, Input, Output, State

from rag.llm_rag import answer_with_llm

PERSIST_DIR = os.path.join("data", "vectorstore", "chroma_wcag22")


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
                    html.Span(
                        f"  |  dist={dist:.4f}",
                        style={"marginLeft": "12px", "color": "#666"},
                    ),
                ]
            ),
            html.Div(
                [
                    html.Div(
                        [
                            html.Span("Source: "),
                            html.A(url, href=url, target="_blank", rel="noreferrer"),
                        ],
                        style={"marginTop": "8px"},
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
app.title = "WCAG 2.2 RAG Demo (Spec-only)"

app.layout = html.Div(
    style={"maxWidth": "980px", "margin": "24px auto", "fontFamily": "system-ui, Arial"},
    children=[
        html.H2("WCAG 2.2 RAG Demo (Spec-only)"),
        html.Div(
            "Answers questions using normative WCAG 2.2 Success Criterion text from a local vector index (Chroma) + an LLM.",
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
                            style={
                                "width": "100%",
                                "padding": "10px",
                                "borderRadius": "8px",
                                "border": "1px solid #ccc",
                            },
                        ),
                    ],
                ),
                html.Div(
                    style={"width": "240px"},
                    children=[
                        html.Label("Top-k"),
                        dcc.Slider(
                            id="topk",
                            min=3,
                            max=10,
                            step=1,
                            value=5,
                            marks={i: str(i) for i in range(3, 11)},
                            tooltip={"placement": "bottom", "always_visible": False},
                        ),
                    ],
                ),
                html.Button(
                    "Ask",
                    id="search_btn",
                    n_clicks=0,
                    style={
                        "padding": "10px 16px",
                        "borderRadius": "10px",
                        "border": "1px solid #333",
                        "cursor": "pointer",
                    },
                ),
            ],
        ),
        html.Div(id="status", style={"marginTop": "14px"}),
        html.Hr(style={"margin": "18px 0"}),
        html.H3("Answer"),
        dcc.Loading(
            type="default",
            children=html.Div(
                id="answer_box",
                style={
                    "whiteSpace": "pre-wrap",
                    "background": "white",
                    "border": "1px solid #e5e5e5",
                    "borderRadius": "12px",
                    "padding": "14px",
                },
            ),
        ),
        html.H3("Citations"),
        html.Div(id="citations_box"),
        html.Hr(style={"margin": "18px 0"}),
        html.H3("Retrieved chunks (debug)"),
        html.Div(id="results"),
    ],
)


@app.callback(
    Output("status", "children"),
    Output("answer_box", "children"),
    Output("citations_box", "children"),
    Output("results", "children"),
    Input("search_btn", "n_clicks"),
    State("query", "value"),
    State("topk", "value"),
)
def on_search(n_clicks, query, topk):
    if not n_clicks:
        return html.Div("Enter a question and click Ask."), "", "", []

    if not query or not query.strip():
        msg = html.Div("Please enter a question.", style={"color": "crimson"})
        return msg, "", "", []

    if not os.path.exists(PERSIST_DIR):
        msg = html.Div(
            f"Vector index not found at {PERSIST_DIR}. Run: python retrieval/build_index.py",
            style={"color": "crimson"},
        )
        return msg, "", "", []

    try:
        out = answer_with_llm(query.strip(), k=int(topk), model="gpt-4o-mini")
    except Exception as e:
        msg = html.Div(f"RAG call failed: {e}", style={"color": "crimson"})
        return msg, "", "", []

    refused = out.get("refused", False)
    status = html.Div(
        "Refused (low-confidence retrieval). Try rephrasing."
        if refused
        else "Answered from retrieved WCAG 2.2 sources."
    )

    answer_text = out.get("answer", "")
    citations = out.get("citations", []) or []
    results = out.get("results", []) or []

    if citations:
        citations_box = html.Ul(
            [
                html.Li(
                    [
                        html.Span(
                            f"{c.get('sc_id','')} ({c.get('level','')}) — {c.get('sc_title','')} "
                        ),
                        html.A(
                            "source",
                            href=c.get("url", ""),
                            target="_blank",
                            rel="noreferrer",
                        ),
                    ]
                )
                for c in citations
            ]
        )
    else:
        citations_box = html.Div("(No citations)")

    cards = []
    for i, r in enumerate(results):
        cards.append(result_card(i, r["meta"], r["distance"], r["text"]))

    return status, answer_text, citations_box, cards


if __name__ == "__main__":
    app.run_server(debug=True)
