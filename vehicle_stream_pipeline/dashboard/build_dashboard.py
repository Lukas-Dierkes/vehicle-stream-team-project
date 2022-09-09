import warnings

import dash
import dash_bootstrap_components as dbc
import dash_labs as dl
from dash import Input, Output, State, html

warnings.filterwarnings("ignore")

app = dash.Dash(
    __name__, plugins=[dl.plugins.pages], external_stylesheets=[dbc.themes.BOOTSTRAP]
)

# Code inspired and copied from https://github.com/Coding-with-Adam/Dash-by-Plotly/tree/master/Dash_More_Advanced_Shit/Intro%20to%20Python%20multipage/App-A

offcanvas = html.Div(
    [
        dbc.Button("Explore", id="open-offcanvas", n_clicks=0),
        dbc.Offcanvas(
            dbc.ListGroup(
                [
                    dbc.ListGroupItem(page["name"], href=page["path"])
                    for page in dash.page_registry.values()
                    if page["module"] != "pages.not_found_404"
                ]
            ),
            id="offcanvas",
            is_open=False,
        ),
    ],
    className="my-3",
)

app.layout = dbc.Container(
    [offcanvas, dl.plugins.page_container],
    fluid=True,
)


@app.callback(
    Output("offcanvas", "is_open"),
    Input("open-offcanvas", "n_clicks"),
    [State("offcanvas", "is_open")],
)
def toggle_offcanvas(n1, is_open):
    if n1:
        return not is_open
    return is_open


if __name__ == "__main__":
    app.run_server(debug=True)
