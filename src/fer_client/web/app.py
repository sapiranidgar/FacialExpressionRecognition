"""Streamlit website for uploading a face image and seeing its predicted expression.

The page never touches a model: each upload is forwarded to fer_server
through FerClient, and the returned prediction is rendered.

Run (fer_server must already be running) from the repo root:
    PYTHONPATH=src streamlit run src/fer_client/web/app.py
"""

from __future__ import annotations

import altair as alt
import pandas as pd
import streamlit as st

from fer_client.client import FerClient, FerClientError, Prediction
from fer_client.constants import (
    CHART_COLUMN_EMOTION,
    CHART_COLUMN_PROBABILITY,
    CHART_TITLE,
    PAGE_ICON,
    PAGE_TITLE,
    PREVIEW_CAPTION,
    RESULT_CONFIDENCE_TEMPLATE,
    RESULT_METRIC_LABEL,
    SPINNER_TEXT,
    UPLOAD_ACCEPTED_TYPES,
    UPLOAD_HELP,
    UPLOAD_LABEL,
)


@st.cache_resource
def get_client() -> FerClient:
    """Create one FerClient shared across reruns and sessions."""
    return FerClient()


def probability_chart(prediction: Prediction) -> alt.Chart:
    """Build a horizontal bar chart of every class's probability, highest first.

    Args:
        prediction: The server's prediction.

    Returns:
        An Altair bar chart.
    """
    df = pd.DataFrame(
        {
            CHART_COLUMN_EMOTION: list(prediction.probabilities),
            CHART_COLUMN_PROBABILITY: list(prediction.probabilities.values()),
        }
    )
    return (
        alt.Chart(df, title=CHART_TITLE)
        .mark_bar()
        .encode(
            x=alt.X(CHART_COLUMN_PROBABILITY, scale=alt.Scale(domain=[0, 1]), axis=alt.Axis(format="%")),
            y=alt.Y(CHART_COLUMN_EMOTION, sort="-x", title=None),
            tooltip=[CHART_COLUMN_EMOTION, alt.Tooltip(CHART_COLUMN_PROBABILITY, format=".1%")],
        )
    )


def render_prediction(prediction: Prediction) -> None:
    """Show the top expression with its confidence, then the full probability chart.

    Args:
        prediction: The server's prediction.
    """
    st.metric(
        RESULT_METRIC_LABEL,
        prediction.emotion.capitalize(),
        RESULT_CONFIDENCE_TEMPLATE.format(confidence=prediction.confidence),
        delta_color="off",
    )
    st.altair_chart(probability_chart(prediction), use_container_width=True)


def main() -> None:
    """Render the page: upload widget, then preview + prediction once an image is chosen."""
    st.set_page_config(page_title=PAGE_TITLE, page_icon=PAGE_ICON)
    st.title(PAGE_TITLE)

    uploaded = st.file_uploader(UPLOAD_LABEL, type=UPLOAD_ACCEPTED_TYPES, help=UPLOAD_HELP)
    st.caption(UPLOAD_HELP)
    if uploaded is None:
        return

    image_bytes = uploaded.getvalue()
    preview_col, result_col = st.columns([1, 2])
    preview_col.image(image_bytes, caption=PREVIEW_CAPTION, use_container_width=True)

    with result_col:
        try:
            with st.spinner(SPINNER_TEXT):
                prediction = get_client().predict(image_bytes)
        except FerClientError as e:
            st.error(str(e))
            return
        render_prediction(prediction)


main()
