"""Research-only age validation and biological-program views for the atlas."""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import plotly.express as px
import streamlit as st


def render_age_research(root: Path) -> None:
    st.subheader("Donor-level age prediction · research evidence")
    st.caption(
        "T05 frozen scGPT representation, 18 independent donors, nested leave-one-donor-out "
        "validation. This is a separate analysis from the site's earlier Elastic Net model."
    )
    bundle = root / "release" / "age_prediction_v1"
    needed = [bundle / name for name in (
        "summary.json", "donor_age_predictions.tsv", "permutation_summary.tsv"
    )]
    if not all(path.is_file() for path in needed):
        st.info(
            "The checked T05 age-result bundle has not been imported into this site. "
            "The existing Donor evidence page still shows the earlier Elastic Net analysis. "
            "Run the repository's export_t05_site_bundle.py on the analysis server, "
            "then synchronize release/age_prediction_v1."
        )
        return

    try:
        summary = json.loads(needed[0].read_text(encoding="utf-8"))
        donors = pd.read_csv(needed[1], sep="\t")
        permutation = pd.read_csv(needed[2], sep="\t")
    except (ValueError, OSError, pd.errors.ParserError) as exc:
        st.error(f"The T05 release bundle could not be read: {exc}")
        return

    required = {"donor_id", "chronological_age", "predicted_age", "corrected_gap_years"}
    if summary.get("status") != "research_only" or not required.issubset(donors):
        st.error("The T05 release bundle failed its website schema check.")
        return

    st.warning(
        "Research prototype: 18 training donors and no independent multi-donor validation "
        "for this T05 scGPT age model. A donor's corrected residual is not a health diagnosis "
        "or a clinically validated immune age."
    )
    primary = summary["primary_model"]
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Independent donors", str(len(donors)))
    c2.metric("Nested LODO MAE", f"{primary['mae_years']:.2f} years")
    c3.metric("Nested LODO R²", f"{primary['cv_r2']:.3f}")
    c4.metric("Mean-age baseline MAE", f"{primary['baseline_mae_years']:.2f} years")

    shown = donors.copy()
    shown["age_range_extrapolation"] = shown["age_range_extrapolation"].map(
        {True: "Yes", False: "No"}
    )
    figure = px.scatter(
        shown,
        x="chronological_age",
        y="predicted_age",
        color="age_range_extrapolation",
        hover_data=["donor_id", "absolute_error_years"],
        labels={
            "chronological_age": "Chronological age (years)",
            "predicted_age": "Held-out predicted age (years)",
            "age_range_extrapolation": "Outside training age range",
        },
        color_discrete_map={"No": "#0b756d", "Yes": "#c95858"},
    )
    low = float(min(shown["chronological_age"].min(), shown["predicted_age"].min()))
    high = float(max(shown["chronological_age"].max(), shown["predicted_age"].max()))
    figure.add_shape(type="line", x0=low, y0=low, x1=high, y1=high,
                     line={"color": "#666", "dash": "dash"})
    figure.update_layout(height=450, legend_title_text="")
    st.plotly_chart(figure, use_container_width=True)

    st.markdown("#### Full-pipeline permutation tests")
    st.dataframe(
        permutation[["scheme", "n_permutations", "permutation_p_mae_improvement",
                     "permutation_p_cv_r2"]],
        hide_index=True, use_container_width=True,
    )
    st.caption(
        "Each permutation repeats donor-level outer validation and inner Ridge parameter "
        "selection. Within-sex permutation retains the observed sex groups."
    )

    with st.expander("Inspect the 18 held-out donor predictions"):
        st.dataframe(
            shown[["donor_id", "chronological_age", "predicted_age",
                   "absolute_error_years", "corrected_gap_years",
                   "age_range_extrapolation"]],
            hide_index=True, use_container_width=True,
        )
        st.caption(
            "Corrected gap subtracts an age-dependent bias fitted on other donors. "
            "Its direction is not stable across all representation choices."
        )
    st.download_button(
        "Download checked T05 donor results",
        needed[1].read_bytes(),
        file_name="t05_donor_age_predictions.tsv",
        mime="text/tab-separated-values",
    )


