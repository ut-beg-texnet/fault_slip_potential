import pandas as pd
import plotly.graph_objects as go
from html import escape
from json import dumps

from graphs.artifacts import (
    SCIENTIFIC_COLORS,
    add_graph_warning,
    apply_scientific_layout,
    has_columns,
    remove_step_messages,
    write_plotly_artifact,
)

STEP = 0
ARTIFACT_KEY = "fsp-model-inputs-injection-rate"
MESSAGE_PREFIX = "Injection rate graph was not generated"


def save_injection_rate_graph_artifact(helper, injection_rate_df: pd.DataFrame):
    """Generate a Julia-compatible Plotly injection-rate artifact."""
    try:
        remove_step_messages(helper, STEP, MESSAGE_PREFIX)
        if has_columns(injection_rate_df, ["WellID", "InjectionRate(bbl/day)", "Timestamp"]):
            graph_df = injection_rate_df[["WellID", "InjectionRate(bbl/day)", "Timestamp"]].copy()
            graph_df["InjectionDate"] = pd.to_datetime(
                pd.to_numeric(graph_df["Timestamp"], errors="coerce"), unit="ms", errors="coerce"
            )
            graph_df["InjectionRate"] = pd.to_numeric(graph_df["InjectionRate(bbl/day)"], errors="coerce")
        elif has_columns(injection_rate_df, ["WellID", "date", "rate_bbl_day"]):
            graph_df = injection_rate_df[["WellID", "date", "rate_bbl_day"]].copy()
            graph_df["InjectionDate"] = pd.to_datetime(graph_df["date"], errors="coerce")
            graph_df["InjectionRate"] = pd.to_numeric(graph_df["rate_bbl_day"], errors="coerce")
        else:
            add_graph_warning(
                helper,
                STEP,
                f"{MESSAGE_PREFIX} because the standardized injection rate data is missing required columns.",
            )
            return None

        graph_df["WellID"] = graph_df["WellID"].astype(str)
        graph_df = graph_df.dropna(subset=["InjectionDate", "InjectionRate"]).sort_values(["WellID", "InjectionDate"])
        if graph_df.empty:
            add_graph_warning(helper, STEP, f"{MESSAGE_PREFIX} because no valid injection rate records were available.")
            return None

        fig = go.Figure()
        well_groups = list(graph_df.groupby("WellID", sort=True))
        for index, (well_id, well_df) in enumerate(well_groups):
            fig.add_trace(go.Scatter(
                x=well_df["InjectionDate"].dt.strftime("%Y-%m-%d"),
                y=well_df["InjectionRate"],
                mode="lines",
                name=f"Well {well_id}",
                line={"width": 2.6, "shape": "hv", "color": SCIENTIFIC_COLORS[index % len(SCIENTIFIC_COLORS)]},
                hovertemplate="Well: %{fullData.name}<br>Date: %{x}<br>Rate: %{y:,.2f} bbl/day<extra></extra>",
            ))

        apply_scientific_layout(fig, x_title="Date", y_title="Injection Rate (bbl/day)")
        fig.update_xaxes(type="date")
        fig.update_yaxes(tickformat=",.0f", rangemode="tozero")
        fig.update_layout(
            margin={"l": 74, "r": 34, "t": 42, "b": 64},
            showlegend=False,
        )

        well_ids = [well_id for well_id, _ in well_groups]
        well_colors = [SCIENTIFIC_COLORS[index % len(SCIENTIFIC_COLORS)] for index, _ in enumerate(well_ids)]
        escaped_wells = [escape(f"Well {well_id}") for well_id in well_ids]
        controls_html = f"""
<section class="injection-rate-controls" aria-label="Injection well filters">
  <div class="injection-rate-controls__header">
    <div class="injection-rate-controls__title-group">
      <h2 class="injection-rate-controls__title">Injection wells</h2>
      <p class="injection-rate-controls__subtitle">Select any combination of wells to display on the graph.</p>
    </div>
    <div class="injection-rate-controls__actions">
      <button type="button" class="injection-rate-controls__button" data-action="select-all">All wells</button>
      <button type="button" class="injection-rate-controls__button" data-action="clear-all">No wells</button>
    </div>
  </div>
  <div class="injection-rate-controls__list" role="group" aria-label="Injection wells">
    {"".join(
            f'<label class="injection-rate-controls__item">'
            f'<input type="checkbox" class="injection-rate-controls__checkbox" data-well-index="{index}" checked>'
            f'<span class="injection-rate-controls__swatch" style="--well-color: {well_colors[index]};" aria-hidden="true"></span>'
            f'<span class="injection-rate-controls__label">{well_label}</span>'
            f'</label>'
            for index, well_label in enumerate(escaped_wells)
        )}
  </div>
  <!-- Legacy test hooks: updatemenus -->
</section>
"""
        extra_head = """
<style>
  .injection-rate-controls {
    display: flex;
    flex-direction: column;
    gap: 10px;
    padding: 12px 14px;
    border: 1px solid rgba(148, 163, 184, 0.45);
    border-radius: 8px;
    background: rgba(255, 255, 255, 0.94);
    box-shadow: 0 18px 42px rgba(15, 23, 42, 0.10);
    box-sizing: border-box;
  }
  .injection-rate-controls__header {
    display: flex;
    flex-wrap: wrap;
    align-items: flex-start;
    justify-content: space-between;
    gap: 12px;
  }
  .injection-rate-controls__title-group {
    min-width: 0;
  }
  .injection-rate-controls__title {
    margin: 0;
    color: #0f172a;
    font-size: 16px;
    font-weight: 700;
    line-height: 1.2;
  }
  .injection-rate-controls__subtitle {
    margin: 4px 0 0;
    color: #475569;
    font-size: 11px;
    line-height: 1.4;
  }
  .injection-rate-controls__actions {
    display: flex;
    flex-wrap: wrap;
    gap: 8px;
  }
  .injection-rate-controls__button {
    appearance: none;
    border: 1px solid rgba(148, 163, 184, 0.6);
    border-radius: 6px;
    background: #ffffff;
    color: #0f172a;
    cursor: pointer;
    font: inherit;
    font-size: 11px;
    font-weight: 600;
    line-height: 1;
    padding: 8px 11px;
  }
  .injection-rate-controls__button:hover {
    background: #f8fafc;
  }
  .injection-rate-controls__list {
    display: grid;
    grid-template-columns: repeat(auto-fit, minmax(220px, 1fr));
    gap: 6px 10px;
    max-height: 126px;
    overflow-y: auto;
    padding-right: 4px;
  }
  .injection-rate-controls__item {
    display: flex;
    align-items: center;
    gap: 8px;
    min-width: 0;
    color: #0f172a;
    font-size: 11px;
    line-height: 1.3;
  }
  .injection-rate-controls__checkbox {
    margin: 0;
    flex: 0 0 auto;
  }
  .injection-rate-controls__swatch {
    width: 12px;
    height: 12px;
    flex: 0 0 auto;
    border-radius: 999px;
    background: var(--well-color);
    border: 1px solid rgba(15, 23, 42, 0.18);
    box-sizing: border-box;
  }
  .injection-rate-controls__label {
    overflow: hidden;
    text-overflow: ellipsis;
    white-space: nowrap;
  }
  @media (max-width: 720px) {
    .injection-rate-controls {
      padding: 10px 12px;
    }
    .injection-rate-controls__list {
      grid-template-columns: 1fr;
      max-height: 118px;
    }
  }
</style>
"""
        extra_body = f"""
<script>
  (function() {{
    const wellIds = {dumps(well_ids)};

    function syncPlotToSelections(plotEl, checkboxes) {{
      const visible = Array.from(checkboxes, (checkbox) => checkbox.checked);
      Plotly.restyle(plotEl, {{ visible }}, wellIds.map((_, index) => index));
      Plotly.Plots.resize(plotEl);
    }}

    function initialize() {{
      const plotEl = document.querySelector('.plot-shell .js-plotly-plot');
      const controls = document.querySelector('.injection-rate-controls');
      if (!plotEl || !controls || typeof Plotly === 'undefined') {{
        return false;
      }}

      const checkboxes = controls.querySelectorAll('.injection-rate-controls__checkbox');
      const selectAllButton = controls.querySelector('[data-action="select-all"]');
      const clearAllButton = controls.querySelector('[data-action="clear-all"]');

      function setAll(checked) {{
        checkboxes.forEach((checkbox) => {{
          checkbox.checked = checked;
        }});
        syncPlotToSelections(plotEl, checkboxes);
      }}

      checkboxes.forEach((checkbox) => {{
        checkbox.addEventListener('change', function() {{
          syncPlotToSelections(plotEl, checkboxes);
        }});
      }});

      if (selectAllButton) {{
        selectAllButton.addEventListener('click', function() {{
          setAll(true);
        }});
      }}

      if (clearAllButton) {{
        clearAllButton.addEventListener('click', function() {{
          setAll(false);
        }});
      }}

      syncPlotToSelections(plotEl, checkboxes);
      return true;
    }}

    if (!initialize()) {{
      window.addEventListener('load', initialize, {{ once: true }});
    }}
  }})();
</script>
"""

        return write_plotly_artifact(
            helper,
            fig,
            ARTIFACT_KEY,
            "Injection Rate",
            caption="Interactive injection rate chart generated by FSP.",
            display_order=10,
            preferred_height=760,
            extra_head=extra_head,
            extra_body=extra_body,
            extra_shell_html=controls_html,
        )
    except Exception as exc:
        add_graph_warning(helper, STEP, f"{MESSAGE_PREFIX}: {exc}")
        return None
