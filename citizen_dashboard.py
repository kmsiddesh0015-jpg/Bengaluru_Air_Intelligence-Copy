
import html

from citizen_insights import (
    build_citizen_insights_with_pollutant_chart,
    generate_air_quality_insights
)
from citizen_hotspot_map import build_citizen_hotspot_map


def build_complete_citizen_dashboard(project_dir):
    summary, report, aqi_chart, pollutant_chart = (
        build_citizen_insights_with_pollutant_chart(project_dir)
    )

    hotspot_map = build_citizen_hotspot_map(project_dir)
    map_document = hotspot_map.get_root().render()

    safe_document = html.escape(map_document, quote=True)

    map_html = f"""
    <iframe
        srcdoc="{safe_document}"
        style="width:100%; height:620px; border:0; border-radius:8px;"
        sandbox="allow-scripts allow-same-origin allow-popups"
        loading="lazy">
    </iframe>
    """

    automated_insights = generate_air_quality_insights(project_dir)

    return (
        summary,
        report,
        aqi_chart,
        map_html,
        pollutant_chart,
        automated_insights
    )
