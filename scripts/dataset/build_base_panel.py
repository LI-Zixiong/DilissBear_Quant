"""
Build the unified daily base panel from raw Tushare + CSMAR + disclosure data.

This is the one-time historical build entry point.
"""
from src.pipeline.base_panel import BasePanelConfig, build_base_panel

if __name__ == "__main__":
    config = BasePanelConfig()
    build_base_panel(config)
