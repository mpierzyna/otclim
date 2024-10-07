"""
Step 1a: Download ERA5 data for domain of interest

Define all settings for this script in `config.py`.
"""

import logging

import config
import otclim

logging.basicConfig(level=logging.INFO)
logger = otclim.logging.get_logger("data")

if __name__ == "__main__":
    logger.info(f"Downloading {len(config.ERA5_VARS)} variables: {config.ERA5_VARS}")
    logger.info("Note that this may take a while...")

    # Always queue 10 requests at the same time.
    otclim.era5.get_sl_parallel(
        variables=config.ERA5_VARS,
        area=config.ERA5_BOX,
        years=config.ERA5_YEARS,
        months=config.ERA5_MONTHS,
        days=config.ERA5_DAYS,
        root=config.DATA_RAW_ERA5,
    )
