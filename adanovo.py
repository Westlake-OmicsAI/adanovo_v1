import datetime
import functools
import logging
import os
import re
import shutil
import sys
import warnings
from typing import Optional, Tuple

warnings.filterwarnings("ignore", category=DeprecationWarning)

import appdirs
import click
import github
import requests
import torch
import tqdm
import yaml
from pytorch_lightning.lite import LightningLite

import utils
from data import ms_io
from denovo import model_runner
from config import Config

logger = logging.getLogger("adanovo")


@click.command()
@click.option(
    "--mode",
    required=True,
    default="train",
    help="The mode in which to run Adanovo:"
    '- "denovo" will predict peptide sequences for unknown MS/MS spectra.\n'
    '- "train" will train a model (from scratch or by continuing training a '
    "previously trained model)."
    '- "eval" will evaluate the performance of a trained model using '
    "previously acquired spectrum annotations.",
    type=click.Choice(["denovo", "train", "eval"]),
)
@click.option(
    "--model",
    help="The file name of the model weights (.ckpt file).",
    type=click.Path(exists=True, dir_okay=False),
)
@click.option(
    "--peak_path",
    required=True,
    help="The file path with peak files for predicting peptide sequences or "
    "training Adanovo.",
)
@click.option(
    "--peak_path_val",
    help="The file path with peak files to be used as validation data during "
    "training.",
)
@click.option(
    "--config",
    help="The file name of the configuration file with custom options",
    type=click.Path(exists=True, dir_okay=False),
)
@click.option(
    "--output",
    help="The base output file name to store logging (extension: .log) and "
    "(optionally) prediction results (extension: .mztab).",
    type=click.Path(dir_okay=False),
)
@click.option(
    "--s1",
    default="0.3",
    help="the standard deviation for W_aa before softmax",
)
@click.option(
    "--s2",
    default="0.1",
    help="the standard deviation for W_psm before softmax",
)
def main(
    mode: str,
    model: Optional[str],
    peak_path: str,
    peak_path_val: Optional[str],
    config: Optional[str],
    output: Optional[str],
    s1: str,
    s2: str,
):
    if output is None:
        output = os.path.join(
            os.getcwd(),
            f"Adanovo_{datetime.datetime.now().strftime('%Y%m%d%H%M%S')}",
        )
    else:
        basename, ext = os.path.splitext(os.path.abspath(output))
        output = basename if ext.lower() in (".log", ".mztab") else output

    # Configure logging.
    logging.captureWarnings(True)
    root = logging.getLogger()
    root.setLevel(logging.DEBUG)
    log_formatter = logging.Formatter(
        "{asctime} {levelname} [{name}/{processName}] {module}.{funcName} : "
        "{message}",
        style="{",
    )
    console_handler = logging.StreamHandler(sys.stderr)
    console_handler.setLevel(logging.DEBUG)
    console_handler.setFormatter(log_formatter)
    root.addHandler(console_handler)
    file_handler = logging.FileHandler(f"{output}.log")
    file_handler.setFormatter(log_formatter)
    root.addHandler(file_handler)
    
    # Disable dependency non-critical log messages.
    logging.getLogger("depthcharge").setLevel(logging.INFO)
    logging.getLogger("github").setLevel(logging.WARNING)
    logging.getLogger("h5py").setLevel(logging.WARNING)
    logging.getLogger("numba").setLevel(logging.WARNING)
    logging.getLogger("pytorch_lightning").setLevel(logging.WARNING)
    logging.getLogger("torch").setLevel(logging.WARNING)
    logging.getLogger("urllib3").setLevel(logging.WARNING)

    # Read parameters from the config file.
    config = Config(config)

    LightningLite.seed_everything(seed=config["random_seed"], workers=True)

    # Download model weights if these were not specified (except when training).
    if model is None and mode != "train":
        raise FileNotFoundError(f"No model input in {mode} mode!")

    # Log the active configuration.
    logger.debug("mode = %s", mode)
    logger.debug("model = %s", model)
    logger.debug("peak_path = %s", peak_path)
    logger.debug("peak_path_val = %s", peak_path_val)
    logger.debug("config = %s", config.file)
    logger.debug("output = %s", output)
    for key, value in config.items():
        logger.debug("%s = %s", str(key), str(value))

    logger.debug("s1 = %s", s1)
    logger.debug("s2 = %s", s2)
    
    # Run Adanovo in the specified mode.
    if mode == "denovo":
        logger.info("Predict peptide sequences with Adanovo.")
        writer = ms_io.MztabWriter(f"{output}.mztab")
        writer.set_metadata(config, model=model, config_filename=config.file)
        model_runner.predict(peak_path, model, config, writer)
        writer.save()
    elif mode == "eval":
        logger.info("Evaluate a trained Adanovo model.")
        model_runner.evaluate(peak_path, model, config)
    elif mode == "train":
        logger.info("Train the Adanovo model.")
        model_runner.train(peak_path, peak_path_val, model, config, s1, s2)


if __name__ == "__main__":
    main()
