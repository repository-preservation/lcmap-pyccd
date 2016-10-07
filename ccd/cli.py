#!/usr/bin/env python3
""" Command line interface to Python Continuous Change Detection. """

from ccd import app
import ccd
from click_plugins import with_plugins
from pkg_resources import iter_entry_points
import click
import numpy as np

logger = app.logging.getLogger(__name__)


@with_plugins(iter_entry_points('core_package.cli_plugins'))
@click.group()
def cli():
    """Commandline interface for yourpackage."""
    logger.info("CLI running...")


@cli.command()
@click.argument('path')
def sample(path):
    """Subcommand for processing sample data."""

    logger.info("Loading data...")
    samples = np.genfromtxt(path, delimiter=',', dtype=np.int).T

    logger.info("Building change model...")
    results = ccd.detect(*samples)

    logger.info("Done...")
    click.echo(results)


@cli.command()
def another_subcommand():
    """Another Subcommand that does something."""
    logger.info("Another Subcommand running...")

if __name__ == '__main__':
    cli()
