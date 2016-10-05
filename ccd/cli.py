#!/usr/bin/env python3
""" Command line interface to Python Continuous Change Detection. """

from ccd import app
from click_plugins import with_plugins
from pkg_resources import iter_entry_points
import click

logger = app.logging.getLogger(__name__)


@with_plugins(iter_entry_points('core_package.cli_plugins'))
@click.group()
def cli():
    """Commandline interface for yourpackage."""
    logger.info("CLI running...")


@cli.command()
def subcommand():
    """Subcommand that does something."""
    logger.info("Subcommand running...")


@cli.command()
def another_subcommand():
    """Another Subcommand that does something."""
    logger.info("Another Subcommand running...")

if __name__ == '__main__':
    cli()
