#!/usr/bin/env python3
""" Command line interface to Python Continuous Change Detection.

Currently unsupported through beta.

However, the click conventions used to import a .csv test dataset could be
replaced with other conventions (API, wget, etc) to formulate a an
observations data set for invoking pyccd with a similar client call to:
results = ccd.detect() .
"""

from ccd import app
import ccd
from click_plugins import with_plugins
from pkg_resources import iter_entry_points
import click
import json
import numpy as np
import time
import timeit


logger = app.logging.getLogger(__name__)


@with_plugins(iter_entry_points('core_package.cli_plugins'))
@click.group()
def cli():
    """Commandline interface for your package."""
    logger.info("CLI running...")


@cli.command()
@click.argument('path')
@click.option('--format', default='json', type=click.Choice(['json', 'table']))
def sample(path, format):
    """Subcommand for processing sample data."""

    logger.debug("Loading data...")
    samples = np.genfromtxt(path, delimiter=',', dtype=np.int).T

    logger.debug("Building change model...")

    start_time = timeit.default_timer()
    results = ccd.detect(*samples)
    print("ElapsedTime: ", round((timeit.default_timer() - start_time), 3))

    if format == 'table':
        click.echo(results_to_table(results))
    else:
        click.echo(json.dumps(results, indent=2))

    logger.debug("Done...")


@cli.command()
def another_subcommand():
    """Another Subcommand that does something."""
    logger.info("Another Subcommand running...")


def results_to_table(results):
    """Output change detection results into text table"""
    change_format = "Time Segment {0}: {1}...{2}"
    band_format = "{0:10} {1:10.4f} {2:10.4f} {3:10.4} {4:10.4f} {5:10.4f} {6:>10.4f} {7:>15.2f}"
    columns = ["band", "mags", 'rmse', 'c1', 'c2', 'c3', 'c4', 'intercept']
    for ix, segment in enumerate(results):

        # describe the segment period and inputs
        click.echo(change_format.format(ix, segment['start_day'], segment['end_day']))

        click.echo("{0:<10} {1:>10} {2:>10} {3:>10} {4:>10} {5:>10} {6:>10} {7:>15}".format(*columns))
        for color in ['red', 'green', 'blue', 'nir', 'swir1', 'swir2']:
            click.echo(band_format.format(*[color,
                                          segment[color]['magnitude'],
                                          segment[color]['rmse'],
                                          segment[color]['coefficients'][0],
                                          segment[color]['coefficients'][1],
                                          segment[color]['coefficients'][2],
                                          segment[color]['coefficients'][3],
                                          segment[color]['intercept']]))

if __name__ == '__main__':
    cli()
