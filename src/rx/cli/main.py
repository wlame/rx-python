"""Main CLI entry point with command groups"""

import multiprocessing

import click

from rx.__version__ import __version__
from rx.cli.check import check_command
from rx.cli.compress import compress_command
from rx.cli.index import index_command
from rx.cli.samples import samples_command
from rx.cli.serve import serve_command
from rx.cli.trace import trace_command


class DefaultCommandGroup(click.Group):
    """Custom Click Group that allows a default command"""

    def parse_args(self, ctx, args):
        # During shell completion, don't redirect to default command
        # This allows Click to suggest subcommands properly
        if ctx.resilient_parsing:
            return super().parse_args(ctx, args)

        # If --help or --version is requested, show group help/version
        if args and args[0] in ('--help', '-h', '--version'):
            return super().parse_args(ctx, args)

        # Check if first arg is a known command
        if args and args[0] in self.commands:
            return super().parse_args(ctx, args)

        # Otherwise, treat as trace command (default)
        # Insert 'trace' as the command name
        return super().parse_args(ctx, ['trace'] + args)


@click.group(cls=DefaultCommandGroup, invoke_without_command=True)
@click.version_option(version=__version__, prog_name='RX')
@click.pass_context
def cli(ctx):
    """
    RX (Regex Tracer) - High-performance file tracing and analysis tool.

    \b
    Commands:
      rx <pattern> [path ...]   Trace files for patterns (default command)
      rx index <path>           Create/manage file indexes (use --analyze for full analysis)
      rx check <pattern>        Analyze regex complexity
      rx compress <path>        Create seekable zstd for optimized access
      rx samples <path>         Get file content around byte offsets
      rx serve                  Start web API server

    \b
    Examples:
      rx "error" /var/log/app.log
      rx "error.*failed" /var/log/ -i
      rx index /var/log/app.log --analyze
      rx check "(a+)+"
      rx serve --port 8000

    \b
    For more help on each command:
      rx --help
      rx index --help
      rx check --help
      rx serve --help
    """
    if ctx.invoked_subcommand is None:
        click.echo(ctx.get_help())
        ctx.exit(0)


# Register subcommands (trace is the default command)
cli.add_command(trace_command, name='trace')
cli.add_command(check_command, name='check')
cli.add_command(compress_command, name='compress')
cli.add_command(index_command, name='index')
cli.add_command(samples_command, name='samples')
cli.add_command(serve_command, name='serve')


def main():
    """Entry point for the CLI"""
    # Support for multiprocessing in frozen binaries (PyInstaller)
    # https://docs.python.org/3/library/multiprocessing.html#multiprocessing.freeze_support
    multiprocessing.freeze_support()

    cli()


if __name__ == '__main__':
    main()
