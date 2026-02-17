import click

@click.command()
@click.option('--config', type=str, help='Path to the config file')
def run_mnemosyne(config):
    pass

