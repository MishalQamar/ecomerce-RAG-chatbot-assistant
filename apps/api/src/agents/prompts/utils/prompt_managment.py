import yaml
from jinja2 import Template


def prompt_template_config(yaml_file, prompt_key):
    with open(yaml_file, "r") as file:
        config = yaml.safe_load(file)
    prompt_template = config["prompts"][prompt_key]
    template = Template(prompt_template)
    return template
