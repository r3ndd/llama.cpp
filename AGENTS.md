## Frequently referenced docs

- [CONTRIBUTING.md](CONTRIBUTING.md)
- [Build docs](docs/build.md)
- [Server usage](tools/server/README.md)
- [Server dev scope](tools/server/README-dev.md)
- [PEG parser](docs/development/parsing.md)
- [Autoparser](docs/autoparser.md)
- [Jinja engine](common/jinja/README.md)
- [How to add a new model](docs/development/HOWTO-add-model.md)
- [PR template](.github/pull_request_template.md)

## Project-wide learnings

- `-hf/--hf-repo` downloads require TLS-enabled builds (OpenSSL/BoringSSL/LibreSSL). Without TLS support, `llama-cli` fails before inference when fetching from Hugging Face.
