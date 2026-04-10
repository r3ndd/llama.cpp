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

- `llama-cli -hf/--hf-repo` requires a TLS-enabled build (OpenSSL/BoringSSL/LibreSSL); without TLS, HF downloads fail before inference starts.
