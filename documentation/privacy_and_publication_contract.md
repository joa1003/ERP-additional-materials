# Privacy and publication contract

The distributable repository must not contain contributor-specific account names, home-directory paths, cluster-login identities, private development history, credentials, tokens or machine-specific absolute filesystem paths.

## Runtime paths

Absolute filesystem paths may be used internally while code is executing, because canonical absolute paths reduce working-directory ambiguity. They must not be persisted into public forecasting provenance metadata.

Formal forecasting therefore records repository-relative paths for:

- canonical configuration files;
- predictions and metric outputs;
- checkpoints and learning-curve files;
- source-checkpoint references.

The Python executable is recorded by executable filename only, not by its account-specific installation path. Failed-run diagnostics redact the repository root and user home prefix before status metadata are written.

## Generated outputs

The public repository does not distribute `outputs/`, raw provider data or processed intermediate data. If generated outputs are ever added to a public release, `validation/validate_public_privacy.py` must pass on the exact release tree before publication.

The privacy validator checks public text and included runtime outputs for common user-home, institutional mounted-user, network-scratch, Windows user-profile and cluster-login patterns. It also scans included model/checkpoint binaries for path prefixes that can be embedded in serialised metadata.

## Credentials

No password, API key, SSH private key, access token or other credential is required by the reproducibility code and none should be stored in the repository. Dataset access follows the provider instructions in `data_access/README_data_access.md`.
