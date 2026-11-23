# Shared by update.sh and webui-guard.sh. Sourced, never executed: it only
# defines things, so both scripts check the chat UI settings the same way.
# shellcheck shell=bash

# WEBUI_MODEL goes into OPENAI_API_CONFIGS as JSON in the Docker compose
# files. A quote, comma or bracket would break that JSON or add entries, so
# only plain id characters are allowed.
QCM_MODEL_ID_RE='^[A-Za-z0-9._:/-]+$'

# webui_model_ok <id>: true for an empty id (no fixed model) or a plain one.
webui_model_ok() {
  [[ -z "${1:-}" || "$1" =~ $QCM_MODEL_ID_RE ]]
}

# webui_auth_on: true when WEBUI_AUTH (from .env) turns the login on.
webui_auth_on() {
  [[ "$(printf '%s' "${WEBUI_AUTH:-false}" | tr '[:upper:]' '[:lower:]')" == "true" ]]
}
