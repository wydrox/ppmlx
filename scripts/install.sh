#!/usr/bin/env bash
# ppmlx installer — CLI for Apple Silicon LLMs via MLX
# Usage: curl -fsSL https://raw.githubusercontent.com/wydrox/ppmlx/main/scripts/install.sh | sh
set -euo pipefail

# ── colours ──────────────────────────────────────────────────────────────────
RED='\033[0;31m'; YELLOW='\033[1;33m'; GREEN='\033[0;32m'
CYAN='\033[0;36m'; BOLD='\033[1m'; RESET='\033[0m'

info()    { printf "${CYAN}[ppmlx]${RESET} %s\n" "$*"; }
success() { printf "${GREEN}[ppmlx]${RESET} %s\n" "$*"; }
warn()    { printf "${YELLOW}[ppmlx]${RESET} %s\n" "$*"; }
error()   { printf "${RED}[ppmlx]${RESET} %s\n" "$*" >&2; exit 1; }

# ── platform checks ──────────────────────────────────────────────────────────
info "Checking platform..."

if [[ "$(uname -s)" != "Darwin" ]]; then
    error "ppmlx requires macOS. Detected: $(uname -s)"
fi

if [[ "$(uname -m)" != "arm64" ]]; then
    error "ppmlx requires Apple Silicon (arm64). Detected: $(uname -m)"
fi

MACOS_VERSION=$(sw_vers -productVersion)
info "macOS ${MACOS_VERSION} on Apple Silicon detected."

# Check Python 3.11+ is available (not strictly required — uv handles it)
PYTHON_OK=false
for py in python3.11 python3.12 python3; do
    if command -v "$py" &>/dev/null; then
        VER=$("$py" -c "import sys; print(f'{sys.version_info.major}.{sys.version_info.minor}')" 2>/dev/null || echo "0.0")
        MAJOR=$(echo "$VER" | cut -d. -f1)
        MINOR=$(echo "$VER" | cut -d. -f2)
        if [[ "$MAJOR" -eq 3 && "$MINOR" -ge 11 && "$MINOR" -le 12 ]]; then
            PYTHON_OK=true
            PYTHON_BIN="$py"
            break
        fi
    fi
done

# ── install uv ────────────────────────────────────────────────────────────────
install_via_uv() {
    if ! command -v uv &>/dev/null; then
        info "uv not found. Installing uv..."
        curl -LsSf https://astral.sh/uv/install.sh | sh
        # Add uv to PATH for this session
        export PATH="$HOME/.cargo/bin:$HOME/.local/bin:$PATH"
        if ! command -v uv &>/dev/null; then
            warn "uv installed but not found in PATH. Trying pipx fallback..."
            return 1
        fi
        success "uv installed."
    else
        info "uv already installed: $(uv --version)"
    fi

    info "Installing ppmlx via uv tool install..."
    uv tool install ppmlx --python 3.11

    # Ensure uv tool bin dir is on PATH
    UV_BIN_DIR="$(uv tool dir)/bin"
    if [[ ":$PATH:" != *":$UV_BIN_DIR:"* ]]; then
        warn "Add to your shell profile: export PATH=\"${UV_BIN_DIR}:\$PATH\""
    fi

    return 0
}

# ── fallback: pipx ────────────────────────────────────────────────────────────
install_via_pipx() {
    if ! command -v pipx &>/dev/null; then
        info "pipx not found. Installing pipx..."
        if command -v brew &>/dev/null; then
            brew install pipx
            pipx ensurepath
        elif $PYTHON_OK; then
            "$PYTHON_BIN" -m pip install --user pipx
            "$PYTHON_BIN" -m pipx ensurepath
        else
            error "Cannot install pipx — no suitable Python 3.11/3.12 or Homebrew found."
        fi
    fi

    info "Installing ppmlx via pipx..."
    pipx install ppmlx
}

# ── main install flow ─────────────────────────────────────────────────────────
if install_via_uv; then
    INSTALL_METHOD="uv"
else
    install_via_pipx
    INSTALL_METHOD="pipx"
fi

# ── verify installation ───────────────────────────────────────────────────────
info "Verifying installation..."
if command -v ppmlx &>/dev/null; then
    VERSION=$(ppmlx --version 2>/dev/null || echo "unknown")
    success "ppmlx ${VERSION} installed successfully via ${INSTALL_METHOD}!"
else
    warn "ppmlx installed but not found in current PATH."
    warn "Please restart your terminal or run: source ~/.zshrc"
fi

# ── quick-start ───────────────────────────────────────────────────────────────
printf "\n${BOLD}Quick Start:${RESET}\n"
printf "  ${CYAN}ppmlx pull llama3${RESET}          # download Llama 3 8B\n"
printf "  ${CYAN}ppmlx run llama3${RESET}           # interactive chat REPL\n"
printf "  ${CYAN}ppmlx serve${RESET}                # start OpenAI-compatible server on :6767\n"
printf "  ${CYAN}ppmlx list${RESET}                 # list downloaded models\n"
printf "  ${CYAN}ppmlx aliases${RESET}              # show all model aliases\n"
printf "  ${CYAN}ppmlx --help${RESET}               # full command reference\n"
printf "\n${BOLD}Server usage:${RESET}\n"
printf "  ${CYAN}curl http://localhost:6767/v1/models${RESET}\n"
printf "  ${CYAN}curl http://localhost:6767/v1/chat/completions \\${RESET}\n"
printf "    ${CYAN}-H 'Content-Type: application/json' \\${RESET}\n"
printf "    ${CYAN}-d '{\"model\":\"llama3\",\"messages\":[{\"role\":\"user\",\"content\":\"Hello\"}]}'${RESET}\n"
printf "\n${GREEN}Enjoy ppmlx!${RESET}\n"
