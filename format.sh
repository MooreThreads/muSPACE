#!/usr/bin/env bash

# Copyright (c) 2023 by FlashInfer team.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#   http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#
# Modified from MATE and https://github.com/tile-ai/tilelang/blob/main/format.sh.

set -eo pipefail

install_or_update_yapf() {
    local REQUIRED_VERSION="0.40.2"
    if ! command -v yapf &>/dev/null; then
        echo "yapf not found. Installing yapf==$REQUIRED_VERSION..."
        pip install "yapf==$REQUIRED_VERSION"
    else
        local INSTALLED_VERSION
        INSTALLED_VERSION=$(yapf --version | awk '{print $2}')
        if [[ "$INSTALLED_VERSION" != "$REQUIRED_VERSION" ]]; then
            echo "Incorrect yapf version ($INSTALLED_VERSION). Installing yapf==$REQUIRED_VERSION..."
            pip install "yapf==$REQUIRED_VERSION"
        fi
    fi
}

install_or_update_ruff() {
    local REQUIRED_VERSION="0.6.5"
    if ! command -v ruff &>/dev/null; then
        echo "ruff not found. Installing ruff==$REQUIRED_VERSION..."
        pip install "ruff==$REQUIRED_VERSION"
    else
        local INSTALLED_VERSION
        INSTALLED_VERSION=$(ruff --version | awk '{print $2}')
        if [[ "$INSTALLED_VERSION" != "$REQUIRED_VERSION" ]]; then
            echo "Incorrect ruff version ($INSTALLED_VERSION). Installing ruff==$REQUIRED_VERSION..."
            pip install "ruff==$REQUIRED_VERSION"
        fi
    fi
}

install_or_update_codespell() {
    local REQUIRED_VERSION="2.3.0"
    if ! command -v codespell &>/dev/null; then
        echo "codespell not found. Installing codespell==$REQUIRED_VERSION..."
        pip install "codespell==$REQUIRED_VERSION"
    else
        local INSTALLED_VERSION
        INSTALLED_VERSION=$(codespell --version | awk '{print $2}' | sed 's/^v//')
        if [[ "$INSTALLED_VERSION" != "$REQUIRED_VERSION" ]]; then
            echo "Incorrect codespell version ($INSTALLED_VERSION). Installing codespell==$REQUIRED_VERSION..."
            pip install "codespell==$REQUIRED_VERSION"
        fi
    fi
}

install_or_update_clang_format() {
    local REQUIRED_VERSION="15.0.7"
    if ! command -v clang-format &>/dev/null; then
        echo "clang-format not found. Installing clang-format==$REQUIRED_VERSION..."
        pip install "clang-format==$REQUIRED_VERSION"
    else
        local INSTALLED_VERSION
        INSTALLED_VERSION=$(clang-format --version | awk '{print $3}')
        if [[ "$INSTALLED_VERSION" != "$REQUIRED_VERSION" ]]; then
            echo "Incorrect clang-format version ($INSTALLED_VERSION). Installing clang-format==$REQUIRED_VERSION..."
            pip install "clang-format==$REQUIRED_VERSION"
        fi
    fi
}

install_or_update_yapf
install_or_update_ruff
install_or_update_codespell
install_or_update_clang_format

builtin cd "$(dirname "${BASH_SOURCE:-$0}")"
ROOT="$(git rev-parse --show-toplevel)"
builtin cd "$ROOT" || exit 1

base_branch() {
    if git show-ref --verify --quiet refs/remotes/origin/master; then
        echo "origin/master"
    else
        echo "master"
    fi
}

changed_files() {
    local MERGEBASE
    MERGEBASE="$(git merge-base "$(base_branch)" HEAD)"
    git diff --name-only --diff-filter=ACM "$MERGEBASE" -- "$@"
}

echo 'muSPACE yapf: Check Start'

YAPF_FLAGS=(
    '--recursive'
    '--parallel'
)

YAPF_EXCLUDES=(
    '--exclude' 'build/**'
    '--exclude' '.eggs/**'
    '--exclude' 'model_examples/**/checkouts/**'
)

format() {
    yapf --in-place "${YAPF_FLAGS[@]}" "$@"
}

format_changed() {
    local FILES
    FILES=$(changed_files '*.py' '*.pyi')
    if [[ -n "$FILES" ]]; then
        echo "$FILES" | xargs -P 5 yapf --in-place "${YAPF_EXCLUDES[@]}" "${YAPF_FLAGS[@]}"
    fi
}

format_all() {
    yapf --in-place "${YAPF_FLAGS[@]}" "${YAPF_EXCLUDES[@]}" mu_space tests model_examples
}

# TODO: enable yapf auto format if muSPACE adopts a shared YAPF configuration.
echo 'Skip yapf auto format'

echo 'muSPACE codespell: Check Start'

spell_check() {
    codespell "$@"
}

spell_check_all() {
    codespell --toml pyproject.toml
}

spell_check_changed() {
    local FILES
    FILES=$(changed_files '*.py' '*.pyi')
    if [[ -n "$FILES" ]]; then
        echo "$FILES" | xargs codespell
    fi
}

if [[ "$1" == '--files' ]]; then
    spell_check "${@:2}"
elif [[ "$1" == '--all' ]]; then
    spell_check_all
else
    spell_check_changed
fi

echo 'muSPACE codespell: Done'

echo 'muSPACE ruff: Check Start'

lint() {
    ruff check "$@"
}

lint_changed() {
    local FILES
    FILES=$(changed_files '*.py' '*.pyi')
    if [[ -n "$FILES" ]]; then
        echo "$FILES" | xargs ruff check
    fi
}

if [[ "$1" == '--files' ]]; then
    lint "${@:2}"
elif [[ "$1" == '--all' ]]; then
    lint mu_space tests model_examples
else
    lint_changed
fi

echo 'muSPACE ruff: Done'

echo 'muSPACE clang-format: Check Start'

if command -v clang-format &>/dev/null; then
    CLANG_FORMAT_FLAGS=("-i")

    clang_format() {
        clang-format "${CLANG_FORMAT_FLAGS[@]}" "$@"
    }

    clang_format_all() {
        find csrc -type f \( -name '*.c' -o -name '*.cc' -o -name '*.cpp' -o -name '*.h' -o -name '*.hpp' -o -name '*.mu' -o -name '*.muh' \) \
            -not -path "./build/*" \
            -exec clang-format -i {} +
    }

    clang_format_changed() {
        local FILES
        FILES=$(changed_files '*.c' '*.cc' '*.cpp' '*.h' '*.hpp' '*.mu' '*.muh')
        if [[ -n "$FILES" ]]; then
            echo "$FILES" | xargs clang-format -i
        fi
    }

    if [[ "$1" == '--files' ]]; then
        clang_format "${@:2}"
    elif [[ "$1" == '--all' ]]; then
        clang_format_all
    else
        clang_format_changed
    fi
else
    echo "clang-format not found. Skipping C/C++ formatting."
fi

echo 'muSPACE clang-format: Done'

if ! git diff --quiet &>/dev/null; then
    echo 'Reformatted files. Please review and stage the changes.'
    echo 'Changes not staged for commit:'
    echo
    git --no-pager diff --name-only
    exit 1
fi

echo 'muSPACE: All checks passed'
