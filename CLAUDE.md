# Instructions for Claude Code (expert-poker-player)

## Git & Commit Conventions

- **Commit Format**: Always format commit messages as:
  `[branch_name] type(scope): short description in present tense`
  - Types: `feat`, `fix`, `ref`, `chore`, `styling`, `docs`, `test`
  - Example: `[main] feat(agent): add CFR algorithm base implementation`
  - Example: `[feature/poker-engine] fix(evaluator): fix straight flush edge case`

## Project Context

- **Repository**: `expert-poker-player`
- **Purpose**: Master's Thesis project focused on AI poker algorithms and strategy.
- **Language & Stack**: Python 3.x / PyTorch / Gymnasium / Pytest

## Guidelines & Best Practices

- **Code Style**:
  - Write modular, clean, and well-typed code (type hints preferred).
  - Keep AI/game logic decoupled from UI and evaluation scripts.
- **Language Preference**: Communicate with me in **Polish**, but write all code, inline comments, commit messages, and documentation in **English** on level B2, with one exception comments for functions write in **Polish**
- **Commands**:
  - Run tests: `pytest`
  - Code formatting: `black .` / `ruff check .`
