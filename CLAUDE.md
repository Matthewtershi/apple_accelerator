# apple_accel

Private local AI stack for a base M4 Mac mini (16 GB). Two parts:
- `README.md`: research catalog. Docs only.
- `engine/`: **bonsai-engine**, a C++20 inference engine for Ternary Bonsai 8B on the MLX C++ API. `engine/README.md` is the build plan and source of truth for decisions, layout and milestones (M1–M12).

## Who you're working with
The owner is learning C++ through this project. **They write every line of code themselves, from scratch, and must understand each one.**

### Who writes the code
- **Don't write implementation code** (C++, CMake, presets, Python scripts, tests) into files unless the owner explicitly asks for that specific piece.
- Default: **outline**, with what goes in each file and *why*, naming the APIs or commands to look up. When the owner asks for the code, show it in chat with why-comments so they can type it in.
- When the owner is stuck: explain the concept or point at the bug. Offer a fix only if asked.
- When asked to review: correctness first, then comments against the Comments section below.
- Claude may edit `CLAUDE.md` and `.gitignore` freely. Other docs (`README.md`, `engine/README.md`) only when asked.
- The `implementer` agent is only for work the owner explicitly hands off. Default to the `reviewer` agent.

## Workflow
- Work one milestone (or less) at a time. Don't build ahead of the current milestone.
- Plan before editing anything non-trivial; state which files change and why.
- Correctness gate before performance gate: a change isn't done until its test passes.
- If something contradicts `engine/README.md`, stop and ask. Don't silently diverge.
- Dev box is WSL (MLX CPU backend). Never present WSL timings as benchmarks.

## Build & test
_Not set up yet. Fill in once M1 lands (CMake presets `linux-cpu-debug`, `macos-metal-release`; Catch2 tests)._

## C++ conventions
- C++20, `namespace bonsai`, `namespace mx = mlx::core;`
- Headers in `engine/include/bonsai/`, one `.cpp` per header in `engine/src/`.
- Shapes are constants from `ModelConfig`; check at load time and fail loudly.
- MLX is lazy. Never trigger `eval`/`.item()`/printing inside the decode hot loop by accident.
- `-Wall -Wextra -Wpedantic -Werror` must stay clean.

## Comments
Goal: the owner can read the code top to bottom and understand *why*.
- Explain **why**, not what. Never restate the code (`// increment i`).
- One short line above a non-obvious block. No paragraphs; if it needs more, link the README section (`// see engine/README.md §4.1`).
- Every function in a header gets a one-line purpose comment.
- Call out C++ idioms a learner may not know, once per file, briefly: `// std::move: hand off the buffer, no copy`.
- Flag hidden assumptions: shapes, dtypes, laziness, alignment (`// lazy: nothing runs until mx::eval below`).
- Don't comment obvious code. Fewer, better comments beat many.
