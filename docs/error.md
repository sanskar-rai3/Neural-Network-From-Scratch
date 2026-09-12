# Error Handling

Error handling is intentionally avoided in performance-critical sections of the codebase, such as `matrix.c`, `loss.c`and `activation.c`.

Instead, `assert` is used to catch programming errors and validate assumptions during development. Assertions are disabled in release builds, eliminating their runtime overhead in production.
