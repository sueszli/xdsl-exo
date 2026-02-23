# Code Style

We aim to follow these rules for all changes in this repository:

- We aim for consistency in the code style and architectural patterns throughout the
  codebase in order to make it as easy as possible to understand and modify any part of
  the code.

- We fix issues immediately rather than relying on future refactoring, as technical debt
  tends to accumulate and become harder to address over time.

- We prefer simplicity: no code is better than obvious code, which is better than clever
  code. Premature abstraction often adds complexity without clear benefit.

- We prioritize code locality over DRY (Don't Repeat Yourself). Keeping related logic close
  together - even if it results in slight duplication - makes it easier to understand
  code in isolation. We minimize variable scope.

- We write self-describing code by using descriptive variable names and constant
  intermediary variables rather than relying heavily on comments.

- We use guard-first logic, handling edge cases, invalid inputs and errors at the start
  of functions. Returning early keeps the "happy path" at the lowest indentation level,
  making the main logic easier to follow.

- We keep if/else blocks small and avoid nesting beyond two levels when possible, as
  flat structures are easier to read and reason about.

- We centralize control flow in parent functions, keeping leaf functions as pure logic.
  This separation makes the codebase more predictable and testable.

- We fail fast by detecting unexpected conditions immediately and raising exceptionsExpand commentComment on line R193
  rather than corrupting state, as this makes debugging easier.

- We follow the Python philosophy of
  "[ask for forgiveness not permission](https://docs.python.org/3/glossary.html#term-EAFP)":
  assume keys and attributes exist and catch exceptions when they don't. For single-value
  lookups, prefer the walrus operator to avoid a double lookup:

  ```python
  # Good: single lookup
  if (value := mapping.get(key)) is None:
      raise MyException()

  # Good: EAFP, when no sentinel is available
  try:Expand commentComment on lines R195 to R206
      return mapping[key]
  except KeyError:
      return default_value

  # Bad: LBYL, double lookup
  if key not in mapping:
      raise MyException()
  return mapping[key]
  ```
