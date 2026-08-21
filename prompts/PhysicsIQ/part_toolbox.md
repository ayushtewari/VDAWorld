The following WorldAPI methods are available to your simulator via `self.api`. Call `get_api_documentation` with a list of method names to retrieve their full signatures and docstrings before writing code that uses them.

[AVAILABLE_TOOLS]

## Robust API usage — mandatory fallback principle

**After every API call inside `fit`, check whether the returned value is usable.** An API call may succeed but return an empty or degenerate result (e.g. an empty array, zero-magnitude values, fewer items than expected, or `None`). When this happens you must not leave the simulator in an uninitialised state. Instead, substitute a sensible geometric prior and log that you have done so.

The guiding principle is: **your intuition about the scene is always better than producing nothing**. A simulator that initialises from a rough geometric guess and runs correctly while logging that issue is more useful than one which produces an output where items are missing unexpectedly.

**Important distinction — what to guard and what to let through:**

- **Guard against degenerate _results_**: after the call returns, check the value (e.g. `if result is None or len(result) == 0:`). This is where you apply the geometric fallback.
- **Do NOT suppress exceptions from incorrect API usage**: if you call a method with the wrong name, wrong argument types, or violate the API contract, the exception must propagate. Wrapping the entire call in a bare `except Exception: pass` or similar hides bugs in your own code and prevents the error-correction loop from fixing them. Only use `try/except` to guard against documented failure modes (e.g. a depth estimator that may raise when the image has no valid depth), and always re-raise or log with full detail.
