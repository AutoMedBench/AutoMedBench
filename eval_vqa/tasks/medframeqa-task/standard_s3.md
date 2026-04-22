Validation requirements are unchanged from lite:

- smoke test on `1-10` questions first
- inspect raw output
- verify the parser picks exactly one label in `A-E` and that all frames were shown to the model
- confirm output files match schema (one letter A-E) before any larger slice
