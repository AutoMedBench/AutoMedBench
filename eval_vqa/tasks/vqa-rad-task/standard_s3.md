Validation requirements are unchanged from lite:

- smoke test on `1-10` questions first
- inspect raw output
- confirm yes/no questions receive `yes` or `no` and open questions get a short phrase
- confirm output files match schema (open-ended text answer (yes/no for closed questions, short phrase for open)) before any larger slice
