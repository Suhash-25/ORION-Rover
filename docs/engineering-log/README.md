# ORION engineering log

Everything that happened while building ORION's software, so nobody has to guess later.

| File | What goes in it | When to write |
| --- | --- | --- |
| [JOURNAL.md](JOURNAL.md) | Chronological log: every request, plan, change of plan, result. Newest at the bottom. | Every working session, as it happens |
| [DECISIONS.md](DECISIONS.md) | Design decisions with the options rejected and why. | When a choice is made that someone could later question |
| [MISTAKES.md](MISTAKES.md) | Mistakes and bugs: what went wrong, root cause, what we learnt, the guard that now prevents it. | Every time something fails that we thought worked |
| [../../LESSONS.md](../../LESSONS.md) | Short lessons in `hx` format (`hx lesson add` appends here). | Same as MISTAKES, one-paragraph summary |

Rules:
- Facts with evidence (command, test name, number), not impressions.
- A "pass" is a command that exited 0, quoted with the command. No `echo` checks.
- Never delete an entry; add a correction below it.
