# Not a held-out score

These 10 tasks were selected because the models already disagreed at thinking `none`. On those same tasks at `none`, the base model scored 7/10 and the pilot scored 3/10. The table below is only the later `high` comparison. Its +1 / −0 line is not an improvement and must not be averaged with 11/20 or 23/60.

| Model | Tasks | Pass@1 | Median Turns | Median Tool Calls | Edit Error Rate |
|---|---:|---:|---:|---:|---:|
| Inkling-2-Mini (base) | 10 | 0.0% | 10.0 | 10.0 | 0.0% |
| PINGI | 10 | 10.0% | 17.0 | 15.0 | 0.0% |

Paired diff at thinking `high` only: +1 newly solved / −0 lost. The base model had solved 7 of these 10 at thinking `none`.
