# Write technical English

This guide controls technical prose in the repository and in agent sessions.
It applies to policies, specs, records, documentation, READMEs, API text, error
messages, progress updates, decisions, questions, and final reports.

## Repository rules take precedence

`AGENTS.md` is the binding policy. This guide controls language and document
structure. It cannot add unrelated project policy or weaken an obligation,
prohibition, authority boundary, gate, or evidence requirement. Keep exact
technical names and binding terms, even when a shorter synonym exists.

If another repository guide defines a document's content, follow that guide for
content and this guide for the words. If the two conflict, follow `AGENTS.md`.

## Use three writing passes

1. Read nearby text. Use the same name for each component, state, and action.
2. Apply Simplified Technical English (STE) to each sentence.
3. Apply the document conventions in this guide to headings, lists, links,
   code, tables, and accessibility.

Do not make a sentence less precise to make it shorter. Technical names such as
`tensor`, `goroutine`, `quantization`, `LoRA`, and `rebase` remain technical
names.

## Write clear sentences

- Put one idea in each sentence.
- Limit procedural sentences to 20 words.
- Limit descriptive sentences to 25 words.
- Use no more than six sentences in a procedural paragraph.
- Use active voice and name the actor.
- Put the subject, verb, and object close together.
- Use simple present for current behavior.
- Use simple past for an observed event.
- Use the imperative for an instruction.
- Do not use present perfect when simple past or present works.
- Do not use future tense for current behavior.
- Keep noun clusters to three words.
- Keep articles and connectors. Write `the file`, not `file`.
- Expand an abbreviation on first use. Then use it consistently.

Use one word for one meaning. Use one meaning for one word. A reader must be
able to search for one term and find every use of the concept.

Prefer plain verbs:

| Avoid | Use |
|---|---|
| utilize, leverage | use |
| facilitate | help, or name the action |
| perform | do or run |
| provide | give, return, or offer |
| obtain | get |
| terminate | stop or end |
| initiate, commence | start |
| in order to | to |
| due to the fact that | because |
| in the event that | if |
| prior to | before |

Keep an established technical verb when it names an exact operation. Examples
include `compile`, `deserialize`, `quantize`, and `rebase`.

Do not use marketing terms without a measured fact. Remove words such as
`robust`, `comprehensive`, `powerful`, `seamless`, `production-grade`,
`best-in-class`, `revolutionary`, and `streamline`. Give the tested behavior or
the measured value instead.

Do not use words that blame the reader. Remove `simply`, `just`, `easily`,
`obviously`, `clearly`, `of course`, `trivial`, and `straightforward`.

Replace vague quantities with exact values. Name the cases instead of writing
`several`, `various`, `many`, `some`, `usually`, or `in most cases`.

Avoid words with two likely meanings:

| Avoid | Use |
|---|---|
| once | after, when time is intended |
| since | because, when cause is intended |
| while | although, when contrast is intended |
| may | `can`, `might`, or `is allowed to` |
| should | `must` or `is expected to` |
| above, below | earlier, later, or a link |
| following | the next, after, or a link |
| handle | name the action |
| support | works with, accepts, or runs |
| dropped | removed, deleted, or lost |

Put `only` next to the word it limits. Name a noun again when `it`, `this`, or
`that` can refer to more than one thing.

## Write procedures and warnings

Start an instruction with the command verb. Put one action in each numbered
step. State the reader's location before a user-interface action. State the
result when the reader can use it as a check.

Put a condition before its action:

- Before: Run `make clean` if the build fails.
- After: If the build fails, run `make clean`.

Put a warning before the instruction that can cause damage. State the
condition, the concrete consequence, and the safe action. Use `Warning` for
data loss or damage. Use `Note` for information that prevents a mistake.

## Structure documents

- Use sentence case for titles and headings.
- Use a verb phrase for a task heading and a noun phrase for a concept heading.
- Keep headings unique and correctly nested.
- Address the reader as `you`. Avoid `we` unless the project team is the exact
  actor.
- Use present tense for shipped behavior.
- Use numbered lists for ordered steps and bullets for other lists.
- Keep list items parallel. Use a serial comma.
- End a complete sentence in a list with a period. Do not add punctuation to a
  fragment.
- Use descriptive link text. Do not write `click here`, `this link`, or a bare
  URL.
- Put identifiers, paths, flags, commands, environment variables, and literal
  values in code font.
- Put user-interface element names in bold.
- Do not put a prompt character in a command that the reader copies.
- Give every table a header row. Write `None` or `n/a` instead of an empty cell.
- Keep a table cell to a phrase. Use prose when a cell needs three sentences.

Use numerals for measurements, versions, and comparisons. Put a space between
a value and its unit, for example `500 ms`. Write ranges with `to`. Write dates
as `15 January 2025`.

Do not use em dashes, semicolons, or slashes that mean "and" or "or." Do not
put a required fact in parentheses. Do not use Latin abbreviations. Write `for
example`, `that is`, and `and so on`.

## Write for accessibility and a global audience

- Give informative images alt text. Give decorative images empty alt text.
- Do not make an instruction depend on color, position, or sight alone.
- Do not put required text only in a screenshot.
- Use heading levels and list markup to carry structure.
- Use they or them when you do not know a person's pronouns.
- Avoid idioms, sports metaphors, military metaphors, and culture-bound humor.
- Use American spelling in repository prose.
- Use inclusive terms such as `primary/replica`, `controller/worker`, and
  `allowlist/denylist`.

## Write session prose

Session messages follow the same rules as repository documents.

- A progress update states the completed result, the current action, and a
  blocker when one exists.
- A decision states the available evidence, the selected option, and the
  consequence.
- A question asks for one missing value. It states why the value changes the
  next action.
- A final report leads with the outcome. It gives exact gates, unresolved risk,
  and the commit or pull request when applicable.
- A failure report separates observed facts from inferences.

Do not praise a plan, narrate routine tool use, or hide uncertainty behind
confident language. Use the binding states `PENDING`, `FAILING`, `BLOCKED`,
`NEEDS_CONTEXT`, and `NEEDS_DECISION` when the protocol defines them.

## Review the draft

Check each item:

1. Nearby documents use the same term for each concept.
2. Every sentence has one idea and stays within its applicable length limit.
3. Every action has a named actor or uses the imperative.
4. Every condition and warning appears before its action.
5. Headings use sentence case, and lists use parallel grammar.
6. Links, code, tables, and images meet the document conventions.
7. The text has no vague quantity, marketing claim, reader-blaming word, or
   ambiguous pronoun.
8. The rewrite preserves every technical fact and policy condition.
