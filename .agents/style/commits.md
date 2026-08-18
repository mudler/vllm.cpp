# Write commits and pull requests

This guide controls writing that ships with a change. It applies to commit
subjects and bodies, pull request titles and descriptions, branch names,
changelog entries, and release-note lines.

## Repository rules take precedence

`AGENTS.md` is the binding policy. This guide cannot add unrelated project
policy or weaken a repository rule. Apply these repository rules before the
generic writing rules in this guide:

- Use the dominant `type(ROW-ID): subject` convention for row commits.
- Use `row/<ROW-ID>` for a claimed row branch.
- Give every commit an authored prose body. The body explains the reason for
  the change. The protocol marker and trailers do not count as authored prose.
- Put a bare `FOLLOWING_AGENTS_PROTOCOL` paragraph before the trailers.
- Add `Following-Agents-Protocol`, `AI-Assisted`, and `Assisted-by` trailers in
  the exact form that `AGENTS.md` requires.
- Do not add an AI `Signed-off-by` or `Co-Authored-By` trailer.
- Never rewrite or force-push `main`.

If this guide conflicts with `AGENTS.md`, follow `AGENTS.md`.

## Detect the existing convention

Before you draft text, run:

```sh
git log --format='%s' -30
```

Read `CONTRIBUTING.md`, `.gitmessage`, commit-lint configuration, and the pull
request template when they exist. Follow the dominant pattern in the last 30
commits. Do not introduce a new prefix, capitalization rule, or tense. If a
repository has fewer than five commits, use an imperative sentence.

The current `vllm.cpp` convention uses a lowercase type, an uppercase row ID in
parentheses, and a lowercase imperative subject:

```text
policy(POLICY-SINGLE-PR-AND-STYLE): require reasons in commit bodies
```

## Write the commit subject

Write a subject that completes this sentence: "If applied, this commit will
___." Name the observable change, not the activity that produced it.

- Aim for 50 characters after the prefix. Do not exceed 72 characters for the
  complete subject when the repository convention permits it.
- Follow the capitalization of nearby commits.
- Use the imperative mood.
- Do not end the subject with a period.
- Make the subject understandable without the pull request open.
- Use one term for each concept in the subject and body.

The style checker enforces the final-period rule. It does not enforce a length
limit because the measured repository history does not meet one. The length
limit remains a writing and review rule.

Avoid subjects such as `Update files`, `Fix bug`, `Refactor loader`, `Improve
performance`, `Address review comments`, and `WIP`. Name the exact change or its
measured effect.

## Write the commit body

Separate the subject and body with one blank line. Every commit in this
repository needs authored prose after the subject. Explain what changed and why
the change is needed. The reader already has the diff, so do not walk through
the changed files.

Get the reason from the diff, the linked issue, and the session evidence. Never
invent a reason. If no motivation is in the evidence, state what changed and
stop.

Use these rules:

- Put one idea in each sentence.
- Prefer 20 words or fewer in a sentence.
- Use active voice and name the actor.
- Use simple present or simple past. Do not use present perfect.
- Keep noun clusters to three words.
- Wrap body prose at 72 columns when practical.
- Keep one change in one commit. If the body needs "also," split the change.
- Do not use `comprehensive`, `robust`, `streamline`, `leverage`, `simply`, or
  `just`.

Put the protocol marker and trailers after the authored prose. Separate the
marker, the trailer block, and the authored prose with blank lines.

## Write pull request titles and descriptions

Write a pull request title with the same rules as a commit subject. A squash
merge can make this title permanent Git history.

Follow the repository pull request template. The description answers four
questions:

1. What changed?
2. Why is the change needed?
3. How can a reviewer verify it?
4. What remains unverified or out of scope?

Give exact commands and observed results. Do not write only "tested locally."
Link the tracking issue. Use a closing keyword when the pull request completes
the issue, for example `Closes #123`.

Do not use a screenshot as the only description. Images can become unavailable
and cannot replace searchable text.

**Never put a bare `---` line in a pull request body.** The repository sets
`squash_merge_commit_message = PR_BODY`, so the body becomes the landed commit
message, and `git interpret-trailers` treats a bare `---` as the end of that
message. Everything below it — including the trailer block — becomes invisible
to the parser, so `commit-protocol-tag` reports trailers the body plainly
carries as missing. A markdown horizontal rule is the usual way this happens.
Measured on pull request #950: two `---` rules, trailer block present and
correct at the end, `git interpret-trailers --parse` returned nothing; deleting
the two rules returned all three trailers. Use a heading to separate sections
instead. A markdown table's `|---|---|` separator is not affected, because the
line is not bare.

## Name branches

For a claimed row, use the binding `row/<ROW-ID>` format. For other work, follow
the nearby branch convention. Use a short, descriptive, kebab-case name. Do not
use a ticket number or personal shorthand by itself.

## Write changelog and release-note lines

Describe the user-visible effect, not the implementation:

- Before: `Add a nil check in the model loader`
- After: `Fix a crash when a model directory has no config.yaml`

Put one change in each entry. Name the version or date when behavior stops or a
feature is removed.

## Review before you commit

Check each item:

1. The message follows the detected repository convention.
2. The subject is imperative, specific, concise, and has no final period.
3. A blank line separates the subject from the authored prose body.
4. The body states the reason without inventing one.
5. The body does not repeat the diff file by file.
6. Each sentence has one idea and uses one term per concept.
7. The protocol marker and trailers are exact and at the end.
8. The commit contains one coherent change.
