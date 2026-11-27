# Tuning `--spec auto` (Inkling 2)

`--spec auto` chooses, for every decode step of an Inkling 2 request, a round of the DSpark drafter, a
round of the DFlash2 drafter or an ordinary step
([`DrafterChoice`](../../src/models/inkling2/program/drafter_choice.h); the model side is in
[Inkling 2 model](inkling2-model.md#both-drafters---spec-auto)). The choice weighs how many drafts each
drafter gets accepted against what its round costs, so it moves with anything that changes either:
the drafters, the kernels of a round or of a step, LoopCD, the board. This page says what adapts by
itself, what has to be retuned, and how.

## What adapts by itself

The costs. The Program times every decode step it runs with `--spec auto` (its per-row wall time,
which ends after the execution's own synchronization: no extra synchronization) and keeps, per kind
of step (a DSpark round, a DFlash2 round, a fed ordinary step) and context, an exponential mean of the
ratio to the generated cost table
([`StepCosts`](../../src/models/inkling2/program/drafter_costs.h)). At startup the execution times six
steps and six rounds of each drafter at about 4k tokens, so a different board starts from its own
costs; faster round kernels show within tens of rounds. The choice reads these costs wherever it
compares a round with a step or one drafter with the other.

## What has to be retuned

The constants of the choice and the initial cost table live in one generated file,
[`drafter_tuning.h`](../../src/models/inkling2/program/drafter_tuning.h), with what they were tuned on
in `drafter_tuning.json` beside it. They depend on how the drafters' acceptance behaves on real text
(the far-copy thresholds, the weight and evidence of the means, the exploration of the other drafter
on long prompts, the margin of the ordinary steps), which no live measurement replaces. Retune after
a change to a drafter (weights, K, its context), to LoopCD, to the chat template, or when the quick
check below fails; costs alone do not need it.

[`tools/inkling2_tune/policy.py`](../../tools/inkling2_tune/policy.py) mirrors `DrafterChoice` step
by step: a change to one is a change to the other.

## The tuning corpus

One JSON line per request (`name`, `group`, `body`: an OpenAI chat body). The groups are what the
choice must serve: short prompts of prose and code in each language, greedy and sampled; a long
technical explanation; agentic turns at 100-150k tokens; long free writing; and documents at 4k, 8k,
32k and 128k tokens with two tasks, a summary (which copies from the document) and an essay inspired
by it (which does not), three texts each up to 32k (prefixes `[run 1..3]`, names `summary_N`,
`summary2_N`, `summary3_N`). Texts of 256-384 tokens decide fast or not at all: which drafter wins
differs from text to text by up to 30 %, so a group of one text is noise for the grid. The corpus
stays fixed between tunings, so that two tunings compare; it lives with the measurements, not in this
repository (its agentic turns and documents are private).

## Retune: one command

From a built tree, on both GPUs (`tools/inkling2_tune/tune.sh`):

```bash
tools/inkling2_tune/tune.sh --artifact inkling2_mini_both.ninfer --corpus corpus.jsonl --out /tmp/tune
```

1. **Texts** (`record.py`): a server without drafter, LoopCD off then on, sends every request and gets
   its token ids back from `--prefill-miss-dump`. A text does not depend on the drafter (greedy, or
   sampled at the request's seed), so these are the texts of every `--spec` mode.
2. **Drafts**: the tf driver's `--mode drafts --drafts-k 3` proposes with each drafter at every
   position of every text (teacher forced) and times its rounds; the recording server's decode gives
   the step time at each context.
3. **Grid** (`tune.py`): the simulator walks every text under each combination of the constants, with
   each condition's measured costs, and keeps the combination whose worst group is best against two
   bounds: the better fixed drafter less 2 %, and DSpark less 1 %. It writes `drafter_tuning.h` and
   `drafter_tuning.json` (the constants, the traces' hash, the expected rates per group and the
   runner-up combinations) and prints the expected decode rate of every group against DSpark, DFlash2
   and no drafter.

`--steps record,drafts,tune` runs a part; `--steps tune` reruns the grid on traces already recorded
(minutes, CPU only).

Duration on two RTX 5070 Ti: about 25 minutes per LoopCD condition for the texts and both drafters'
drafts (measured: 51.5 minutes for 63 texts in two conditions), 3 minutes for the grid of 768
combinations on 8 cores, under a minute for the parity check.

## Check: one command

`tools/inkling2_tune/check.sh --artifact ... --corpus ... --out DIR` serves 16 requests of the corpus
(one or two per group) with `--spec auto`, `mtp` and `dflash2`, LoopCD off and on, and fails when
auto decodes more than 2 % below the better fixed drafter in some group (6 minutes). Run it after any
change that touches rounds, ordinary steps or the drafters; when only the costs moved it passes as
before (the engine measures them), when it fails in new groups, retune. Its groups are single texts:
a group a point or two below the bar is noise, a group several points below is not.

## Where the choice stands (v0.16.6)

On the 126 traces of the tuning corpus the generated choice is at worst 4.2 % below the better fixed
drafter and 2.7 % below DSpark; the quick check fails on answers about long documents (a summary of an
8k document without LoopCD, an agentic turn, an essay on an 8k document and a summary of a 4k one with
LoopCD), where which drafter wins changes inside the answer and a 256-token answer leaves too few
rounds to measure both. Proposing with the drafter not chosen without verifying it (a shadow, at the
start of long prompts or every few rounds) did worse on the same traces in every combination tried.
That is why the launcher keeps `--spec mtp` and `--spec auto` is opt-in.

## What to look at

- The table `tune.py` prints: the groups where auto is below the better drafter, and by how much.
  A group that no combination brings within 2 % means the policy lacks a signal for it.
- `drafter_tuning.json`: whether the runner-up combinations are close (a flat optimum) or the best one
  is alone (likely fitted to the corpus' noise).
- After the check passes, the live bench of the release (per group, n = 3, LoopCD off and on) remains
  the measurement that counts.
