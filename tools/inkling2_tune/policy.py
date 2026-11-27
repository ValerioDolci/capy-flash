"""The choice of --spec auto (src/models/inkling2/program/drafter_choice.cpp) over recorded traces, for tune.py: the same
decisions in the same order, so that a grid over its constants predicts the engine.

A trace is a text (prompt + generated ids, drafter-independent) with each drafter's K drafts at every anchor; a round of
drafter d at anchor p accepts the longest prefix of its drafts equal to the text, within the round's extent. Costs are
milliseconds per round of each drafter and per fed ordinary step, by context (Costs)."""

from collections import deque


class FarCopies:
    """The far-copy prior of DrafterChoice: for every committed generated token, whether its 4-gram last ended more than
    `window` positions back (and not since)."""

    def __init__(self, seq, n_prompt, window, recent):
        self.seq, self.n, self.window, self.recent = seq, n_prompt, window, recent
        self.last = {}
        self.seen = 0
        self.tail = deque(maxlen=recent)

    def advance(self, upto):
        seq = self.seq
        while self.seen < min(upto, len(seq)):
            i = self.seen
            if i >= 3:
                g = tuple(seq[i - 3 : i + 1])
                found = self.last.get(g)
                if i >= self.n:
                    self.tail.append(found is not None and found + self.window <= i)
                self.last[g] = i
            self.seen += 1

    def share(self):
        return sum(self.tail) / len(self.tail) if len(self.tail) == self.recent else None


def walk(text, costs, c, K=3, steps=None):
    """Decode time (ms) of `text` under --spec auto with constants `c` (a dict, the names of drafter_tuning.h); with a list
    `steps`, appends the steps as drafter_replay prints them (M<a>, D<a>, -)."""
    N, p = len(text.seq), text.n
    s = text.sampled
    ms = 0.0
    prior = "dsp" if text.n >= c["kLongPrompt"] else "df2"
    far = FarCopies(text.seq, text.n, c["kWindow"], c["kRecent"])
    mean = {"dsp": 0.0, "df2": 0.0}
    rounds = {"dsp": 0, "df2": 0}
    current = prior
    plain, since, explore = False, 0, 0
    explores = text.n >= c["kLongPrompt"] and c["kExplore"] > 0
    stats = {"rounds": 0, "steps": 0, "explore": 0}
    while p + 1 < N:
        far.advance(p + 1)
        share = far.share()
        if share is not None:
            if prior == "df2" and share >= c["kEnter"]:
                prior = "dsp"
            elif prior == "dsp" and share <= c["kLeave"]:
                prior = "df2"
        if plain and since < c["kProbe"]:
            since += 1
            ms += costs.step(p, s)
            stats["steps"] += 1
            if steps is not None:
                steps.append("-")
            p += 1
            continue
        # The proposer: the prior until both drafters hold kEvidence rounds, then the better measured rate, with
        # hysteresis against the current one; on long prompts the other drafter every kExplore rounds.
        choice = prior
        if c["kCompare"] and rounds["dsp"] >= c["kEvidence"] and rounds["df2"] >= c["kEvidence"]:
            rate = {d: (1.0 + mean[d]) / costs.round(d, p, s) for d in rate_keys()}
            other = "df2" if current == "dsp" else "dsp"
            choice = other if rate[other] > rate[current] * (1.0 + c["kHysteresis"]) else current
            if c["kPriorWins"] and prior == "dsp" and share is not None and share >= c["kEnter"]:
                choice = "dsp"
        drafter = choice
        started = rounds["dsp"] + rounds["df2"]
        other = "df2" if choice == "dsp" else "dsp"
        gap = None
        if rounds["dsp"] >= 2 and rounds["df2"] >= 2:
            r = {d: (1.0 + mean[d]) / costs.round(d, p, s) for d in rate_keys()}
            gap = r[other] / r[choice]
        settled = gap is not None and gap < 1.0 - c.get("kSettle", 1.0)
        if text.n >= c["kLongPrompt"] and started < 2 * c.get("kStart", 0) and not settled:
            drafter = ("dsp", "df2")[started % 2]
        elif explores and not settled:
            explore += 1
            if explore >= c["kExplore"]:
                drafter = other
                explore = 0
                stats["explore"] += 1
        extent = min(K, N - p - (1 if text.stopped else 2))
        a = text.accepted(drafter, p, extent)
        if steps is not None:
            steps.append(("M" if drafter == "dsp" else "D") + str(a))
        ms += costs.round(drafter, p, s)
        stats["rounds"] += 1
        rounds[drafter] += 1
        weight = c["kWeight"] if drafter == choice and not plain else c.get("kOtherWeight", c["kWeight"])
        mean[drafter] += max(weight, 1.0 / rounds[drafter]) * (a - mean[drafter])
        since = 0
        if drafter == choice:
            current = choice
        measured = [d for d in rate_keys() if rounds[d] >= c["kEvidence"]]
        if measured:
            best = max((1.0 + mean[d]) / costs.round(d, p, s) for d in measured)
            plain = best < (1.0 - c["kMargin"]) / costs.step(p, s)
        p += a + 1
    return ms, stats


def rate_keys():
    return ("dsp", "df2")


def fixed(text, costs, drafter, K=3):
    """Decode time of `text` with one drafter (--spec mtp or dflash2), or without (drafter None)."""
    N, p, ms = len(text.seq), text.n, 0.0
    while p + 1 < N:
        if drafter is None:
            ms += costs.plain_step(p, text.sampled)
            p += 1
            continue
        extent = min(K, N - p - (1 if text.stopped else 2))
        a = text.accepted(drafter, p, extent)
        ms += costs.single_round(drafter, p, text.sampled)
        p += a + 1
    return ms
