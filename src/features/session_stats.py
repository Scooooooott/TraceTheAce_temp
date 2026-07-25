"""Stateless per-session features: pure functions of a transcript's text.

Must produce identical output whether called on a training session (batch,
cached) or a single test session at inference time -- this is the one
implementation both paths call, so it can never depend on anything outside
a single session's transcript.
"""

import os
import re
from pathlib import Path

# Must be set before `import tiktoken` triggers any encoding load: the
# submission container has no network access, so cl100k_base's BPE file
# can't be downloaded on first use like tiktoken normally does. Point it at
# a bundled copy instead (fetched once in dev, see
# scripts/fetch_tiktoken_cache.py) -- same relative path from this file
# whether it's running from src/features/ in this repo or from
# submission_src/src/features/ in the packaged submission, since both copy
# the same src/ subtree structure.
os.environ.setdefault(
    "TIKTOKEN_CACHE_DIR", str(Path(__file__).resolve().parent.parent / "assets" / "tiktoken_cache")
)

import pandas as pd
import tiktoken

SOCRATIC_RE = re.compile(r"why (?:do|did|would|does|is|are) you think|can you explain|how did you", re.I)
METACOG_RE = re.compile(r"does?\s?that make sense|did you understand|are you clear|is that clear", re.I)
PRAISE_RE = re.compile(r"\bwell done\b|\bexcellent\b|\bexactly\b|\bgreat job\b|\bhurray\b", re.I)
SELF_CORRECT_RE = re.compile(r"\bwait,? no\b|\bactually\b|\boh no\b|\bi mean\b", re.I)
# Tutor correcting the STUDENT (distinct from SELF_CORRECT_RE, which is the
# student catching their own mistake). Anchored to turn-start for the bare
# "no" forms to avoid matching "no" as a normal word mid-sentence (e.g. "no
# idea", already covered by UNSURE_RE) -- see experiments.md's Level 4
# transcript-reading notes for why this was added: praise/politeness
# language (PRAISE_RE) turned out NOT to reliably separate correct vs wrong
# outcomes in a hand-read sample (a wrong-outcome session was just as
# praise-heavy as correct ones), but how often the tutor has to correct the
# student is a more direct proxy for how much the student actually
# struggled during the live session.
CORRECTION_RE = re.compile(
    r"^\s*no[,.!]|\bnot quite\b|\bnot correct\b|\bnot right\b|\btry again\b|\bthat'?s not\b|\balmost,? but\b",
    re.I,
)
UNSURE_RE = re.compile(r"i'?m not sure|i don'?t know|no idea", re.I)
UNCLEAR_RE = re.compile(r"\[unclear\]", re.I)
SPEAKER_TAG_RE = re.compile(r"\[speaker\]", re.I)
DIGIT_RE = re.compile(r"\d")

_ENCODING = tiktoken.get_encoding("cl100k_base")


def clean_content(content: pd.Series) -> pd.Series:
    """Strip embedded [Speaker] turn-boundary artifacts before any text stat.

    ~35% of sessions have utterances that merge multiple speakers' turns into
    one row under a single role label (see experiments.md). There isn't
    enough information to reliably reassign the merged segments to the right
    speaker, so this only removes the marker itself -- it does not fix the
    underlying role mis-attribution.

    Public (not module-private) because src/features/llm_prompt.py needs the
    same cleaning when it builds full ordered transcripts for prompts --
    single source of truth rather than a second copy of the regex.
    """
    return content.astype(str).str.replace(SPEAKER_TAG_RE, " ", regex=True)


def compute_session_features(transcript_df: pd.DataFrame) -> dict:
    """transcript_df: utterance_id, role, content, timestamp for one session."""
    content = clean_content(transcript_df["content"])
    role = transcript_df["role"]

    tutor = content[role == "tutor"]
    student = content[role == "student"]
    n_tutor_utt = max(len(tutor), 1)
    n_student_utt = max(len(student), 1)

    tutor_text = " ".join(tutor)
    student_text = " ".join(student)
    n_tokens_tutor = len(_ENCODING.encode(tutor_text))
    n_tokens_student = len(_ENCODING.encode(student_text))
    n_tokens_total = len(_ENCODING.encode(" ".join(content)))

    # giveaway proxy: tutor turn right after a student "not sure/don't know"
    # turn that contains a digit
    giveaway = 0
    unsure_events = 0
    roles_list = role.tolist()
    contents_list = content.tolist()
    for i in range(len(transcript_df) - 1):
        if roles_list[i] == "student" and UNSURE_RE.search(contents_list[i]):
            unsure_events += 1
            if roles_list[i + 1] == "tutor" and DIGIT_RE.search(contents_list[i + 1]):
                giveaway += 1

    n_unclear = content.str.contains(UNCLEAR_RE).sum()

    n_utterances = len(transcript_df)

    # Recency-weighted correction rate: struggle near the end of a session
    # (closest, chronologically, to whatever the follow-up quiz question
    # covers) is hypothesized to matter more than struggle early on that the
    # student went on to recover from. "Late" = final third of the
    # transcript by turn order (transcripts are chronological, see
    # compute_features_for_sessions's docstring).
    late_start = int(n_utterances * 2 / 3)
    late_role = role.iloc[late_start:]
    late_tutor = content.iloc[late_start:][late_role == "tutor"]
    n_late_tutor = max(len(late_tutor), 1)

    return {
        "n_tokens_total": n_tokens_total,
        "n_tokens_tutor": n_tokens_tutor,
        "n_tokens_student": n_tokens_student,
        "n_utterances": n_utterances,
        # tokens-per-utterance, not n_utterances itself: raw utterance count
        # correlates with n_tokens_total at r=0.70 (many short vs few long
        # turns for the same total length is real signal, but the raw count
        # mostly just re-measures "how long was this session"). Dividing by
        # total tokens brings that down to r=0.25 while keeping the pacing
        # information n_utterances alone was too collinear to contribute.
        "avg_utterance_length": n_tokens_total / max(n_utterances, 1),
        "tutor_share": n_tokens_tutor / max(n_tokens_tutor + n_tokens_student, 1),
        "unclear_rate": n_unclear / max(len(transcript_df), 1),
        "socratic_rate": tutor.str.contains(SOCRATIC_RE).sum() / n_tutor_utt,
        "metacog_rate": tutor.str.contains(METACOG_RE).sum() / n_tutor_utt,
        "praise_rate": tutor.str.contains(PRAISE_RE).sum() / n_tutor_utt,
        "self_correct_rate": student.str.contains(SELF_CORRECT_RE).sum() / n_student_utt,
        "correction_rate": tutor.str.contains(CORRECTION_RE).sum() / n_tutor_utt,
        "late_correction_rate": late_tutor.str.contains(CORRECTION_RE).sum() / n_late_tutor,
        "unsure_rate": unsure_events / n_student_utt,
        "unsure_events": unsure_events,
        "giveaway_events": giveaway,
        "giveaway_rate": giveaway / unsure_events if unsure_events else float("nan"),
        "speaker_tag_count": int(
            transcript_df["content"].astype(str).str.contains(SPEAKER_TAG_RE).sum()
        ),
        # Raw text for TF-IDF (src/features/tfidf.py) -- reuses the same
        # cleaned, role-split text already joined above for token counting,
        # no extra pass over the transcript.
        "tutor_text": tutor_text,
        "student_text": student_text,
    }


def compute_features_for_sessions(session_ids, transcripts_dir) -> pd.DataFrame:
    """transcripts_dir: directory containing {session_id}.csv files.

    No dependency on src.data on purpose -- this must run unmodified against
    data/test_transcripts/ in the submission container, not just against the
    training path src.data hardcodes. Callers pass whichever directory
    applies (see src.data.TRAIN_TRANSCRIPTS_DIR for the training-time path).
    """
    transcripts_dir = Path(transcripts_dir)
    rows = []
    for sid in session_ids:
        transcript_df = pd.read_csv(transcripts_dir / f"{sid}.csv")
        feats = compute_session_features(transcript_df)
        feats["session_id"] = sid
        rows.append(feats)
    return pd.DataFrame(rows)
