"""The answer-text renderer drops duplicate tables (ADR-0012, R-107).

The UI draws result tables from the tool payload, so a markdown table in the model's
prose is always a second, paraphrased copy of the one below it. The prompts ask the model
not to write them; these tests cover the enforcement, because prompt compliance is not a
guarantee and `gpt-4o-mini` demonstrably ignores it.

The interesting half is the negative cases: a stripper that also eats prose, lists or
code would be a worse bug than the duplication it fixes.
"""

from __future__ import annotations

from movieagent.ui.text import strip_markdown_tables

TABLE_ANSWER = """\
The 10 most common movie genres are:

| Genre | Movie Count |
| --- | --- |
| Drama | 2297 |
| Comedy | 1722 |
| Thriller | 1274 |

Drama leads by a wide margin."""


class TestStripping:
    def test_a_table_is_removed_and_the_prose_survives(self) -> None:
        cleaned = strip_markdown_tables(TABLE_ANSWER)
        assert "Drama | 2297" not in cleaned
        assert "| --- |" not in cleaned
        assert cleaned.startswith("The 10 most common movie genres are:")
        assert cleaned.endswith("Drama leads by a wide margin.")

    def test_alignment_colons_are_recognised(self) -> None:
        text = "Results:\n\n| a | b |\n|:---|---:|\n| 1 | 2 |\n\nDone."
        assert strip_markdown_tables(text) == "Results:\n\nDone."

    def test_two_tables_both_go(self) -> None:
        text = (
            "First:\n\n| a | b |\n| --- | --- |\n| 1 | 2 |\n\n"
            "Second:\n\n| c | d |\n| --- | --- |\n| 3 | 4 |\n\nEnd."
        )
        cleaned = strip_markdown_tables(text)
        assert "|" not in cleaned
        assert cleaned == "First:\n\nSecond:\n\nEnd."

    def test_a_table_only_answer_is_left_alone(self) -> None:
        """A bare table is a poor answer; a blank message is a worse one."""
        text = "| Genre | Count |\n| --- | --- |\n| Drama | 2297 |"
        assert strip_markdown_tables(text) == text


class TestWhatItMustNotTouch:
    def test_prose_containing_a_pipe_stays(self) -> None:
        text = "The filter was `genre in ['Drama'] | year >= 2010` and it matched 12 films."
        assert strip_markdown_tables(text) == text

    def test_a_bullet_list_stays(self) -> None:
        text = "Top three:\n- Drama (2,297)\n- Comedy (1,722)\n- Thriller (1,274)"
        assert strip_markdown_tables(text) == text

    def test_a_fenced_code_block_stays(self) -> None:
        text = "Query:\n\n```sql\nSELECT a | b\n| --- |\nFROM t\n```\n\nThat is the query."
        assert strip_markdown_tables(text) == text

    def test_empty_and_plain_text_are_returned_unchanged(self) -> None:
        assert strip_markdown_tables("") == ""
        assert strip_markdown_tables("Inception (2010) is in the dataset.") == (
            "Inception (2010) is in the dataset."
        )


class TestRecordCardGuard:
    """A detail card is drawn only for something with details (R-060, R-061).

    `fuzzy_movie_search` used to return its match under a `record` key, but a match is a
    *reference* -- id, title, year. The renderer drew a card for it anyway, so "Tell me
    about Intersteler" produced two Interstellar cards: an empty one from the fuzzy
    match, then the real one from `movie_details`.
    """

    def test_a_bare_reference_is_not_a_record(self) -> None:
        from movieagent.ui.components import is_record

        assert not is_record({"movie_id": 157336, "title": "Interstellar", "year": 2014})

    def test_a_full_record_is(self) -> None:
        from movieagent.ui.components import is_record

        assert is_record(
            {
                "id": 157336,
                "title": "Interstellar",
                "release_year": 2014,
                "release_date": "2014-11-05",
                "genres": ["Adventure", "Drama", "Science Fiction"],
                "director": ["Christopher Nolan"],
                "vote_average": 8.1,
            }
        )

    def test_junk_is_not_a_record(self) -> None:
        from movieagent.ui.components import is_record

        assert not is_record(None)
        assert not is_record({})
        assert not is_record("Interstellar")
