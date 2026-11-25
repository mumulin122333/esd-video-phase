from esdvideo.contracts import parse_prediction, prompt


def test_prompt_mentions_duration_and_phases():
    p = prompt(32)
    assert p.startswith("<video>\nSegment this 32-second ESD clip") and "Hemostasis" in p


def test_parse_two_segments_and_snaps_boundary():
    q = parse_prediction('```json\n{"segments":[{"phase":"Diagnosis","start_seconds":0,"end_seconds":20},'
                         '{"phase":"Local injection","start_seconds":20.5,"end_seconds":32}]}\n```', 32)
    assert q.valid and [s.phase for s in q.segments] == ["Diagnosis", "Local injection"]
    assert q.segments[0].end_seconds == q.segments[1].start_seconds == 20.25


def test_parse_rejects_regression_and_unknown_phase():
    assert not parse_prediction('{"segments":[{"phase":"Hemostasis","start_seconds":0,"end_seconds":10},'
                                '{"phase":"Diagnosis","start_seconds":10,"end_seconds":32}]}', 32).valid
    assert not parse_prediction('{"segments":[{"phase":"Marking","start_seconds":0,"end_seconds":32}]}', 32).valid
    assert not parse_prediction("no json here", 32).valid
