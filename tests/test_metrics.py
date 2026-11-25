from esdvideo.metrics import phase_report

P = ("Diagnosis", "LocalInjection", "IncisionDissection", "Hemostasis")


def test_case_macro_averages_present_phases_then_cases():
    cases = ["a"] * 4 + ["b"] * 2
    truth = ["Diagnosis", "Diagnosis", "Hemostasis", "Hemostasis", "Diagnosis", "Diagnosis"]
    pred = ["Diagnosis", "Diagnosis", "Diagnosis", "Hemostasis", "Diagnosis", "UNPARSED"]
    r = phase_report(cases, truth, pred, P)
    # case a: Diagnosis F1 = 2*2/(2*2+1+0) = 0.8, Hemostasis F1 = 2*1/(2*1+0+1) = 2/3
    assert abs(r.case_macro_f1["a"] - (0.8 + 2 / 3) / 2) < 1e-9
    # case b: Diagnosis F1 = 2*1/(2*1+0+1) = 2/3; UNPARSED is always wrong
    assert abs(r.case_macro_f1["b"] - 2 / 3) < 1e-9
    assert r.frame_micro_f1 == {"a": 0.75, "b": 0.5}
