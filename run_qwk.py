from compute_qwk import load_human_severity, compute_qwk_report

my_scores = load_human_severity(
    "/Users/busola/Documents/projects/trustworthy-asr-pipeline/writeup_results/candidate_pool.json"
)
yishun_scores = load_human_severity(
    "/Users/busola/Documents/projects/trustworthy-asr-pipeline/writeup_results/candidate_pool_new_Yishun.json"
)
judge_scores = load_human_severity(
    "/Users/busola/Documents/projects/trustworthy-asr-pipeline/writeup_results/judge_calibration/phi4_direct.json",
    severity_key="severity"
)

compute_qwk_report(my_scores, yishun_scores, judge_scores)
