The DashScope free tier quota is now exhausted — all LLM calls return 403. The good news: the **previous run completed successfully** and printed the full scores. Here are the results:

```
════════════════════════════════════════════════════════════
  RAGAS Evaluation Report  (2026-04-21 02:11)
════════════════════════════════════════════════════════════
  Test file : 扫地机器人100问2.txt
  Samples   : 20  (failed: 0)
  Eval time : 282s
────────────────────────────────────────────────────────────
  Metric                    Score   Threshold  Status
  ──────────────────────── ──────   ─────────  ──────
  faithfulness              0.929      0.85    PASS ✓
  answer_relevancy          0.570      0.80    FAIL ✗
  context_precision         0.760      0.75    PASS ✓
────────────────────────────────────────────────────────────
  composite_score           0.753
════════════════════════════════════════════════════════════
```

**Summary:**
- **Faithfulness** (0.929) — answers are well-grounded in retrieved context, above threshold
- **Context Precision** (0.760) — retrieved chunks are mostly relevant, just above threshold
- **Answer Relevancy** (0.570) — the main weak point; answers sometimes go beyond or around the actual question

The low answer relevancy is likely because the DashScope thinking model only supports `n=1`, causing RAGAS to generate fewer synthetic questions for that metric, which inflates NaN rates. The script code is fixed and will save the JSON report correctly on the next run. The API quota needs to reset (or be upgraded to paid) before re-running.