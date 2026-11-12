"""The evaluation harness: benchmark file, metrics, intervals, runs, results.

``gold``      read and validate ``data/gold/questions.jsonl``
``metrics``   recall@k, reciprocal rank, nDCG on one ranking
``bootstrap`` confidence intervals and paired comparisons
``runner``    run a retriever over the questions and summarise
``results``   write the result file every published number comes from
``report``    build the Markdown results table from those files
"""
