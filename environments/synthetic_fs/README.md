# Synthetic Filesystem Environment

- `dataset.py` builds grouped RL examples from prepared corpora and queries.
- `message_env.py` runs builder trajectories and tool interactions.
- `workspace.py` implements filesystem operations and snapshot serialization.
- `reward.py` computes intermediate answerability and terminal correctness rewards.
- `prompts.py` defines builder and answering prompts.
- `clients.py` provides answerer and judge clients.
- `coverage.py` implements outcome-blind construction coverage checks.
