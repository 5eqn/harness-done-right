# 产品介绍

生成符合 Storytelling 原则的产品介绍，由 LLM 自动验证。这里以介绍 `hdr` 本身为例。

## Features

- [x] 在 `define_concept.py` 使用元合约创建合理的合约，要求生成可通过 LLM 验证的正反例
- [x] 由元合约自动生成合约 `concept.py`
- [x] 在 `finish_concept.py` 实现合约