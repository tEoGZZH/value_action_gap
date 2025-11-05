# Value Alignment in LLMs
<!-- # value_action_data -->

This repository includes codes for two papers presented in [EMNLP 2025](https://2025.emnlp.org/) regarding **value alignment in LLMs**:

- [EMNLP 2025 Main](https://arxiv.org/pdf/2501.15463?): [Mind the Value-Action Gap: Do LLMs Act in Alignment with Their Values?](https://arxiv.org/pdf/2501.15463?)
- [EMNLP 2025 WiNLP Workshop](https://arxiv.org/pdf/2409.09586): [ValueCompass: A Framework for Measuring Contextual Value Alignment Between Human and LLMs](https://arxiv.org/pdf/2409.09586)

These two papers aim to answer the core concerning questions related to value alignment:
- **RQ1**: *How can we systematically capture human values and evaluate the extent to which LLM aligns with them?* (Value Alignment between **Humans & LLMs**, [EMNLP 2025 WiNLP Workshop](https://arxiv.org/pdf/2409.09586))
- **RQ2**: *To what extent do LLM-generated value statements align with their value-informed actions?* (Value Alignment between LLM's **value claim & the corresponding actions**, [EMNLP 2025 Main](https://arxiv.org/pdf/2501.15463?))

## Overview


To address **RQ1** -- *Human-AI Value Alignment* -- we propose **ValueCompass**, a framework for systematically measuring value alignment between LLMs and humans across contextual scenarios. See below figure for an overview. 

<img src="figures/figure-valuecompass.png" width="66%" alt="Description">




To address **RQ2** -- *LLM's Value-Action Alignment* -- we propose **ValueActionLens** Framework, associated with the VIA (Value-Informed Dataset) dataset, to assess the alignment between LLMs’ stated values & value-informed actions. See below figure for an example of GPT4o's Value-Action Gap. 

<img src="figures/figure-value-action-gap.png" width="66%" alt="Description">



## Evaluating Value Alignment

To evaluate the value alignment in LLMs, codes are released in this directory: [Value-Action Alignment Tasks](https://github.com/huashen218/value_action_gap/tree/main/src/tasks).




## VIA Dataset

The full VIA dataset can be accessed in this directory: [Value-Informed Dataset (VIA)](https://github.com/huashen218/value_action_gap/tree/main/outputs/data_release)



```bibtex
@article{shen2024valuecompass,
    title={Valuecompass: A framework of fundamental values for human-ai alignment},
    author={Shen, Hua and Knearem, Tiffany and Ghosh, Reshmi and Yang, Yu-Ju and Mitra, Tanushree and Huang, Yun},
    journal={arXiv preprint arXiv:2409.09586},
    year={2024}
}


@article{shen2025mind,
  title={Mind the Value-Action Gap: Do LLMs Act in Alignment with Their Values?},
  author={Shen, Hua and Clark, Nicholas and Mitra, Tanushree},
  journal={arXiv preprint arXiv:2501.15463},
  year={2025}
}
