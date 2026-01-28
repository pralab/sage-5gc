<img align="right" width="250" src="assets/logo.png">

# SAGE-5GC: Security-Aware Guidelines for Evaluating Anomaly Detection in the 5G Core Network

This repository contains the artifacts for the paper "SAGE-5GC: Security-Aware Guidelines for Evaluating Anomaly Detection in the 5G Core Network", accepted to ITASEC 26.

## Abstract

Machine learning-based anomaly detection systems are increasingly being adopted in 5G Core networks to monitor complex, high-volume traffic. However, most existing approaches are evaluated under strong assumptions that rarely hold in operational environments, notably the availability of independent and identically distributed (IID) data and the absence of adaptive attackers. In this work, we study the problem of detecting 5G attacks in the wild, focusing on realistic deployment settings. We propose a set of Security-Aware Guidelines for Evaluating anomaly detectors in 5G Core Network (SAGE-5GC), driven by domain knowledge and consideration of potential adversarial threats. Using a realistic 5G Core dataset, we first train several anomaly detectors and assess their baseline performance against standard 5GC control-plane cyberattacks targeting PFCP-based network services. We then extend the evaluation to adversarial settings, where an attacker tries to manipulate the observable features of the network traffic to evade detection, under the constraint that the intended functionality of the malicious traffic is preserved. Starting from a selected set of controllable features, we analyze model sensitivity and adversarial robustness through randomized perturbations. Finally, we introduce a practical optimization strategy based on genetic algorithms that operates exclusively on attacker-controllable features and does not require prior knowledge of the underlying detection model. Our experimental results show that adversarially crafted attacks can substantially degrade detection performance, underscoring the need for robust, security-awareevaluation methodologies for anomaly detection in 5G networks deployed in the wild.

## File structure

- [`data`](/data/) - contains datasets and trained models.
- [`reproducibility`](/reproducibility/README.md) - Contains scripts for training, evaluating models, and generating plots.
- [`attacks`](/attacks/README.md) - Contains implementations of various adversarial attack strategies.
- [`results`](/results/) - Contains results from experiments and generated figures.

## Getting Started

To run a specific module, navigate the corrisponding folder (e.g `attack`) and follow the steps in its `README.md`.

## How to cite us

To cite our work, please use the following BibTeX entry:

```bibtex
```

## Acknowledgements

This work was partially supported by the EU-funded project [Sec4AI4Sec - Cybersecurity for AI-Augmented Systems](https://www.sec4ai4sec-project.eu) (grant no. 101120393); and by the project [SERICS](https://serics.eu/) (PE00000014) (PE00000014) under the MUR NRRP funded by EU-NextGenEU.

<img src="assets/sec4AI4sec.png" alt="sec4ai4sec" style="width:100px;"/> &nbsp;&nbsp;
<img src="assets/SERICS.png" alt="serics" style="width:330px;"/> &nbsp;&nbsp;
<img src="assets/FundedbytheEU.png" alt="LInf" style="width:300px;"/>