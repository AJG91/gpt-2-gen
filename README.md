# GPT-Style Language Modeling

[my-website]: https://AJG91.github.io "my-website"

This repository showcases how language models such as GPT-2/3 generate text. In the process, you will learn how to fine-tune such models.

## Getting Started

* This project relies on `python=3.12`. It was not tested with different versions
* Clone the repository to your local machine
* Once you have, `cd` into this repo and create the virtual environment (assuming you have `conda` installed) via
```bash
conda env create -f environment.yml
```
* Enter the virtual environment with `conda activate gpt-2-gen-env`
* Install the packages in the repo root directory using `pip install -e .` (you only need the `-e` option if you intend to edit the source code in `gpt_2_gen/`)

## Example

See [my website][my-website] for examples on how to use this code.

## Citation

If you use this project, please use the citation information provided by GitHub via the **“Cite this repository”** button or cite it as follows:

```bibtex
@software{gpt_2_gen_2025,
  author = {Alberto Garcia},
  title = {GPT-Style Language Model},
  year = {2025},
  publisher = {GitHub},
  url = {https://github.com/AJG91/gpt-2-gen},
  license = {MIT}
}
```