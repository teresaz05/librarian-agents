# Data

- BrowseComp+ is available from `Tevatron/browsecomp-plus`. See
  <https://github.com/texttron/BrowseComp-Plus>. BrowseComp+ is distributed
  under the MIT License.
- BRIGHT is by Su et al. and is available from `xlangai/BRIGHT` under
  [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/). The included
  manifest selects the 150-question subset used in the paper. See
  <https://huggingface.co/datasets/xlangai/BRIGHT>.

Obtain BrowseComp+ through its official distribution process, then set
`BROWSECOMP_SOURCE` to the resulting local JSONL file and run
`scripts/prepare_browsecomp_plus.sh` to prepare the local 680/50 split.
Run `scripts/prepare_bright_qa.sh` to prepare the BRIGHT QA evaluation set.
The exact 50-corpus, 443-question BrowseComp+ amortization manifest is stored in
`data/splits/browsecomp_amortized50.jsonl`.

The bundled BrowseComp+ manifests contain evaluation metadata and generated
question-answer records derived from BrowseComp+. The BRIGHT manifest contains
only the selected upstream question identifiers and domains. All benchmark
materials remain subject to their upstream licenses and attribution terms.
