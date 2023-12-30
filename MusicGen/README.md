## Music Gen
Has all the files for the musicgen transformer model. 

### Dependencies

I recommend you make a `venv` and run this within that.

Install the dependencies
`pip install -r requirements.txt`

### Training methods

Ways to train this model are either

1. Using dora: https://github.com/lyramakesmusic/finetune-musicgen/blob/main/README.md
2. the pytorch stuff: https://github.com/neverix/musicgen_trainer/tree/main

### Files

AudioBreakdown --> converts the large 5 hour files into 30 second files

ModifyingMusicGen --> the demo repository provided by the developers of musicgen.

small_model --> uses `musicgen-stereo-small` and finetunes it on our data.

generate --> uses a provided `.lm` file to generate data. 
