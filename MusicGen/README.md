## Music Gen
Has all the files for the musicgen transformer model. 

### Dependencies

I recommend you make a `venv` and run this within that.

Install the dependencies
`pip install -r requirements.txt`

### Files

ModifyingMusicGen --> the demo repository provided by the developers of musicgen.

small_model --> uses `musicgen-stereo-small` and finetunes it on our data.

generate --> uses a provided `.lm` file to generate data. 
