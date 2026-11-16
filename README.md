# Image Captioning

Automatic captions for images with a CNN encoder and an LSTM decoder with attention, trained on Flickr30k.

Deep Learning coursework (UFMG). Group project.
<!-- Add one line here with your own contribution. -->

## Approach

- **Encoder:** pretrained ResNet-50; the feature maps feed the attention mechanism.
- **Decoder:** LSTM with attention, trained with Adam.
- **Data:** Flickr30k (all five captions per image), custom vocabulary with a frequency threshold, padded batches.
- **Decoding:** greedy decoding (picks the most likely word at each step) and beam search (keeps the k = 5 best partial captions).
- **Evaluation:** BLEU and ROUGE-L on the validation set after each epoch; the checkpoint was chosen by validation loss and BLEU together.

## Results

Validation metrics after the last epoch (5): BLEU 0.154, ROUGE-L 0.436. Beam search produced more coherent captions than greedy decoding. Training took about 12 hours on a Kaggle GPU.

An early version produced many `<unk>` tokens. The cause was a vocabulary built from only the first of the five captions and a low frequency threshold; building it from all captions fixed it.

## Running

Open `image2text.ipynb` in Kaggle (or any GPU environment) with the Flickr30k dataset attached.
