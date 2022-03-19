import argparse
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('-t', '--text', type=str, help='Text to summarize')
    return parser.parse_args()


args = parse_args

tokenizer = AutoTokenizer.from_pretrained("nsi319/legal-pegasus")
model = AutoModelForSeq2SeqLM.from_pretrained("nsi319/legal-pegasus")

input_tokenized = tokenizer.encode(
    args.text, return_tensors='pt', max_length=1024, truncation=True
)
summary_ids = model.generate(input_tokenized,
                             num_beams=9,
                             no_repeat_ngram_size=3,
                             length_penalty=2.0,
                             min_length=150,
                             max_length=250,
                             early_stopping=True)
summary = [tokenizer.decode(g, skip_special_tokens=True,
                            clean_up_tokenization_spaces=False) for g in summary_ids][0]

print(summary)
