import torch
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
from datasets import load_from_disk
import pandas as pd
from nltk.translate.meteor_score import meteor_score
from textSummarizer.entity import ModelEvaluationConfig




class ModelEvaluation:
    def __init__(self, config: ModelEvaluationConfig):
        self.config = config

    def generate_batch_sized_chunks(self, list_of_elements, batch_size):
        """Split the dataset into smaller batches that we can process simultaneously.
        Yield successive batch-sized chunks from list_of_elements."""
        for i in range(0, len(list_of_elements), batch_size):
            yield list_of_elements[i : i + batch_size]

    def calculate_metric_on_test_ds(self, dataset, model, tokenizer, 
                                    batch_size=16, device="cuda" if torch.cuda.is_available() else "cpu", 
                                    column_text="article", 
                                    column_summary="highlights"):
        article_batches = list(self.generate_batch_sized_chunks(dataset[column_text], batch_size))
        target_batches = list(self.generate_batch_sized_chunks(dataset[column_summary], batch_size))

        scores = []

        for article_batch, target_batch in tqdm(
            zip(article_batches, target_batches), total=len(article_batches)):
            
            inputs = tokenizer(article_batch, max_length=1024, truncation=True, 
                               padding="max_length", return_tensors="pt")
            
            summaries = model.generate(input_ids=inputs["input_ids"].to(device),
                                       attention_mask=inputs["attention_mask"].to(device), 
                                       length_penalty=0.8, num_beams=8, max_length=128)
            # Length penalty ensures that the model does not generate sequences that are too long.
            
            # Decode the generated texts
            decoded_summaries = [tokenizer.decode(s, skip_special_tokens=True, 
                                                  clean_up_tokenization_spaces=True) 
                                 for s in summaries]
            
            for ref, pred in zip(target_batch, decoded_summaries):
                # METEOR score expects reference and hypothesis as lists of tokens
                reference = ref.split()
                prediction = pred.split()
                score = meteor_score([reference], prediction)
                scores.append(score)
            
        average_score = sum(scores) / len(scores)
        return average_score

    def evaluate(self):
        device = "cuda" if torch.cuda.is_available() else "cpu"
        tokenizer = AutoTokenizer.from_pretrained(self.config.tokenizer_path)
        model_pegasus = AutoModelForSeq2SeqLM.from_pretrained(self.config.model_path).to(device)
       
        # Loading data 
        dataset_samsum_pt = load_from_disk(self.config.data_path)

        score = self.calculate_metric_on_test_ds(
            dataset_samsum_pt['test'][0:10], model_pegasus, tokenizer, 
            batch_size=2, column_text='dialogue', column_summary='summary'
        )

        df = pd.DataFrame({'meteor': [score]}, index=['pegasus'])
        df.to_csv(self.config.metric_file_name, index=False)

