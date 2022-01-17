import logging
from celery import Task
from transformers import PegasusForConditionalGeneration, PegasusTokenizer
from .worker import app


class PredictTask(Task):
    """
    Abstraction of Celery's Task class to support loading ML model.
    """
    abstract = True

    def __init__(self):
        super().__init__()
        self.model = None
        self.tokenizer = None

    def __call__(self, *args, **kwargs):
        """
        Load model on first call (i.e. first task processed)
        Avoids the need to load model on each task request
        """
        if not self.model:
            logging.info('Loading Model...')
            self.model = PegasusForConditionalGeneration.from_pretrained('google/pegasus-large')
            self.tokenizer = PegasusTokenizer.from_pretrained('google/pegasus-large') 
            logging.info('Model loaded')
        return self.run(*args, **kwargs)


@app.task(ignore_result=False,
          bind=True,
          base=PredictTask, # override default Celery Task to initialize the machine learning model at the start
          path=('celery_task_app.ml.model', 'TextGenerationModel'),
          name='{}.{}'.format(__name__, 'TextAnalytics'))
def predict_text_analytics_single(self, inp_texts):
    if isinstance(inp_texts, str):
        inp_texts = [inp_texts]
    batch = self.tokenizer.prepare_seq2seq_batch(src_texts=inp_texts, max_length=1024, truncation=True, padding='longest', return_tensors='pt')
    input_ids = batch['input_ids']#.to(device)
    attention_mask = batch['attention_mask']#.to(device)
    generated_ids = self.model.generate(input_ids=input_ids, attention_mask=attention_mask, \
    use_cache=True, num_beams=64, top_p=0.8, top_k=50, num_return_sequences=10, temperature=1.5, no_repeat_ngram_size=2, \
    repetition_penalty=2.0, early_stopping=True)
    result = self.tokenizer.batch_decode(generated_ids.tolist(), skip_special_tokens=True)
    return result