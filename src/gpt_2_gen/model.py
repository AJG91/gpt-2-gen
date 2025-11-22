import torch as tc
from transformers import (
    AutoModelForCausalLM, AutoTokenizer, 
    PreTrainedTokenizerBase, PreTrainedModel,
    DataCollatorForLanguageModeling
)
from transformers.generation.utils import GenerateOutput
from datasets import Dataset, DatasetDict
from typing import Union
from gpt_2_gen.utils import get_device, to_device

class CausalTextGeneration():
    """
    A class for loading and managing a tokenizer and decoder model.

    This class contains all the functionality necessary to load
    a decoder model, as well as functions for loading and applying
    a tokenizer and a data collator.
        
    Attributes
    ----------
    model_name : str
        Name of model that will be loaded.
    device : tc.device
        Device that tensors will be moved to.
    tokenizer : PreTrainedTokenizerBase
        The tokenizer instance loaded from the pretrained model.

    Parameters
    ----------
    model_name : str
        Name of model that will be loaded.
    """
    def __init__(
        self, 
        model_name: str
    ):
        self.model_name = model_name
        self.device = get_device()
        self.tokenizer: PreTrainedTokenizerBase = AutoTokenizer.from_pretrained(model_name)
        self.tokenizer.pad_token = self.tokenizer.eos_token

    def clean_data(
        self,
        dataset: Union[Dataset, DatasetDict],
        cutoff: int = 2000
        ) -> Union[Dataset, DatasetDict]:
        """
        Cleans dataset by removing empty/whitespace and keeps only
        samples where the length is less than cutoff.

        Parameters
        ----------
        dataset : datasets.Dataset or datasets.DatasetDict
            The dataset or dataset dictionary whose `"text"` column will be cleaned.
        cutoff : int, optional (default=2000)
            Filters out samples whose length is greater than value.
    
        Returns
        -------
        datasets.Dataset or datasets.DatasetDict
            Cleaned and filtered dataset.
        """
        dataset = dataset.filter(lambda x: len(x["text"].strip()) > 0)
        dataset = dataset.filter(lambda x: len(x["text"]) < cutoff)
        return dataset

    def tokenize_text(
        self, 
        prompt: str, 
        padding: Union[bool, str] = True, 
        truncation: Union[bool, str] = True
    ) -> dict[str, tc.Tensor]:
        """
        Tokenizes a text prompt (or list of prompts) and moves the 
        resulting tensors to the target device.

        Parameters
        ----------
        prompt : str or list[str]
            The input text or list of texts to tokenize.
        padding : bool or str, optional (default=True)
            Denotes the padding technique to use.
            If True, pad to the longest sequence in the batch.
            If False, does not pad.
        truncation : bool or str, optional (default=True)
            Denotes the truncation technique to use.
            If True, truncates to the model's maximum length.
            If False, does not truncate.
    
        Returns
        -------
        dict[str, tc.Tensor]
            A mapping from input field names (e.g. `'input_ids'`, `'attention_mask'`) 
            to PyTorch tensors located on `self.device`.
        """
        inputs = self.tokenizer(
            prompt,
            padding=padding,
            truncation=truncation,
            return_tensors="pt"
        )
        return to_device(inputs, self.device)
    
    def tokenize_dataset(
        self, 
        dataset: Union[Dataset, DatasetDict], 
        max_token_length: int
    ) -> Union[Dataset, DatasetDict]:
        """
        Tokenizes the text column of a Hugging Face dataset using the model's tokenizer.
        
        Parameters
        ----------
        dataset : datasets.Dataset or datasets.DatasetDict
            The dataset or dataset dictionary whose `"text"` column will be tokenized.
        padding : bool or str, optional (default=True)
            Denotes the padding technique to use.
            If True, pad to the longest sequence in the batch.
            If False, does not pad.
        truncation : bool or str, optional (default=True)
            Denotes the truncation technique to use.
            If True, truncates to the model's maximum length.
            If False, does not truncate.
    
        Returns
        -------
        datasets.Dataset or datasets.DatasetDict
            A new dataset with tokenized columns (e.g., `'input_ids'`, `'attention_mask'`, `'labels'`) and without the original `"text"` or `'label'` column.
        """
        def tokenize_fn(batch):
            tokenized = self.tokenizer(
                batch["text"],
                padding=False,
                truncation=True,
                max_length=max_token_length
            )
            return tokenized

        data_tokenized = dataset.map(
            tokenize_fn,
            batched=True,
            remove_columns=["text", "label"],
        )
        return data_tokenized
    
    def data_collator(
        self, 
        mlm: bool = False
    ) -> DataCollatorForLanguageModeling:
        """
        Creates a data collator for dynamic padding during batching.

        Parameters
        ----------
        mlm : bool, optional (default=False)
            Randomly masks tokens in input. Model then tries to predict masked tokens.
            If fine-tuning encoder, mlm=True since encoder predicts masked tokens.
            If fine-tuning decoder, mlm=False since the decoder predict the next token.

        Returns
        -------
        DataCollatorWithPadding
            A data collator that uses the current tokenizer to dynamically 
            pad input sequences to the length of the longest example in each batch.
        """
        data_collator = DataCollatorForLanguageModeling(
            tokenizer=self.tokenizer,
            mlm=mlm
        )
        return data_collator

    def load_model(
        self,
    ) -> PreTrainedModel:
        """
        Loads a pretrained sequence classification model and 
        moves it to the target device.
    
        Returns
        -------
        PreTrainedModel
            A sequence classification model instance located on `self.device`.
        """
        model = AutoModelForCausalLM.from_pretrained(
            self.model_name,
        ).to(self.device)
        model.config.pad_token_id = self.tokenizer.pad_token_id
        return model
    
    def evaluate_model(
        self,
        prompt: Union[list[str], str],
        model: PreTrainedModel,
        max_new_tokens: int = 25,
        do_sample: bool = True,
        top_k: int = 25,
        top_p: float = 0.95,
        temperature: float = 0.7,
        repetition_penalty: Union[int, float] = 1.1
    ) -> Union[tc.Tensor, GenerateOutput]:
        """
        Tokenizes an input prompt, puts the model in evaluation model, and generates 
        text based on the given prompt.
        
        Parameters
        ----------
        prompt : Union[list[str], str]
            Can be a list of prompts or a single prompt.
        model : PreTrainedModel
            A causal language model instance.
        max_new_tokens : int, optional(default=25)
            Total amount of tokens that will be used in the text generation.
        do_sample : bool, optional (default=True)
            Tells model to sample from probability distribution instead of 
            picking the highest-probability token.
            If False, deterministic.
            If True, random sampling.
        top_k : int, optional (default=25)
            Sorts logits, keeps top N most probably tokens, and samples from those.
            Lower k -> more deterministic.
            Higher k -> more random.
        top_p : float, optional (default=0.95)
            Keeps the smallest set of tokens whose cumulative probability >= N.
        temperature : float, optional (default=0.7)
            Denotes the fraction of temperature that will be used when sampling.
            Lower T -> more deterministic.
            Higher T -> more random.
        repetition_penalty : Union[int, float], optional (default=1.1)
            Specifies how much penalty will be added to the model when repeating text.

        Returns
        -------
        Union[tc.Tensor, GenerateOutput]
            Tokens representing the generated text from the model.
        """
        inputs = self.tokenize_text(prompt)

        model.eval()
        with tc.no_grad():
            outputs = model.generate(
                **inputs, 
                max_new_tokens=max_new_tokens,
                do_sample=do_sample,
                top_k=top_k,
                top_p=top_p,
                temperature=temperature,
                repetition_penalty=repetition_penalty,
                pad_token_id=self.tokenizer.pad_token_id
            )
        return outputs