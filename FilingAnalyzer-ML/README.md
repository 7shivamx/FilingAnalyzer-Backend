# FilingAnalyzer-ML 

1. dict-sentiment.py  -> For sentiment analysis(6 classes - lexicon based)

    - Input: The input of the file is specified in the `text`. 
    - Output: The input is passed to the `get_class_counter` function, which returns the sentiment dictionary, `sent_dict`, containing the results.


2. finbert_inference.py -> For sentiment analysis(3 classes - transformer)

    - Input: The input of the file is specified in the `text`.
    - Output; The input is passed to the `get_output` function, which returns the sentiment dictionary, `sent_dict`.
    
    
3. generate_questions_answers.py -> to generate questions and answers from the text given

    - Input: The only input is the `text`
    - Output: The output of the file is the generated questions and answers in the dictionary `qna_dict`


4. summarize_text.py -> to summarize the text given

    - Input: The only input is the `text`
    - Output: The output of the file is the summary of the text in the variable `summary`   
    
    
5. extract_tables.py -> For extracting tables from the fillings

    - Input: The inputs are `api_key` for accessing the fillings using sec-api, `filing_url` and the `section`
    - Output: The output of the file is the tables extracted from the filing stored in `tables` variable
    
    
6. find_metric.py -> complete pipeline for extracting metrics from filings of a company

    - Input: The inputs are - 
        * `api_key` - for accessing the fillings using sec-api
        * `cik` - cik of the cmopany
        * `metric` - name of the metric in lowercase
        * `startDate` - start date to look for the filing
        * `endDate` - end date to look for the filing
    - Output: The output of the file is `value` of the metric extracted from the filings    
         
