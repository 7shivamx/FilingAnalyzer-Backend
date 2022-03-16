# FilingAnalyzer-Backend
Inter IIT Tech Meet Digital Alpha Challenge IIT Kanpur Backend Repo

## API endpoints

### Company By Name API
```
url: /companybyname
method: POST

Successful: 200
Parameters: {
                "name": <name of the company>
            }

Response: {
            "cik": <cik of the company>, 
            "ticker": <ticker of the company>
          }
```

### Company By Cik API
```
url: /companybycik
method: POST

Successful: 200
Parameters: {
                "cik": <cik of the company>
            }

Response: {
            "name": <name of the company>, 
            "ticker": <ticker of the company>
          }
```

### Company By Ticker API
```
url: /companybyticker
method: POST

Successful: 200
Parameters: {
                "ticker": <ticker of the company>
            }

Response: {
            "name": <name of the company>, 
            "cik": <cik of the company>
          }
```

### Metrics Extraction API
```
url: /extract
method: POST

Successful: 200
Parameters: {
                "cik": <cik of the company>,
                "timeperiod": <annual (10-K forms)/quaterly (10-Q forms)>,
                "from_date": <date of the format yyyy-mm-dd>
                "to_date": <date of the format yyyy-mm-dd>
                "metric": <one from the list: ["churn rate", "revenue retention", "LTV to CAC ratio", "Customer Engagement Score",
"Recurring Revenue", "SAAS Quick Ratio", "SAAS Magic Number"]>,              
            }
            
Response: {
            "correct_value": [(value, date_of_doc_filling), ... ]
            <This will give the respective metric value (or -1 if not found) along with the filing date of the doc>
          }
```

### Dict-Sentiment Analysis API
```
url: /dictsent
method: POST

Successful: 200
Parameters: {   
                "q": <Text String to check Sentiment>              
            }
            
Response: {
           'Constraining',
           'Negative',
           'Uncertainty',
           'Litigious',
           'Weak_Modal',
           'Positive'
          }
```

### Bert-Sentiment Analysis API
```
url: /bertinf
method: POST

Successful: 200
Parameters: {   
                "q": <Text String to check Sentiment>              
            }
            
Response: {
           'negative',
           'neutral',
           'positive'
          }
```
