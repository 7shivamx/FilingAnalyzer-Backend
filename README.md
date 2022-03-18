# FilingAnalyzer-Backend
Inter IIT Tech Meet Digital Alpha Challenge IIT Kanpur Backend Repo

## How to host

1. On the terminal, `cd` into the project folder and run `./setup.sh` to create virtual environment and install required dependencies.

2. Now execute `./start.sh` to host the server.

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
            "ticker": <ticker of the company>,
            "ARR": <latest ARR from db>,
            "NRR": <latest NRR/NDR from db>,
            "Customers": <total customers of the company>
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
            "ticker": <ticker of the company>,
            "ARR": <latest ARR from db>,
            "NRR": <latest NRR/NDR from db>,
            "Customers": <total customers of the company>
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
            "cik": <cik of the company>,
            "ARR": <latest ARR from db>,
            "NRR": <latest NRR/NDR from db>,
            "Customers": <total customers of the company>
          }
```

### Timeseries By Ticker API
```
url: /tsbyticker
method: POST

Successful: 200
Parameters: {
                "ticker": <ticker of the company>
            }

Response: {
            "arrTS": <ARR timeseries>, 
            "nrrTS": <NRR timeseries>,
            "custTS": <customer timeseries>,
            "smTS": <sales and marketing expense timeseries>,
            "empTS": <employee timeseries>,
            "quarTS": <total customers of the company>,
            "srcTS": <source filings for data>            
          }
```

### Trending words By Ticker API
```
url: /twitbyticker
method: POST

Successful: 200
Parameters: {
                "ticker": <ticker of the company>
            }

Response: {
            "trendingWords": <trending words at twitter>            
          }
```

### QnA and Summary By Ticker API
```
url: /qnabyticker
method: POST

Successful: 200
Parameters: {
                "ticker": <ticker of the company>
            }

Response: {
            "qna": <array of dictionaries with keys 'question' and 'answer'> 
            "summary": <text summary>            
          }
```

### Sentiments By Ticker API
```
url: /sentibyticker
method: POST

Successful: 200
Parameters: {
                "ticker": <ticker of the company>
            }

Response: {
            "dictSenti": <Dictionary with probabilities of 6-7 classes according to the found occurrence> 
            "finbSenti": <Dictionary with probabilities of 'positive', 'neutral' & 'negative'>            
          }
```

### Sectionwise Data of last 10k By Ticker API
```
url: /secbyticker
method: POST

Successful: 200
Parameters: {
                "ticker": <ticker of the company>
            }

Response: {
            "sectionwise": <Dictionary of items 1-16 along with 1a,1b,7a,9a,9b>            
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
            "correct_value": [(value, date_of_doc_filling, condition), ... ]
            <This will give the respective metric value (or -1 if not found) along with the filing date of the doc>
          }
```


### Company Overview API
```
url: /overviewbyticker
method: POST

Successful: 200
Parameters: {   
                "ticker": <ticker of the company>              
            }
            
Response: {
            'description',
            'exchange',
            'quater',
            'pe',
            'divi',
            'eps',
            'profitmargin',
            'operatingmarginttm'
          }
```

### Metrics Timeseries API
```
url: /incometimeseries
method: POST

Successful: 200
Parameters: {   
                "ticker": <ticker of the company>,
                "timeperiod": <annual (10-K forms)/quaterly (10-Q forms)>             
            }
            
Response: {
           "data": [(date, opex, gpm, condition), ... ]
          }
```

### EPS Timeseries API
```
url: /earningstimeseries
method: POST

Successful: 200
Parameters: {   
                "ticker": <ticker of the company>,
                "timeperiod": <annual (10-K forms)/quaterly (10-Q forms)>             
            }
            
Response: {
            "data": [(date, eps), ... ]
          }
```
