# FilingAnalyzer-Backend
Inter IIT Tech Meet Digital Alpha Challenge IIT Kanpur Backend Repo

### Metrics Extraction
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
