---
title: API Usage Guide
description: Complete guide for using the Prior Labs TabPFN API
---

# TabPFN API Guide

## Authentication

### Interactive Login

The first time you use TabPFN, you'll be guided through an interactive login process:

```python
from tabpfn_client import init
init()
```

### Managing Access Tokens

You can save your token for use on other machines:

```python
import tabpfn_client
# Get your token
token = tabpfn_client.get_access_token()

# Use token on another machine
tabpfn_client.set_access_token(token)
```

## Rate Limits

Our API implements a fair usage system that resets daily at 00:00:00 UTC.

### Usage Cost Calculation

The cost for each API request is calculated as:
```python
api_cost = (num_train_rows + num_test_rows) * num_cols * n_estimators
```

Where `n_estimators` is by default 4 for classification tasks and 8 for regression tasks.

### Monitoring Usage

Track your API usage through response headers:

| Header | Description |
|--------|-------------|
| `X-RateLimit-Limit` | Your total allowed usage |
| `X-RateLimit-Remaining` | Remaining usage |
| `X-RateLimit-Reset` | Reset timestamp (UTC) |

## Current Limitations

!!! warning "Important Data Guidelines"
    - Do NOT upload any Personally Identifiable Information (PII)
    - Do NOT upload any sensitive or confidential data
    - Do NOT upload any data you don't have permission to share
    - Consider anonymizing or pseudonymizing your data
    - Review your organization's data sharing policies

### Size Limitations

1. Maximum total cells per request must be below 100,000:
```
(num_train_rows + num_test_rows) * num_cols < 100,000
```

2. For regression with full output turned on (`return_full_output=True`), the number of test samples must be below 500.

These limits will be relaxed in future releases.

### Managing User Data

You can access and manage your personal information:

```python
from tabpfn_client import UserDataClient
print(UserDataClient.get_data_summary())
```

## Error Handling

The API uses standard HTTP status codes:

| Code | Meaning |
|------|----------|
| 200 | Success |
| 400 | Invalid request |
| 429 | Rate limit exceeded |

Example response, when limit reached:
```json
{
    "error": "API_LIMIT_REACHED",
    "message": "Usage limit exceeded",
    "next_available_at": "2024-01-07 00:00:00"
}
```
