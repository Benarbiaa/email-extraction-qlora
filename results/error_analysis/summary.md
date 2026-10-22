# Error analysis (test_provisional.jsonl, provisional labels)

## Base zero-shot

- records with at least one wrong scored field: 161/200
- out-of-schema outputs: 17

Out-of-schema values:

- `work_mode: 'null'` ×2
- `email_type: 'final_interview'` ×2
- `sender_role: 'recruiting_team'` ×2
- `sender_role: 'engineering_manager'` ×1
- `sender_role: 'talent_partner'` ×1
- `stage: 'null'` ×1
- `stage: 'onboarding'` ×1
- `sender_role: 'tech_lead'` ×1

| field | gold → predicted | count |
|---|---|---|
| deadline | wrong value | 40 |
| next_step | value -> null | 20 |
| contract_type | internship -> None | 20 |
| sender_role | automated_system -> recruiter | 17 |
| email_type | scheduling -> interview_invite | 13 |
| job_title | value -> null | 12 |
| sender_role | hiring_manager -> recruiter | 11 |
| email_type | info_request -> application_ack | 10 |
| work_mode | None -> remote | 10 |
| stage | None -> screening | 9 |
| stage | technical_interview -> screening | 9 |
| contract_type | apprenticeship -> None | 8 |
| work_mode | hybrid -> None | 7 |
| job_title | null -> value | 7 |
| email_type | rejection -> application_ack | 7 |

## Base few-shot

- records with at least one wrong scored field: 155/200
- out-of-schema outputs: 15

Out-of-schema values:

- `sender_role: 'engineering_manager'` ×2
- `email_type: 'final_interview'` ×2
- `deadline: None was expected` ×1
- `sender_role: 'talent_partner'` ×1
- `sender_role: 'talent_acquisition'` ×1
- `sender_role: 'tech_lead'` ×1
- `sender_role: 'developer'` ×1
- `sender_role: 'community_events'` ×1

| field | gold → predicted | count |
|---|---|---|
| deadline | wrong value | 39 |
| job_title | null -> value | 38 |
| next_step | value -> null | 13 |
| email_type | scheduling -> interview_invite | 12 |
| work_mode | None -> remote | 11 |
| contract_type | None -> CDI | 9 |
| sender_role | automated_system -> HR | 9 |
| email_type | other -> info_request | 7 |
| contract_type | internship -> None | 7 |
| stage | applied -> screening | 7 |
| email_type | rejection -> application_ack | 7 |
| stage | closed -> applied | 7 |
| contract_type | apprenticeship -> None | 6 |
| job_title | wrong value | 6 |
| sender_role | recruiter -> HR | 6 |

## Fine-tuned (QLoRA v1)

- records with at least one wrong scored field: 33/200
- out-of-schema outputs: 3

Out-of-schema values:

- `sender_role: 'peer'` ×1
- `email_type: 'offer_ack'` ×1
- `deadline: None was expected` ×1

| field | gold → predicted | count |
|---|---|---|
| next_step | null -> value | 9 |
| stage | screening -> technical_interview | 6 |
| job_title | wrong value | 2 |
| deadline | wrong value | 2 |
| action_required | False -> True | 2 |
| job_title | null -> value | 2 |
| action_required | True -> False | 2 |
| deadline | value -> null | 1 |
| next_step | value -> null | 1 |
| stage | applied -> screening | 1 |
| sender_role | automated_system -> other | 1 |
| email_type | other -> scheduling | 1 |
| stage | technical_interview -> screening | 1 |
| contract_type | apprenticeship -> internship | 1 |
| sender_role | hiring_manager -> recruiter | 1 |
