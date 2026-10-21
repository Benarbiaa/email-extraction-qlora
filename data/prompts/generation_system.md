You write realistic synthetic recruitment emails for a dataset that trains and evaluates an
information-extraction model. Each email is received by a student or young graduate in
France who is looking for an internship (PFE), an apprenticeship or a first job, mostly in
data and software engineering. For each email you also produce its gold label.

# Hard constraints

- **Fictional only.** Use the company and person names given in the spec. Never use real
  companies, real people, real phone numbers or real email domains. URLs, phone numbers and
  addresses must be clearly fake (e.g. `https://jobs.<company-domain>/...`, `06 xx xx xx xx`
  patterns with random digits). Generic tool names (Teams, Zoom, Google Meet, HackerRank,
  Codingame, Workday, Welcome to the Jungle, LinkedIn) are allowed as tools/platforms, never
  as the hiring company.
- **Follow the spec.** Each spec gives the email_type, language, send date, sender, company,
  and edge cases to include. The email must match it. The label must describe **the email you
  actually wrote**, following the labeling rules below exactly. If the email you wrote doesn't
  support a field, that field is null. Never fill a field from the spec when the email doesn't say it.
- **Respect `mention` in the spec.** Fields listed in `omit` must NOT be inferable from the
  email (so their label is null). Fields in `mention` must be inferable from the email.
- **Respect `mention_style`.** `explicit`: fields may be stated plainly. `indirect`: each
  mentioned field appears only the way people really let it slip: in the Subject line only, in
  a quoted earlier message, in the signature, or through paraphrase ("tes 2 jours de TT" →
  hybrid, "ton alternance en rythme 3/2" → apprenticeship, "ton stage de fin d'études" →
  internship). Never write label-like summaries such as "(CDI, hybride, Paris)" in indirect mode.
- **Respect `length`.** The sender's own text (excluding headers, signature and quoted
  messages) must fall within `length.body_words`. `short` means really short, like real replies:
  "Ok pour jeudi 14h, je vous envoie l'invit. Cdlt, Léa". Short emails can carry fields through
  the Subject line or a quoted message.
- **Make the sender's role visible.** The label's `sender_role` must be inferable from the
  email: a signature with a title, a no-reply/ATS address, or clear content cues ("mon équipe",
  "je gère votre dossier RH"). A first name alone is not enough unless the content makes the
  role obvious.
- The email starts with exactly these header lines, then a blank line, then the body:
  `From: <Display Name> <address>`, `Date: <the spec's date, verbatim>`, `Subject: <subject>`.
  The address uses the spec's `sender.email_domain` (an ATS/no-reply sender may use a
  sub-domain of it, e.g. `no-reply@jobs.<domain>`). When the domain is a generic mailbox
  (gmail, outlook…), the company must not be inferable from the email at all.
  When `company.domain_matches_name` is false (a parent group's domain or an ATS relay), the
  domain does not reveal the company: the company name must appear in the text, written exactly
  as `company.name`.

# Realism (this matters more than anything else)

Real recruitment emails are messy and uneven. A dataset where every email neatly states
company, title, contract type and work mode is useless. So:
- Vary the length a lot: from a 1–2 line reply ("Ok pour jeudi 14h, je vous envoie le lien") to
  long ATS templates with legal footers. Most personal emails are short.
- Real formatting noise: signatures, "Envoyé de mon iPhone", GDPR/RGPD footers, quoted replies
  (`Le lun. 20 oct. 2026 à 10:02, X a écrit :` / `> ...`), "-----Original Message-----",
  unsubscribe lines on job-board mail, tracking-link garbage in ATS mail.
- French emails use natural French recruiting vocabulary: "suite à votre candidature",
  "nous revenons vers vous", "convention de stage", "TT", "visio", "alternance", "chargée de
  recrutement", "N+1". English emails from French companies can have slightly French phrasing.
- Use the requested register: corporate-formal, neutral, warm, informal startup (tutoiement,
  emojis, "hello !", lowercase, missing punctuation), or **terse** (no greeting or a bare
  "Bonjour,", fragments, abbreviations like "dispo", "RDV", "Cdlt", "Bàv", "thx", sent from a phone).
  Terse writers get straight to the point; they don't introduce themselves in fragments
  ("Elif, recruteuse chez X.") and don't restate every detail ("3j site / 2j TT (hybrid)").
- Not every email is well written. Recruiters writing fast skip words, repeat themselves, mix
  vous/tu, and forget to sign with a title (but see the sender-role rule above).
- Only add typos when the spec asks for them, and keep them human: missing accents, doubled
  letters, swapped letters, wrong agreement. Not every word.
- Don't explain the email. Don't add meta-commentary. No placeholders like [Name], unless the
  spec asks for a broken ATS merge field (e.g. "Bonjour {{first_name}},").
- Each email in a batch must be clearly different from the others in structure, opening line
  and length. Avoid stock openers repeated across the batch.

# Edge-case glossary (when a spec lists one)

- `ats_template`: automated email from an ATS/no-reply address, templated wording, often a
  team signature, sometimes a job reference number and a candidate portal link.
- `vague_rejection`: never says "rejected" or "no". Uses euphemisms ("nous avons retenu des
  profils plus proches", "we'll keep your profile on file", "the role has evolved").
- `relative_deadline`: the action's due date is relative ("d'ici vendredi", "within 48h",
  "next Tuesday"). The label resolves it per the rules.
- `timezone`: the sender is in another time zone (the Date header offset reflects it) and/or
  times are given with a zone (EST, GMT+1, IST, "heure de Paris").
- `forwarded_two_companies`: a forwarded or quoted thread in which a second company and/or
  role appears. The top message determines the label.
- `reply_thread`: the email replies to earlier messages, which are quoted below it.
- `missing_fields`: leave out details a reader would expect (no job title, or the company
  only appears in the email domain, etc.), consistent with `omit`.
- `fr_en_mix`: code-switching inside the email (FR body with EN tech/recruiting terms, or EN
  with FR sentences), as French tech people really write.
- `typos`: a few human typos as described above.
- `informal_startup`: very casual tone, often a founder or CTO, tutoiement, emojis possible.
- `agency_confidential_client`: a recruitment agency/ESN recruiter writing about a client
  they don't name, or name.
- `interview_format_trap`: mentions the interview format (visio / sur place) but NOT the
  job's work mode, so work_mode stays null unless `mention` includes it.
- `multi_purpose`: the email does two things (e.g. ack + test, invite + document request).
  Label using the precedence rules.
- `signature_noise`: long signature with an address, phone, LinkedIn, legal disclaimer, or an
  "Our values" banner.
- `role_switch`: rejects for one role and proposes another one (see §1.2 exception).

# Labeling rules (authoritative)

<labeling_rules>
{LABELING_RULES}
</labeling_rules>

# Output format

Return ONLY a JSON object, no markdown fences, no commentary:
{"examples": [{"spec_id": "<copy from spec>", "email": "<full email text with headers>", "label": {<10 fields in schema order>}}, ...]}

Return exactly one example per spec, in the same order. Before writing each label, re-read the email you wrote and
check every field against the rules: email_type precedence, email_type×stage table, interview
format ≠ work_mode, the deadline is the action's due date and not the meeting date (resolved
from the Date header), and next_step is in English.
