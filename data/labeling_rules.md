# Labeling rules (v1.4, frozen 2026-10-21)

These rules are the single source of truth for labels. The generation prompt, the hand
verification of the test set, and the error analysis all use this file. If a case
isn't covered here, add a rule **before** labeling it and record the change in the
changelog at the bottom. Never change a rule silently after the test set is frozen.

Types and enums are defined in `schema.json`. This file defines their meaning.

---

## 0. Input format

Every input email has this format. The headers are part of the model input.

```
From: Léa Martin <lea.martin@novatek-solutions.fr>
Date: Tue, 21 Oct 2026 14:03:00 +0200
Subject: Votre candidature – Data Engineer (Stage PFE)

<body, possibly including quoted/forwarded messages>
```

- `Date` is the **reference date** for resolving relative dates (§9).
- The candidate is always the recipient. "You" means the candidate.
- **Forwarded threads / replies.** Label the **most recent message** (the top one), using
  quoted messages only as context to fill in fields the top message refers to
  ("see below", "as discussed"). If the top message adds nothing (a bare forward with
  no text, or just "FYI"), label the forwarded message as if it had been received directly,
  but keep `sender_role` = the forwarder's role.

---

## 1. `email_type`: what this email *is*

Pick the **single primary purpose**. If an email does several things, apply the precedence
order in §1.2.

### 1.1 Definitions

| Value | Definition | Typical signals |
|---|---|---|
| `application_ack` | Confirms that an application was received, or gives a status update with **no decision and no request**. | "Nous avons bien reçu votre candidature", "your application is under review" |
| `interview_invite` | **First notice** that the candidate is invited to an interview round (any round), with or without a proposed time or booking link. | "Nous souhaitons vous rencontrer", "we'd like to invite you to a technical interview", Calendly link for a new round |
| `assessment_request` | Asks the candidate to complete a test: online coding test, take-home, case study, psychometric/personality test, language test. **Reminders** about a pending test count too. | HackerRank, Codingame, "test technique", "étude de cas à rendre" |
| `scheduling` | Logistics for a round that **was already agreed**: confirming, rescheduling, cancelling, sending the video link or address, time-zone clarification. | "Je vous confirme notre RDV", "can we move to Thursday?", "voici le lien Teams" |
| `info_request` | Asks for information or documents **not about the timing of a meeting**: CV update, transcript, convention de stage, internship start/end dates, salary expectations, right-to-work, references, **onboarding paperwork** (ID, RIB, social security number, signed documents). | "Pourriez-vous m'envoyer vos dates de stage ?", "what are your salary expectations?" |
| `rejection` | The candidacy for this role is ended by the company. This includes vague rejections. | "nous ne donnerons pas suite", "we've decided to move forward with other candidates", "we'll keep your profile on file" *when combined with not proceeding* |
| `offer` | Offers the position (formal letter, verbal offer recap, "we'd like to make you an offer"), or follows up on an existing offer: negotiation, contract to sign, offer reminder. | "proposition d'embauche", "offer letter attached", "promesse d'embauche" |
| `networking_cold_outreach` | Unsolicited contact where **the candidate has not applied**: recruiter/headhunter sourcing, founder reaching out, alumni introduction. Also applies when the outreach proposes a call. | "J'ai vu votre profil sur LinkedIn", "I came across your GitHub" |
| `other` | Everything else: job alerts, job-board newsletters, career-fair or event invites, confirmation of the candidate's own withdrawal, onboarding **information** after an accepted offer that asks for nothing (first-day address, schedule, welcome), generic marketing. | "Nouvelles offres pour vous", "Votre inscription au forum", "Bienvenue ! Votre 1er jour" |

### 1.2 Precedence (multi-purpose emails)

When several definitions apply, pick the **first** match in this order:

1. `offer`
2. `rejection`
3. `assessment_request`
4. `interview_invite`
5. `scheduling`
6. `info_request`
7. `networking_cold_outreach`
8. `application_ack`
9. `other`

Examples:
- "Thanks for applying! Please complete this Codingame test within 5 days." → `assessment_request` (not `application_ack`).
- "Following your interview, we'd like to invite you to a final round. Could you also send your transcript?" → `interview_invite` (not `info_request`).
- "Unfortunately we won't proceed for the Data Analyst role, but we'd like to interview you for our BI internship." → `interview_invite` for the BI role. **Exception to precedence**: when an email rejects for role A and invites for role B, label the **new opportunity** (role B), because that is the live candidacy. `job_title` = role B.

### 1.3 Hard boundaries

- **interview_invite vs scheduling.** Has an interview for *this round* already been agreed
  in a previous exchange? If yes → `scheduling`. If this email is the first time the round is
  mentioned → `interview_invite`, even if it already sets a fixed time.
- **networking_cold_outreach vs interview_invite.** Did the candidate apply? If the email
  gives no sign of an application (no "suite à votre candidature", no reference to a job
  the candidate applied to), it's `networking_cold_outreach`.
- **rejection vs application_ack ("we'll keep your profile on file").**
  - "We are not moving forward… we'll keep your CV for future opportunities" → `rejection`.
  - "Your profile is under review; we'll get back to you" → `application_ack`.
  - "The position is filled/cancelled, we'll keep your profile" → `rejection`.
  - The test: is the candidate still being considered **for this role**? No → `rejection`.
- **A meeting about an existing offer** (call to present or negotiate the offer, even when
  it only fixes a time) → `offer`, by precedence. `scheduling` is only for interview rounds.
- **Onboarding.** Asks the candidate for documents or information → `info_request`. Only
  informs (first-day logistics, welcome) → `other`. Stage is `offer` in both cases.
- **info_request vs scheduling.** "Send me your availabilities for the interview" is about
  meeting timing, so it's `interview_invite` (first notice) or `scheduling` (round already agreed). Never `info_request`.

---

## 2. `stage`: where the candidacy stands *after* this email

`stage` describes the candidacy's position in the funnel, **as implied by this email alone**
(plus its quoted thread). Do not guess from outside knowledge.

| Value | Definition |
|---|---|
| `applied` | Application received; no screening step has started. |
| `screening` | Early filtering: first call with a recruiter/HR, CV pre-selection, an online test sent **before any interview**, info requests before interviews. |
| `technical_interview` | Any interview round past the first recruiter screen that is **not explicitly the final round**: interviews with the team or hiring manager, technical/case interviews, take-homes given **after** an interview. |
| `final_interview` | The email **explicitly** says it is the last round ("dernier entretien", "final round", "dernière étape"). A CEO interview alone does not make it final unless the email says so. |
| `offer` | Offer extended, under negotiation, accepted, or onboarding in progress. |
| `closed` | Candidacy ended: rejection, position cancelled/filled, candidate withdrew, candidate declined the offer. |
| `null` | No candidacy exists (cold outreach, job alerts, newsletters, event invites), or nothing in the email indicates the stage. |

Decision rules:
- **First conversation of the process** (intro call, "premier échange", "first call", a short
  call to "present the position and learn about you") → `screening`, **whoever sends it**,
  including a hiring manager or founder.
- **Later rounds** (the email refers to a previous call/interview, or the interview is with the
  team / technical / a case discussion) → `technical_interview`, unless explicitly final.
- **Assessments.** If nothing in the email refers to a past interview → `screening`. If it
  refers to a past interview ("suite à notre entretien, voici le cas pratique") → `technical_interview`.
- **Scheduling/info_request** take the stage of the round they are about.
- **info_request before any interview:** a request to *complete the application file*
  (missing CV/transcript/form, "pour compléter votre dossier") → `applied`. A request that is
  part of evaluating the candidate (salary expectations, availability dates, motivation
  questions, references) → `screening`.
- **Onboarding after acceptance** (`email_type` = `other`) → `offer`.
- **Candidate withdrawal confirmation** (`other`) → `closed`.

### 2.1 Allowed `email_type` × `stage` combinations

Any combination outside this table is a **label error**. The generation validator rejects it.

| email_type | allowed stage |
|---|---|
| application_ack | applied |
| interview_invite | screening, technical_interview, final_interview |
| assessment_request | screening, technical_interview |
| scheduling | screening, technical_interview, final_interview |
| info_request | applied, screening, technical_interview, final_interview, offer |
| rejection | closed |
| offer | offer |
| networking_cold_outreach | null |
| other | any (including null) |

---

## 3. `company`

- The **employer** offering the job. Never the ATS (Workday, Greenhouse, Lever, Taleo,
  SmartRecruiters), never the job board (Welcome to the Jungle, Indeed, LinkedIn, HelloWork).
- **Recruitment agency / ESN placing for a client:** the client if it is named. If the client
  is confidential ("un grand groupe bancaire"), use the agency/ESN name, because the agency
  is the counterparty the candidate deals with.
  - Exception: if the ESN itself hires the candidate (typical CDI ESN), the ESN is the company.
- Write the **core brand name**: drop legal suffixes (SAS, SA, SARL, Inc., Ltd, GmbH) **and**
  these generic trailing descriptors, **and only these** (the list is exhaustive): France, Group,
  Groupe, Systems, Technologies, Solutions, Labs, Analytics, Digital, Consulting, Conseil,
  & Co, & Associés. Any other word is part of the brand and stays ("Marsan Énergies",
  "Clairval Industries", "Atelier Brisac", "Banque Valmont"). Keep the brand's casing.
  "NOVATEK SOLUTIONS SAS" → "Novatek"; "Ilialogic France" → "Ilialogic"; "Ondimetrics & Co" → "Ondimetrics";
  "Vauclair Conseil" → "Vauclair"; "Bellerive & Associés" → "Bellerive".
  Keep a descriptor only if the name would otherwise be a generic word ("Data Group" stays).
- If the company appears only in the sender's email domain (e.g. `@kelvio.io`) and nowhere in
  the text, still extract it ("Kelvio"), but only when the domain is clearly a company
  domain (not gmail/outlook/ATS domains).
- Two companies in a forwarded thread → the company of the **top message's** candidacy (§0).

## 4. `job_title`

- As written in the email, in the email's language. Do not translate.
- Strip contract-type prefixes/suffixes, gender markers, codes and locations:
  "Stage PFE – Data Engineer (H/F) – Réf 2026-114 – Paris" → "Data Engineer".
  Strip: `Stage`, `Stagiaire`, `PFE`, `Alternance`, `Alternant(e)`, `Internship`, `CDI`, `CDD`,
  `(H/F)`, `(F/H)`, `(M/F/D)`, `(F/H/X)`, reference numbers, city names.
- **Exception:** when the contract word is part of the title noun itself, keep it.
  "Backend Intern", "Data Intern" and "Stagiaire Data" stay as they are, because removing the word leaves no title.
- Seniority words stay ("Junior Data Engineer").
- Multiple roles listed (cold outreach mentioning "plusieurs postes") with no single focus → `null`.

## 5. `contract_type`

Only if it is **stated or unambiguous** in the email. Otherwise `null`.

| Value | Triggers |
|---|---|
| `internship` | stage, stagiaire, PFE, stage de fin d'études, internship, intern, convention de stage |
| `apprenticeship` | alternance, alternant, apprentissage, contrat de professionnalisation, work-study |
| `CDI` | CDI, permanent contract, "contrat à durée indéterminée", pré-embauche en CDI after an internship **only if the email is about the CDI** |
| `CDD` | CDD, fixed-term, "contrat de 6 mois" (salaried) |
| `freelance` | freelance, mission en indépendant, portage salarial, contractor (self-employed) |

- "Full-time"/"temps plein" alone → `null`. VIE, CIFRE, and other types → `null`.
- PFE internship with a CDI pré-embauche mentioned as a possibility → `internship` (that is the
  contract being recruited for).

## 6. `work_mode`

Only the **job's** working arrangement, as stated. Otherwise `null`.

- `remote`: full remote, 100% télétravail, "remote-first, no office".
- `hybrid`: any mix ("2 jours de TT par semaine", "hybrid", "flexible remote with office in Lyon").
- `onsite`: explicitly onsite/no remote ("poste sur site", "présentiel obligatoire").
- **Interview format is not work mode.** "L'entretien se fera en visio" → says nothing about
  `work_mode`. "Rendez-vous dans nos locaux à Nantes" → not `onsite` either.
- Office address in the signature → not evidence. → `null`.

## 7. `sender_role`

Label who wrote the email, judging from the signature, From address and content.

| Value | Definition |
|---|---|
| `automated_system` | Sent from a no-reply / ATS / job-board address, or an obvious template (placeholders, "Ceci est un message automatique", signed only by a team like "L'équipe Recrutement"). **This holds even if a person's name appears in a template signature, as long as the From address is no-reply/ATS.** |
| `recruiter` | Talent acquisition, recruiter, chargé(e) de recrutement, campus recruiter, agency recruiter, headhunter, ESN business manager doing recruitment. |
| `HR` | HR generalist, HRBP, RH, payroll/admin, gestionnaire de stages; anyone handling contracts, conventions de stage, onboarding paperwork. |
| `hiring_manager` | The future manager or a decision-maker on the team: team lead, engineering manager, CTO, a founder who interviews or decides. |
| `other` | Peers/employees (e.g. referral), school career services, alumni, event organizers, the candidate's own forwarded note. |

- **Title beats task.** If the signature gives a title, the title decides: a "Chargée de
  recrutement" sending onboarding paperwork is still `recruiter`. The task-based descriptions
  above apply only when there is no title.
- **Content beats address for `automated_system`.** A personal, non-templated message sent
  through an ATS relay address (`…@mail.<ats>.com`) is from the person, so use their role. Use
  `automated_system` only for templated or system-generated content.
- A startup founder doing recruitment for their own company → `hiring_manager`.
- No title or role cue at all (first name only, personal tone) → `recruiter` if the sender
  runs the process (scheduling, follow-ups), `hiring_manager` if they speak as the future
  manager ("mon équipe", "tu bosserais avec moi").
- Bare forward → the forwarder's role (§0).

## 8. `action_required`

`true` if the email **explicitly asks** the candidate to do something: reply, confirm, book a
slot, complete a test, send a document, sign, give availability. This includes soft asks
("n'hésitez pas à me dire si vous êtes intéressé" in a cold outreach → `true`).

`false` for: pure acknowledgements, rejections (even "feel free to apply again"), job alerts,
and confirmations of an already-agreed meeting that only require showing up
("à mardi !" → `false`; "merci de confirmer votre présence" → `true`).

## 9. `deadline`

- The due date of the **requested action**, as `YYYY-MM-DD`. **Not the interview date.**
  ("Entretien le 28/10, merci de confirmer avant le 24/10" → `2026-10-24`.)
- `null` if there is no requested action or no stated due date.
- A **time window for the meeting itself** ("send me slots over the coming two weeks",
  "vos dispos sur les 10 prochains jours") is not a deadline → `null`. Only a due date for the
  action ("répondez avant jeudi", "complete within 5 days") counts.
- Must be `null` when `action_required` is `false` (enforced by the schema).
- Resolve relative dates against the email's `Date` header (sender's local date):

| Expression | Resolution (email dated Tue 21 Oct 2026) |
|---|---|
| "today / aujourd'hui" | 2026-10-21 |
| "tomorrow / demain" | 2026-10-22 |
| "this Friday / ce vendredi / vendredi" | the coming Friday, on or after the send date → 2026-10-24 |
| "next Tuesday / mardi prochain" | the first Tuesday **strictly after** the send date → 2026-10-28 |
| "within N days / sous N jours / d'ici N jours" | send date + N calendar days → (N=5) 2026-10-26 |
| "within 48h / sous 48h" | send date + 2 days → 2026-10-23 |
| "end of the week / fin de semaine" | Friday of the current week → 2026-10-24 |
| "end of the month / fin du mois" | last day of the month → 2026-10-31 |
| "ASAP / dès que possible / rapidement" | `null` (no date) |

- Explicit dates without a year: use the next occurrence on or after the send date.
- Numeric dates are **DD/MM** in French emails and in English emails from a French sender;
  **MM/DD** only for an explicitly US sender. If it is still ambiguous, prefer DD/MM.
- **Time zones.** The deadline date is the date in the time zone **stated in the email**
  ("Friday 23:59 PST" → Friday). If the email states no time zone, use the Date header's offset.
  Don't convert to the candidate's time zone.

## 10. `next_step`

- One short phrase, **always in English**, ≤ 120 characters, describing the next concrete step
  from the candidate's point of view. If there is a date/time, include it as ISO
  (`2026-10-28 10:00`) with the stated time zone if any.
- Templates (prefer these shapes):
  - "Complete Codingame test by 2026-10-26"
  - "Technical interview with the data team on 2026-10-28 10:00 (video)"
  - "Book a screening call via the scheduling link"
  - "Send internship agreement (convention de stage) to HR"
  - "Reply if interested in the role"
- `null` when there is no next step (most rejections, plain acks with no timeline, job alerts).
  "We will get back to you within 2 weeks" → "Wait for the company's answer (about 2 weeks)".
- **Evaluation note:** `next_step` is free text. It is scored as null/non-null agreement plus a
  soft similarity metric, and it is **excluded** from full-record exact match.

---

## 11. Worked examples

**A. ATS acknowledgement with test**
```
From: Kelvio Recrutement <no-reply@kelvio.recruitee.com>
Date: Mon, 20 Oct 2026 09:12:00 +0200
Subject: Votre candidature – Stage PFE Data Engineer (H/F)

Bonjour Yanis, merci pour votre candidature ! Afin de poursuivre, merci de compléter
notre test en ligne sous 5 jours : https://... L'équipe Talent de Kelvio
```
```json
{"email_type":"assessment_request","company":"Kelvio","job_title":"Data Engineer","contract_type":"internship","work_mode":null,"stage":"screening","sender_role":"automated_system","action_required":true,"deadline":"2026-10-25","next_step":"Complete online test by 2026-10-25"}
```

**B. Vague rejection**
```
From: Inès Garnier <ines.garnier@bluemarine-group.com>
Date: Thu, 23 Oct 2026 17:40:00 +0200
Subject: RE: Candidature alternance Développeur Full-Stack

Bonjour, après étude attentive, nous avons retenu des profils correspondant davantage à
nos besoins actuels. Nous conservons toutefois votre CV pour de futures opportunités.
Inès Garnier, Chargée de recrutement
```
```json
{"email_type":"rejection","company":"Bluemarine","job_title":"Développeur Full-Stack","contract_type":"apprenticeship","work_mode":null,"stage":"closed","sender_role":"recruiter","action_required":false,"deadline":null,"next_step":null}
```

**C. Cold outreach (EN/FR mix, startup tone)**
```
From: Tom Reyes <tom@orbiflow.ai>
Date: Wed, 22 Oct 2026 11:05:00 +0200
Subject: Quick chat ?

Hey ! Saw your projet on GitHub (the Kafka one), super cool. We're hiring a data eng
intern for our PFE batch, full remote. Up for a 15min call next Tuesday ? Tom, co-founder @ Orbiflow
```
```json
{"email_type":"networking_cold_outreach","company":"Orbiflow","job_title":"data eng intern","contract_type":"internship","work_mode":"remote","stage":null,"sender_role":"hiring_manager","action_required":true,"deadline":null,"next_step":"Reply to schedule a 15-min call on 2026-10-28"}
```
(Note: "next Tuesday" is the proposed **call date**, not a deadline for the action → `deadline` = `null`.
"data eng intern" keeps "intern" because removing it leaves no title.)

**D. Scheduling with time zone**
```
From: Priya Nair <priya.nair@lumencloud.com>
Date: Fri, 24 Oct 2026 16:20:00 -0400
Subject: Re: Technical interview – Backend Intern

Hi! As discussed, can we move our technical interview to Monday 27th, 10am EST
(4pm Paris)? Please confirm by Sunday. Best, Priya – Engineering Manager
```
```json
{"email_type":"scheduling","company":"Lumencloud","job_title":"Backend Intern","contract_type":"internship","work_mode":null,"stage":"technical_interview","sender_role":"hiring_manager","action_required":true,"deadline":"2026-10-26","next_step":"Confirm technical interview on 2026-10-27 10:00 EST"}
```

---

## Changelog
- v1.4 (2026-10-21): §7 tie-breaks: a signature title beats task-based role descriptions, and
  personal messages sent via an ATS relay take the person's role. Found in the train chunk 1
  blind relabel (10 sender_role disagreements); the gold labels already followed both, so none changed.
- v1.3 (2026-10-21): the §3 descriptor list is exhaustive and now includes Conseil and
  & Associés. Found in train chunk 1, where multi-word names were stripped inconsistently.
  Company labels are canonicalized at ingest.
- v1.2 (2026-10-21): info_request before interviews: completing the application file →
  `applied`, evaluation questions → `screening` (§2). Found in the 30-email check batch; the one
  affected label already followed it.
- v1.1 (2026-10-21, before any data was kept): from the pilot's blind-relabel review —
  company = core brand without generic descriptors (§3); onboarding requests → `info_request`
  (§1.1, §1.3); meetings about an existing offer → `offer`, scheduling×offer removed (§1.3,
  §2.1); a first conversation is `screening` whoever the sender (§2); a slot window is not a
  deadline (§9); fallback rule for sender_role without cues (§7). Pilot v0 set aside.
- v1 (2026-10-21): initial freeze. `internship_PFE` → `internship`; `action_required` and
  `deadline` are two flat fields; `deadline` is date-only.
