"""
Synthetic medical transcript generator for LLM benchmarking.

Produces realistic doctor-patient conversation transcripts with randomized
clinical details. NOT real patient data — purely for load testing purposes.

Token distribution target: ~600–1200 tokens input (realistic ambient session).
Output note expected: ~300–600 tokens (concise SOAP).
"""
import random
from typing import NamedTuple

# ── Clinical term pools ────────────────────────────────────────────────────────
CHIEF_COMPLAINTS = [
    "chest pain radiating to the left arm for the past three days",
    "shortness of breath on exertion, worse when lying flat at night",
    "persistent headache rated 7 out of 10, photophobic and phonophobic",
    "right knee pain and swelling following a fall two days ago",
    "fatigue and unintentional weight loss of 12 pounds over two months",
    "intermittent palpitations lasting seconds, no syncope",
    "lower back pain radiating down the left leg, worse with sitting",
    "burning epigastric pain, partially relieved by antacids",
    "productive cough with greenish sputum for five days, low-grade fever",
    "worsening ankle swelling bilaterally, worse in the evenings",
]

MEDICATIONS = [
    "metformin 1000 mg twice daily",
    "lisinopril 10 mg once daily",
    "atorvastatin 40 mg at bedtime",
    "levothyroxine 50 mcg every morning",
    "omeprazole 20 mg before meals",
    "amlodipine 5 mg once daily",
    "aspirin 81 mg daily",
    "sertraline 50 mg once daily",
    "albuterol inhaler as needed",
    "metoprolol succinate 25 mg once daily",
]

ALLERGIES = [
    "penicillin — causes hives",
    "sulfa drugs — causes rash",
    "codeine — causes nausea and vomiting",
    "ibuprofen — causes stomach upset",
    "no known drug allergies",
]

FAMILY_HISTORY = [
    "father with type 2 diabetes and hypertension, mother with breast cancer",
    "paternal history of coronary artery disease at age 55",
    "no significant family history",
    "mother with hypothyroidism, brother with asthma",
]

SOCIAL_HISTORY = [
    "non-smoker, occasional alcohol use, works as a teacher",
    "former smoker, 10 pack-years, quit 5 years ago, no alcohol",
    "never smoked, social drinker on weekends, sedentary occupation",
    "active smoker 1 PPD for 20 years, denies illicit drug use",
]

PHYSICAL_EXAM = [
    "Vitals: BP 138/88, HR 82 bpm regular, Temp 37.1°C, SpO2 97% on room air, BMI 28.4.",
    "Vitals: BP 122/76, HR 74 bpm, Temp 36.9°C, SpO2 99% on room air, RR 14.",
    "Vitals: BP 152/94, HR 96 bpm irregular, Temp 37.4°C, SpO2 95% on room air.",
    "Vitals: BP 108/68, HR 58 bpm, Temp 36.7°C, SpO2 98% on room air, weight 91 kg.",
]

EXAM_FINDINGS = [
    "Lungs clear to auscultation bilaterally. Heart regular rate and rhythm, no murmurs.",
    "Mild bibasilar crackles. Heart sounds distant with 2/6 systolic murmur at apex.",
    "Abdomen soft, mildly tender in the right upper quadrant, no rebound. Bowel sounds present.",
    "Mild pitting edema bilateral ankles 1+. JVD absent. No hepatomegaly.",
    "Neurological: alert and oriented ×4, cranial nerves II–XII intact, gait normal.",
    "Right knee: effusion present, range of motion 0–100°, Lachman negative.",
]

DIAGNOSES = [
    "type 2 diabetes mellitus, inadequately controlled",
    "essential hypertension, stage 2",
    "community-acquired pneumonia, mild severity",
    "acute exacerbation of chronic obstructive pulmonary disease",
    "major depressive disorder, moderate episode",
    "osteoarthritis of the right knee, moderate",
    "gastroesophageal reflux disease",
    "hypothyroidism",
    "migraine without aura",
    "heart failure with reduced ejection fraction",
]

PLANS = [
    "Increase metformin to 2000 mg daily. Recheck HbA1c in 3 months. Dietary counseling referral.",
    "Start amlodipine 5 mg daily. Home BP log. Follow up in 4 weeks.",
    "Azithromycin 500 mg daily for 5 days. Chest X-ray ordered. Increase fluids.",
    "Prednisone 40 mg taper over 5 days. Tiotropium inhaler added. Pulmonology referral.",
    "Increase sertraline to 100 mg. Safety plan reviewed. Follow up in 2 weeks.",
    "Orthopedics referral. Naproxen 500 mg BID with food. Physical therapy.",
    "Omeprazole 40 mg daily for 8 weeks. Avoid NSAIDs, caffeine, alcohol. Scope if no improvement.",
    "Increase levothyroxine to 75 mcg. Recheck TSH in 6 weeks. Continue current labs.",
    "Sumatriptan 50 mg PRN. Headache diary. Neurology referral if >4 episodes per month.",
    "Furosemide 20 mg daily added. Fluid restriction 1.5 L/day. Cardiology follow-up in 1 week.",
]

LABS = [
    "CBC with differential", "HbA1c", "comprehensive metabolic panel",
    "lipid panel", "TSH", "urinalysis", "chest X-ray", "12-lead EKG",
    "BNP level", "blood cultures ×2", "sputum culture",
]


class TranscriptSpec(NamedTuple):
    complaint: str
    meds: list[str]
    allergy: str
    family_hx: str
    social_hx: str
    vitals: str
    exam: str
    diagnosis: str
    plan: str
    lab: str


def _build_spec(rng: random.Random) -> TranscriptSpec:
    return TranscriptSpec(
        complaint  = rng.choice(CHIEF_COMPLAINTS),
        meds       = rng.sample(MEDICATIONS, k=rng.randint(2, 5)),
        allergy    = rng.choice(ALLERGIES),
        family_hx  = rng.choice(FAMILY_HISTORY),
        social_hx  = rng.choice(SOCIAL_HISTORY),
        vitals     = rng.choice(PHYSICAL_EXAM),
        exam       = rng.choice(EXAM_FINDINGS),
        diagnosis  = rng.choice(DIAGNOSES),
        plan       = rng.choice(PLANS),
        lab        = rng.choice(LABS),
    )


def generate_transcript(seed: int = 0) -> str:
    """
    Generate a realistic doctor-patient conversation transcript (~600–1000 tokens).

    Args:
        seed: RNG seed for reproducibility.

    Returns:
        Plain text transcript suitable for inclusion in an LLM prompt.
    """
    rng  = random.Random(seed)
    spec = _build_spec(rng)

    med_list = ", ".join(spec.meds)
    transcript = f"""DOCTOR: Good morning. What brings you in today?
PATIENT: I've been having {spec.complaint}. It started about a week ago and hasn't improved.
DOCTOR: I'm sorry to hear that. Any associated symptoms — nausea, vomiting, fever, chills?
PATIENT: Some mild nausea, no fever. I've also been more fatigued than usual.
DOCTOR: Any recent travel, sick contacts, or unusual exposures?
PATIENT: No recent travel. My coworker had a cold last week.
DOCTOR: Are you currently taking any medications?
PATIENT: Yes, I'm on {med_list}.
DOCTOR: Any allergies to medications?
PATIENT: {spec.allergy}.
DOCTOR: Any significant family history?
PATIENT: {spec.family_hx}.
DOCTOR: Social history — do you smoke, drink alcohol, or use any recreational substances?
PATIENT: {spec.social_hx}.
DOCTOR: Alright. Let me do a physical examination.
[PHYSICAL EXAM]
{spec.vitals}
{spec.exam}
DOCTOR: Based on what I'm seeing, my assessment is {spec.diagnosis}.
PATIENT: What does that mean for me, Doctor?
DOCTOR: It means we need to adjust your management. Here's my plan: {spec.plan}
I'm also ordering {spec.lab} today.
PATIENT: Should I be worried?
DOCTOR: It's manageable with the right treatment. We'll monitor closely and adjust as needed.
Any questions before you go?
PATIENT: No, I think I understand. Thank you.
DOCTOR: Take care. See you at your follow-up."""

    return transcript


def generate_prompt(seed: int = 0, model_name: str = "gpt-oss-20b") -> dict:
    """
    Build an OpenAI-format chat completion request from a generated transcript.
    Compatible with vLLM's /v1/chat/completions endpoint.
    """
    from generators.prompt_templates import SYSTEM_PROMPT
    transcript = generate_transcript(seed=seed)
    return {
        "model": model_name,
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user",   "content": f"TRANSCRIPT:\n{transcript}"},
        ],
        "max_tokens": 512,
        "temperature": 0.1,
        "stream": False,
    }
