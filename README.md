# machine-learning-saude
Machine Learning in Health

Data Dictionary

encounter_id: Unique identifier associated with a patient unit stay

patient_id: Unique identifier associated with a patient

hospital_id: Unique identifier associated with a hospital

age: The age of the patient on unit admission

bmi: The body mass index of the person on unit admission

elective_surgery: Whether the patient was admitted to the hospital for an elective surgical operation

ethnicity: The common national or cultural tradition which the person belongs to

gender: Sex of the patient

height: The height of the person on unit admission

icu_admit_source: The location of the patient prior to being admitted to the unit

icu_id: A unique identifier for the unit to which the patient was admitted

icu_stay_type: string

icu_type: A classification which indicates the type of care the unit is capable of providing

pre_icu_los_days: The length of stay of the patient between hospital admission and unit admission

weight: The weight (body mass) of the person on unit admission

apache_2_diagnosis: The APACHE II diagnosis for the ICU admission

apache_3j_diagnosis: The APACHE III-J sub-diagnosis code which best describes the reason for the ICU admission

apache_post_operative: The APACHE operative status; 1 for post-operative, 0 for non-operative

arf_apache: Whether the patient had acute renal failure during the first 24 hours of their unit stay, defined as a 24 hour urine output <410ml, creatinine >=133 micromol/L and no chronic dialysis

gcs_eyes_apache: The eye opening component of the Glasgow Coma Scale measured during the first 24 hours which results in the highest APACHE III score

gcs_motor_apache: The motor component of the Glasgow Coma Scale measured during the first 24 hours which results in the highest APACHE III score

gcs_unable_apache: Whether the Glasgow Coma Scale was unable to be assessed due to patient sedation

gcs_verbal_apache: The verbal component of the Glasgow Coma Scale measured during the first 24 hours which results in the highest APACHE III score

heart_rate_apache: The heart rate measured during the first 24 hours which results in the highest APACHE III score

intubated_apache: Whether the patient was intubated at the time of the highest scoring arterial blood gas used in the oxygenation score

map_apache: The mean arterial pressure measured during the first 24 hours which results in the highest APACHE III score

resprate_apache: The respiratory rate measured during the first 24 hours which results in the highest APACHE III score

temp_apache: The temperature measured during the first 24 hours which results in the highest APACHE III score

ventilated_apache: Whether the patient was invasively ventilated at the time of the highest scoring arterial blood gas using the oxygenation scoring algorithm, including any mode of positive pressure ventilation delivered through a circuit attached to an endo-tracheal tube or tracheostomy

d1_diasbp_max: The patient's highest diastolic blood pressure during the first 24 hours of their unit stay, either non-invasively or invasively measured

d1_diasbp_min: The patient's lowest diastolic blood pressure during the first 24 hours of their unit stay, either non-invasively or invasively measured

d1_diasbp_noninvasive_max: The patient's highest diastolic blood pressure during the first 24 hours of their unit stay, non-invasively measured

d1_diasbp_noninvasive_min: The patient's lowest diastolic blood pressure during the first 24 hours of their unit stay, non-invasively measured

d1_heartrate_max: The patient's highest heart rate during the first 24 hours of their unit stay


d1_heartrate_min: The patient's lowest heart rate during the first 24 hours of their unit stay

d1_mbp_max: The patient's highest mean blood pressure during the first 24 hours of their unit stay, either non-invasively or invasively measured

d1_mbp_min: The patient's lowest mean blood pressure during the first 24 hours of their unit stay, either non-invasively or invasively measured

d1_mbp_noninvasive_max: The patient's highest mean blood pressure during the first 24 hours of their unit stay, non-invasively measured

d1_mbp_noninvasive_min: The patient's lowest mean blood pressure during the first 24 hours of their unit stay, non-invasively measured

d1_resprate_max: The patient's highest respiratory rate during the first 24 hours of their unit stay

d1_resprate_min: The patient's lowest respiratory rate during the first 24 hours of their unit stay

d1_spo2_max: The patient's highest peripheral oxygen saturation during the first 24 hours of their unit stay

d1_spo2_min: The patient's lowest peripheral oxygen saturation during the first 24 hours of their unit stay

d1_sysbp_max: The patient's highest systolic blood pressure during the first 24 hours of their unit stay, either non-invasively or invasively measured

d1_sysbp_min: The patient's lowest systolic blood pressure during the first 24 hours of their unit stay, either non-invasively or invasively measured

d1_sysbp_noninvasive_max: The patient's highest systolic blood pressure during the first 24 hours of their unit stay, invasively measured

d1_sysbp_noninvasive_m: The patient's lowest systolic blood pressure during the first 24 hours of their unit stay, invasively measured 

d1_temp_max: The patient's highest core temperature during the first 24 hours of their unit stay, invasively measured

d1_temp_min: The patient's lowest core temperature during the first 24 hours of their unit stay

h1_diasbp_max: The patient's highest diastolic blood pressure during the first hour of their unit stay, either non-invasively or invasively measured

h1_diasbp_min: The patient's lowest diastolic blood pressure during the first hour of their unit stay, either non-invasively or invasively measured

h1_diasbp_noninvasive_max: The patient's highest diastolic blood pressure during the first hour of their unit stay, invasively measured

h1_diasbp_noninvasive_min: The patient's lowest diastolic blood pressure during the first hour of their unit stay, invasively measured

h1_heartrate_max: The patient's highest heart rate during the first hour of their unit stay

h1_heartrate_min: The patient's lowest heart rate during the first hour of their unit stay

h1_mbp_max: The patient's highest mean blood pressure during the first hour of their unit stay, either non-invasively or invasively measured

h1_mbp_min: The patient's lowest mean blood pressure during the first hour of their unit stay, either non-invasively or invasively measured

h1_mbp_noninvasive_max: The patient's highest mean blood pressure during the first hour of their unit stay, non-invasively measured

h1_mbp_noninvasive_min: The patient's lowest mean blood pressure during the first hour of their unit stay, non-invasively measured

h1_resprate_max: The patient's highest respiratory rate during the first hour of their unit stay

h1_resprate_min: The patient's lowest respiratory rate during the first hour of their unit stay

h1_spo2_max: The patient's highest peripheral oxygen saturation during the first hour of their unit stay

h1_spo2_min: The patient's lowest peripheral oxygen saturation during the first hour of their unit stay

h1_sysbp_max: The patient's highest systolic blood pressure during the first hour of their unit stay, either non-invasively or invasively measured

h1_sysbp_min: The patient's lowest systolic blood pressure during the first hour of their unit stay, either non-invasively or invasively measured

h1_sysbp_noninvasive_max: The patient's highest systolic blood pressure during the first hour of their unit stay, non-invasively measured

h1_sysbp_noninvasive_min: The patient's lowest systolic blood pressure during the first hour of their unit stay, non-invasively measured

d1_glucose_max: The highest glucose concentration of the patient in their serum or plasma during the first 24 hours of their unit stay

d1_glucose_min: The lowest glucose concentration of the patient in their serum or plasma during the first 24 hours of their unit stay

d1_potassium_max: The highest potassium concentration for the patient in their serum or plasma during the first 24 hours of their unit stay

d1_potassium_min: The lowest potassium concentration for the patient in their serum or plasma during the first 24 hours of their unit stay

apache_4a_hospital_death_prob: The APACHE IVa probabilistic prediction of in-hospital mortality for the patient which utilizes the APACHE III score and other covariates, including diagnosis.

apache_4a_icu_death_prob: The APACHE IVa probabilistic prediction of in ICU mortality for the patient which utilizes the APACHE III score and other covariates, including diagnosis

aids: Whether the patient has a definitive diagnosis of acquired immune deficiency syndrome (AIDS) (not HIV positive alone)

cirrhosis: Whether the patient has a history of heavy alcohol use with portal hypertension and varices, other causes of cirrhosis with evidence of portal hypertension and varices, or biopsy proven cirrhosis. This comorbidity does not apply to patients with a functioning liver transplant.

diabetes_mellitus: Whether the patient has been diagnosed with diabetes, either juvenile or adult onset, which requires medication.

hepatic_failure: Whether the patient has cirrhosis and additional complications including jaundice and ascites, upper GI bleeding, hepatic encephalopathy, or coma.

immunosuppression: Whether the patient has their immune system suppressed within six months prior to ICU admission for any of the following reasons; radiation therapy, chemotherapy, use of non-cytotoxic immunosuppressive drugs, high dose steroids (at least 0.3 mg/kg/day of methylprednisolone or equivalent for at least 6 months).

leukemia: Whether the patient has been diagnosed with acute or chronic myelogenous leukemia, acute or chronic lymphocytic leukemia, or multiple myeloma.

lymphoma: Whether the patient has been diagnosed with non-Hodgkin lymphoma.

solid_tumor_with_metastasis: Whether the patient has been diagnosed with any solid tumor carcinoma (including malignant melanoma) which has evidence of metastasis.

apache_3j_bodysystem: Admission diagnosis group for APACHE III

apache_2_bodysystem: Admission diagnosis group for APACHE II

hospital_death:Whether the patient died during this hospitalization