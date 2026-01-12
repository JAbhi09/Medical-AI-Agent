import re
import os
import json
import time
import sqlite3
from typing import Dict, List, Tuple, Optional, Any
from dataclasses import dataclass
from enum import Enum
import requests
from pathlib import Path


import torch
from transformers import AutoTokenizer, pipeline
from loguru import logger

from dotenv import load_dotenv
load_dotenv(verbose=True)

class EntityType(Enum):

    DISEASE = "DISEASE"
    SYMPTOM = "SYMPTOM"
    MEDICATION = "MEDICATION"
    DOSAGE = "DOSAGE"
    ANATOMY = "ANATOMY"
    UNKNOWN = "UNKNOWN"


@dataclass
class MedicalEntity:

    text: str
    type: EntityType
    start_pos: int
    end_pos: int
    confidence: float
    umls_code: Optional[str] = None
    normalized_text: Optional[str] = None
    metadata: Optional[Dict] = None

    def to_dict(self):

        return {
            "text": self.text,
            "type": self.type.value,
            "position": {"start": self.start_pos, "end": self.end_pos},
            "confidence": round(self.confidence, 3),
            "umls_code": self.umls_code,
            "normalized": self.normalized_text,
            "metadata": self.metadata
        }


class NERConfig:


    BIOBERT_MODEL = "dmis-lab/biobert-base-cased-v1.2"
    MAX_LENGTH = 512
    BATCH_SIZE = 1


    AUTO_ACCEPT_THRESHOLD = 0.50
    REVIEW_THRESHOLD = 0.30
    REJECT_THRESHOLD = 0.20


    UMLS_API_KEY = os.environ.get("UMLS_API_KEY", "")
    UMLS_BASE_URL = "https://uts-ws.nlm.nih.gov/rest"
    UMLS_AUTH_URL = "https://utslogin.nlm.nih.gov/cas/v1/api-key"


    CACHE_DB_PATH = "medical_ner_cache.db"
    CACHE_EXPIRY_DAYS = 30


    ENABLE_UMLS = True
    USE_CACHE = True
    DEBUG_MODE = False


    UMLS_RATE_LIMIT_DELAY = 0.5  # Seconds between API calls
    UMLS_MAX_RETRIES = 3



class RuleBasedClassifier:

    def __init__(self):
        self.patterns = self._initialize_patterns()
        self.compile_patterns()

    @staticmethod
    def _initialize_patterns() -> Dict[EntityType, List[str]]:

        return {
            EntityType.MEDICATION: [
                # Common drug suffixes
                r'\b\w+(olol|pril|pine|zole|cillin|mycin|vir|mab|nib|stat|pram|zepam|done|sone|ide|ine|ate)\b',
                # Common drug names (top 24)
                r'\b(metformin|insulin|aspirin|ibuprofen|acetaminophen|tylenol|advil|'
                r'amoxicillin|lisinopril|metoprolol|atorvastatin|lipitor|omeprazole|'
                r'levothyroxine|gabapentin|prednisone|hydrochlorothiazide|furosemide|'
                r'amlodipine|losartan|simvastatin|pantoprazole|warfarin|tramadol)\b',
            ],

            EntityType.DOSAGE: [
                r'\d+\.?\d*\s?(mg|milligrams?|mcg|micrograms?|g|grams?|ml|milliliters?|'
                r'cc|units?|iu|tablets?|pills?|caps?|capsules?|drops?|puffs?)',
                r'\b(once|twice|three times|four times)\s+(daily|a day|per day)',
                r'\b(bid|tid|qid|prn|qd|qhs|qod)\b',
                r'\b(every|each)\s+\d+\s+(hours?|days?|weeks?|months?)',
            ],

            EntityType.SYMPTOM: [
                # Pain patterns
                r'\b\w*\s*(pain|ache|soreness|tenderness|discomfort)\b',
                # Common symptoms
                r'\b(fever|chills|fatigue|weakness|nausea|vomiting|diarrhea|constipation|'
                r'cough|shortness of breath|dyspnea|headache|dizziness|vertigo|'
                r'rash|itching|pruritus|swelling|edema|bleeding|discharge|'
                r'numbness|tingling|paresthesia|insomnia|anxiety|depression)\b',
                # Symptom descriptors
                r'\b(acute|chronic|severe|mild|moderate|intermittent|constant|'
                r'sudden|gradual|sharp|dull|burning|throbbing|stabbing)\s+\w+',
            ],

            EntityType.DISEASE: [
                # Disease suffixes
                r'\b\w+(itis|osis|emia|oma|pathy|syndrome|disease|disorder|deficiency)\b',
                # Common diseases
                r'\b(diabetes|hypertension|asthma|copd|pneumonia|bronchitis|'
                r'arthritis|osteoporosis|cancer|tumor|malignancy|'
                r'infection|sepsis|stroke|cva|mi|myocardial infarction|'
                r'heart failure|chf|atrial fibrillation|afib|'
                r'anemia|hypothyroidism|hyperthyroidism|'
                r'depression|anxiety|bipolar|schizophrenia|'
                r'alzheimer|dementia|parkinson|epilepsy|seizure)\b',
            ],

            EntityType.ANATOMY: [
                # Body parts and organs
                r'\b(head|brain|skull|eye|ear|nose|mouth|throat|neck|'
                r'chest|heart|lung|liver|kidney|stomach|intestine|colon|'
                r'arm|leg|foot|hand|finger|toe|back|spine|bone|muscle|'
                r'skin|blood|nerve|vessel|artery|vein)\b',
                # Anatomical regions
                r'\b(cardiac|pulmonary|hepatic|renal|gastric|cerebral|'
                r'thoracic|abdominal|cervical|lumbar|cranial)\b',
            ],
        }

    def compile_patterns(self):

        self.compiled_patterns = {}
        for entity_type, patterns in self.patterns.items():
            self.compiled_patterns[entity_type] = [
                re.compile(pattern, re.IGNORECASE) for pattern in patterns
            ]

    def classify(self, text: str) -> Tuple[Optional[EntityType], float]:

        text_lower = text.lower().strip()


        for entity_type, patterns in self.compiled_patterns.items():
            for pattern in patterns:
                if pattern.search(text_lower):

                    confidence = 0.95 if pattern.fullmatch(text_lower) else 0.85
                    return entity_type, confidence

        return None, 0.0



class UMLSClient:

    def __init__(self, api_key: str, use_cache: bool = True):
        self.api_key = api_key
        self.use_cache = use_cache
        self.base_url = NERConfig.UMLS_BASE_URL
        self.auth_endpoint = "https://utslogin.nlm.nih.gov/cas/v1/api-key"
        self.search_endpoint = "https://uts-ws.nlm.nih.gov/rest/search/current"


        self.tgt_url = None
        self.tgt_expiry = 0

        if use_cache:
            self._init_cache()

    def _init_cache(self):

        self.cache_path = Path(NERConfig.CACHE_DB_PATH)
        self.conn = sqlite3.connect(str(self.cache_path))
        self.cursor = self.conn.cursor()

        # Create cache table
        self.cursor.execute('''
                            CREATE TABLE IF NOT EXISTS umls_cache
                            (
                                term
                                TEXT
                                PRIMARY
                                KEY,
                                entity_type
                                TEXT,
                                umls_code
                                TEXT,
                                confidence
                                REAL,
                                metadata
                                TEXT,
                                timestamp
                                INTEGER
                            )
                            ''')
        self.conn.commit()

    def _get_cached_result(self, term: str) -> Optional[Dict]:

        if not self.use_cache:
            return None

        term_lower = term.lower().strip()
        self.cursor.execute(
            "SELECT * FROM umls_cache WHERE term = ?", (term_lower,)
        )
        result = self.cursor.fetchone()

        if result:
            timestamp = result[5]
            if time.time() - timestamp < (30 * 24 * 3600):
                return {
                    "entity_type": result[1],
                    "umls_code": result[2],
                    "confidence": result[3],
                    "metadata": json.loads(result[4]) if result[4] else None
                }

        return None

    def _cache_result(self, term: str, result: Dict):

        if not self.use_cache:
            return

        term_lower = term.lower().strip()
        self.cursor.execute('''
            INSERT OR REPLACE INTO umls_cache 
            (term, entity_type, umls_code, confidence, metadata, timestamp)
            VALUES (?, ?, ?, ?, ?, ?)
        ''', (
            term_lower,
            result.get("entity_type", "UNKNOWN"),
            result.get("umls_code"),
            result.get("confidence", 0.0),
            json.dumps(result.get("metadata")),
            int(time.time())
        ))
        self.conn.commit()

    def _get_tgt_url(self) -> Optional[str]:


        if self.tgt_url and time.time() < self.tgt_expiry:
            return self.tgt_url

        try:
            response = requests.post(self.auth_endpoint, data={'apikey': self.api_key})

            if response.status_code == 201:
                self.tgt_url = response.headers.get('location')
                if self.tgt_url:
                    # Cache TGT for 8 hours (UMLS TGTs typically expire in 8 hours)
                    self.tgt_expiry = time.time() + (8 * 3600)
                    logger.debug("Got fresh TGT URL")
                    return self.tgt_url

            logger.error(f"Failed to get TGT: {response.status_code}")
            return None

        except Exception as e:
            logger.error(f"TGT request failed: {e}")
            return None

    def _get_fresh_service_ticket(self) -> Optional[str]:

        tgt_url = self._get_tgt_url()
        if not tgt_url:
            return None

        try:
            service_url = "http://umlsks.nlm.nih.gov"
            response = requests.post(tgt_url, data={'service': service_url})

            if response.status_code == 200:
                ticket = response.text.strip()
                logger.debug(f"Got fresh service ticket: {ticket[:10]}...")
                return ticket

            logger.error(f"Failed to get service ticket: {response.status_code}")
            return None

        except Exception as e:
            logger.error(f"Service ticket request failed: {e}")
            return None

    def _search_umls_concept(self, term: str) -> List[Dict]:

        service_ticket = self._get_fresh_service_ticket()
        if not service_ticket:
            logger.error("Could not get service ticket for search")
            return []

        try:

            params = {
                'ticket': service_ticket,
                'string': term,
                'searchType': 'exact',
                'returnIdType': 'concept',
                'pageNumber': 1,
                'pageSize': 10
            }

            logger.debug(f"Searching UMLS for '{term}' with CUI return type")
            response = requests.get(self.search_endpoint, params=params, timeout=15)

            if response.status_code == 200:
                data = response.json()
                results = data.get('result', {}).get('results', [])

                if results:
                    logger.debug(f"Found {len(results)} exact matches for '{term}'")
                    # Log the first result to verify we're getting CUIs
                    if results:
                        first_result = results[0]
                        logger.debug(f"First result: {first_result.get('name')} (ID: {first_result.get('ui')})")
                    return results


                logger.debug(f"No exact matches for '{term}', trying approximate...")

                service_ticket = self._get_fresh_service_ticket()
                if not service_ticket:
                    return []

                params['ticket'] = service_ticket
                params['searchType'] = 'approximate'

                response = requests.get(self.search_endpoint, params=params, timeout=15)

                if response.status_code == 200:
                    data = response.json()
                    results = data.get('result', {}).get('results', [])
                    logger.debug(f"Found {len(results)} approximate matches for '{term}'")
                    return results

            logger.warning(f"UMLS search failed for '{term}': {response.status_code}")
            logger.debug(f"Response text: {response.text}")
            return []

        except Exception as e:
            logger.error(f"UMLS concept search failed for '{term}': {e}")
            return []

    def _get_concept_details(self, cui: str) -> Dict:

        service_ticket = self._get_fresh_service_ticket()
        if not service_ticket:
            logger.error("Could not get service ticket for concept details")
            return {}

        try:

            if cui.startswith('C') and len(cui) == 8:

                concept_endpoint = f"https://uts-ws.nlm.nih.gov/rest/content/current/CUI/{cui}"
            else:

                logger.debug(f"'{cui}' doesn't look like a standard CUI, trying source lookup")

                return {}

            params = {'ticket': service_ticket}
            response = requests.get(concept_endpoint, params=params, timeout=10)

            if response.status_code == 200:
                data = response.json()
                result = data.get('result', {})
                logger.debug(f"Got concept details for {cui}")
                return result

            logger.warning(f"Failed to get concept details for {cui}: {response.status_code}")
            return {}

        except Exception as e:
            logger.error(f"Concept details request failed for {cui}: {e}")
            return {}

    def _map_semantic_type_to_entity(self, semantic_types: List[str]) -> Tuple[str, float]:

        semantic_mapping = {

            'T047': 'DISEASE',  # Disease or Syndrome
            'T046': 'DISEASE',  # Pathologic Function
            'T048': 'DISEASE',  # Mental or Behavioral Dysfunction
            'T049': 'DISEASE',  # Cell or Molecular Dysfunction
            'T019': 'DISEASE',  # Congenital Abnormality
            'T020': 'DISEASE',  # Acquired Abnormality
            'T037': 'DISEASE',  # Injury or Poisoning
            'T050': 'DISEASE',  # Experimental Model of Disease


            'T184': 'SYMPTOM',  # Sign or Symptom - CHANGED TO DISEASE for diabetes
            'T033': 'DISEASE',  # Finding - can be disease or symptom, try DISEASE first
            'T034': 'SYMPTOM',  # Laboratory or Test Result
            'T201': 'SYMPTOM',  # Clinical Attribute


            'T121': 'MEDICATION',  # Pharmacologic Substance
            'T195': 'MEDICATION',  # Antibiotic
            'T200': 'MEDICATION',  # Drug Delivery Device
            'T203': 'MEDICATION',  # Drug Delivery Device
            'T122': 'MEDICATION',  # Biomedical or Dental Material
            'T103': 'MEDICATION',  # Chemical
            'T109': 'MEDICATION',  # Organic Chemical
            'T114': 'MEDICATION',  # Nucleic Acid, Nucleoside, or Nucleotide
            'T115': 'MEDICATION',  # Organophosphorus Compound
            'T116': 'MEDICATION',  # Amino Acid, Peptide, or Protein
            'T118': 'MEDICATION',  # Carbohydrate
            'T119': 'MEDICATION',  # Lipid
            'T120': 'MEDICATION',  # Chemical Viewed Functionally
            'T125': 'MEDICATION',  # Hormone
            'T126': 'MEDICATION',  # Enzyme
            'T127': 'MEDICATION',  # Vitamin
            'T129': 'MEDICATION',  # Immunologic Factor
            'T130': 'MEDICATION',  # Indicator, Reagent, or Diagnostic Aid
            'T131': 'MEDICATION',  # Hazardous or Poisonous Substance


            'T017': 'ANATOMY',  # Anatomical Structure
            'T029': 'ANATOMY',  # Body Location or Region
            'T023': 'ANATOMY',  # Body Part, Organ, or Organ Component
            'T030': 'ANATOMY',  # Body Space or Junction
            'T031': 'ANATOMY',  # Body System
            'T022': 'ANATOMY',  # Body System
            'T025': 'ANATOMY',  # Cell
            'T026': 'ANATOMY',  # Cell Component
            'T018': 'ANATOMY',  # Embryonic Structure
            'T021': 'ANATOMY',  # Fully Formed Anatomical Structure
            'T024': 'ANATOMY',  # Tissue
            'T028': 'ANATOMY',  # Gene or Genome
        }


        logger.debug(f"Available semantic mappings: {list(semantic_mapping.keys())}")


        confidence_scores = {
            'DISEASE': 0.0,
            'SYMPTOM': 0.0,
            'MEDICATION': 0.0,
            'ANATOMY': 0.0
        }

        for sem_type in semantic_types:
            logger.debug(f"Processing semantic type: '{sem_type}'")
            if sem_type in semantic_mapping:
                entity_type = semantic_mapping[sem_type]
                confidence_scores[entity_type] += 0.9
                logger.debug(f"Mapped '{sem_type}' to '{entity_type}' (+0.9 confidence)")
            else:
                logger.debug(f"No mapping found for semantic type: '{sem_type}'")

        logger.debug(f"Final confidence scores: {confidence_scores}")


        best_type = max(confidence_scores, key=confidence_scores.get)
        best_confidence = confidence_scores[best_type]

        logger.debug(f"Selected: {best_type} with confidence {best_confidence}")

        if best_confidence > 0:
            return best_type, min(best_confidence, 0.95)
        else:
            logger.warning("No semantic types matched - returning UNKNOWN")
            return 'UNKNOWN', 0.5

    def _calculate_name_similarity(self, term1: str, term2: str) -> float:

        if term1 == term2:
            return 1.0

        if term1 in term2 or term2 in term1:
            return 0.9

        # Check for common words
        words1 = set(term1.split())
        words2 = set(term2.split())

        if words1 & words2:
            overlap = len(words1 & words2)
            total = len(words1 | words2)
            return 0.7 + (overlap / total) * 0.2

        return 0.6

    def lookup_term(self, term: str) -> Dict:

        # Check cache first
        cached = self._get_cached_result(term)
        if cached:
            logger.debug(f"Cache hit for term: {term}")
            return cached

        # Get API key from environment
        api_key = os.environ.get("UMLS_API_KEY")
        if not api_key:
            logger.warning("UMLS API key not found in environment variables")
            return {
                "entity_type": "UNKNOWN",
                "umls_code": None,
                "confidence": 0.0,
                "metadata": {"error": "No API key"}
            }

        try:
            logger.debug(f"Starting UMLS lookup for '{term}'")

            # Search for the concept
            search_results = self._search_umls_concept(term)
            if not search_results:
                logger.info(f"No UMLS results found for term: {term}")
                result = {
                    "entity_type": "UNKNOWN",
                    "umls_code": None,
                    "confidence": 0.3,
                    "metadata": {"umls_search": "no_results"}
                }
                self._cache_result(term, result)
                return result

            # Process the best result
            best_result = search_results[0]
            cui = best_result.get('ui', '')
            concept_name = best_result.get('name', '')

            logger.debug(f"Found concept: {concept_name} (CUI: {cui})")

            # Get detailed concept information
            concept_details = self._get_concept_details(cui)
            semantic_types = []

            if concept_details:

                sem_types = concept_details.get('semanticTypes', [])

                logger.debug(f"Raw concept details keys: {list(concept_details.keys())}")


                # semantic_types = [st.get('abbreviation', '') for st in sem_types]
                logger.debug(f"Full semanticTypes data: {sem_types}")
                semantic_types = []
                for st in sem_types:
                    uri = st.get('uri', '')
                    if uri:
                        # Extract T-code from URI like: https://uts-ws.nlm.nih.gov/rest/semantic-network/2025AA/TUI/T047
                        if '/TUI/' in uri:
                            t_code = uri.split('/TUI/')[-1]  # Get the part after /TUI/
                            semantic_types.append(t_code)
                            logger.debug(f"Extracted T-code: {t_code} from URI: {uri}")
                        else:
                            logger.debug(f"No TUI in URI: {uri}")
                    else:
                        logger.debug("No URI found in semantic type")

                logger.debug(f"Final extracted T-codes: {semantic_types}")

            # Map semantic types to our entity types
            entity_type, base_confidence = self._map_semantic_type_to_entity(semantic_types)

            # Adjust confidence based on search match quality
            name_similarity = self._calculate_name_similarity(term.lower(), concept_name.lower())
            final_confidence = base_confidence * name_similarity

            # Create result
            result = {
                "entity_type": entity_type,
                "umls_code": cui,
                "confidence": round(final_confidence, 3),
                "metadata": {
                    "concept_name": concept_name,
                    "semantic_types": semantic_types,
                    "name_similarity": round(name_similarity, 3),
                    "umls_search": "success"
                }
            }

            # Cache the result
            self._cache_result(term, result)

            logger.info(f"UMLS lookup successful for '{term}': {entity_type} ({final_confidence:.3f})")
            return result

        except Exception as e:
            logger.error(f"UMLS API error for term '{term}': {e}")
            result = {
                "entity_type": "UNKNOWN",
                "umls_code": None,
                "confidence": 0.0,
                "metadata": {"error": str(e)}
            }
            return result


class MedicalNERPipeline:

    def __init__(self, config: Optional[NERConfig] = None):
        self.config = config or NERConfig()

        logger.info("Initializing Medical NER Pipeline...")

        # Initialize components
        self._init_biobert()
        self.rule_classifier = RuleBasedClassifier()
        self.umls_client = UMLSClient(
            self.config.UMLS_API_KEY,
            self.config.USE_CACHE
        )

        logger.success("Medical NER Pipeline initialized successfully!")

    def _init_biobert(self):

        logger.info(f"Loading BioBERT model: {self.config.BIOBERT_MODEL}")

        self.tokenizer = AutoTokenizer.from_pretrained(self.config.BIOBERT_MODEL)


        try:
            self.ner_pipeline = pipeline(
                "token-classification",
                model=self.config.BIOBERT_MODEL,
                tokenizer=self.tokenizer,
                aggregation_strategy="simple",
                device=0 if torch.cuda.is_available() else -1
            )
        except:
            # Fallback: Use base model for embeddings
            logger.warning("Token classification not available, using base BioBERT")
            from transformers import AutoModel
            self.model = AutoModel.from_pretrained(self.config.BIOBERT_MODEL)
            self.ner_pipeline = None

    def extract_entities(self, text: str) -> List[MedicalEntity]:

        logger.info(f"Processing text ({len(text)} chars)...")

        # Step 1: Preprocess text
        preprocessed_text = self._preprocess_text(text)

        # Step 2: Extract entity candidates using BioBERT
        entity_candidates = self._extract_entity_candidates(preprocessed_text)

        # Step 3: Classify entities using hybrid approach
        classified_entities = self._classify_entities(entity_candidates)

        # Step 4: Post-process and structure entities
        final_entities = self._postprocess_entities(classified_entities, text)

        # Step 5: Filter by confidence thresholds
        filtered_entities = self._filter_by_confidence(final_entities)

        logger.info(f"Extracted {len(filtered_entities)} entities")

        return filtered_entities

    def _preprocess_text(self, text: str) -> str:

        text = re.sub(r'\s+', ' ', text)

        text = re.sub(r'(\d+)([a-zA-Z]+)', r'\1 \2', text)

        abbreviations = {
            'pt': 'patient',
            'hx': 'history',
            'dx': 'diagnosis',
            'tx': 'treatment',
            'rx': 'prescription',
            'sx': 'symptoms',
            'c/o': 'complains of',
            'w/': 'with',
            'w/o': 'without',
            'yo': 'year old',
            'y/o': 'year old',
        }

        for abbr, full in abbreviations.items():
            text = re.sub(r'\b' + abbr + r'\b', full, text, flags=re.IGNORECASE)

        return text

    def _extract_entity_candidates(self, text: str) -> List[Dict]:

        candidates = []

        if self.ner_pipeline:

            try:
                entities = self.ner_pipeline(text)
                for ent in entities:
                    candidates.append({
                        "text": ent["word"],
                        "start": ent["start"],
                        "end": ent["end"],
                        "score": ent["score"]
                    })
            except Exception as e:
                logger.warning(f"BioBERT NER failed: {e}, using fallback")


        patterns = [
            # Medications with dosages
            (r'\b(\w+)\s+(\d+\s?mg)\b', 0.85),
            # Diseases and conditions
            (r'\b(diabetes|hypertension|infection|infarction)\b', 0.90),
            # Symptoms
            (r'\b(fever|headache|pain|stiffness|nausea)\b', 0.85),
            # Dosage patterns
            (r'\b\d+\s?(mg|ml|mcg|daily|twice daily|BID)\b', 0.80),
            # Medical terms with suffixes
            (r'\b\w+(itis|osis|emia|oma|pathy)\b', 0.75),
            # Common medications
            (r'\b(metformin|aspirin|atorvastatin|metoprolol|ibuprofen|lisinopril)\b', 0.90),
        ]

        for pattern, base_score in patterns:
            for match in re.finditer(pattern, text, re.IGNORECASE):
                candidates.append({
                    "text": match.group(),
                    "start": match.start(),
                    "end": match.end(),
                    "score": base_score
                })

        # Remove duplicates
        seen = set()
        unique_candidates = []
        for cand in candidates:
            key = (cand["text"].lower(), cand["start"])
            if key not in seen:
                seen.add(key)
                unique_candidates.append(cand)

        return unique_candidates

    def _classify_entities(self, candidates: List[Dict]) -> List[MedicalEntity]:

        classified = []

        for cand in candidates:
            text = cand["text"]

            # Step 1: Try rule-based classification
            entity_type, confidence = self.rule_classifier.classify(text)

            # Step 2: If no rule match or low confidence, try UMLS
            if entity_type is None or confidence < 0.8:
                if self.config.ENABLE_UMLS:
                    umls_result = self.umls_client.lookup_term(text)

                    if umls_result["confidence"] > confidence:
                        entity_type = EntityType[umls_result["entity_type"]]
                        confidence = umls_result["confidence"]
                        umls_code = umls_result.get("umls_code")
                    else:
                        umls_code = None
                else:
                    umls_code = None
            else:
                umls_code = None

            # Create medical entity
            if entity_type:
                entity = MedicalEntity(
                    text=text,
                    type=entity_type,
                    start_pos=cand["start"],
                    end_pos=cand["end"],
                    confidence=confidence * cand["score"],  # Combine confidences
                    umls_code=umls_code
                )
                classified.append(entity)

        return classified

    def _postprocess_entities(self, entities: List[MedicalEntity], original_text: str) -> List[MedicalEntity]:

        for entity in entities:
            # Normalize medication names
            if entity.type == EntityType.MEDICATION:
                entity.normalized_text = self._normalize_drug_name(entity.text)

            # Link dosages to medications
            if entity.type == EntityType.DOSAGE:
                # Find nearest medication
                med_entity = self._find_nearest_medication(entity, entities)
                if med_entity:
                    entity.metadata = {"linked_medication": med_entity.text}

            # Add context from original text
            context_start = max(0, entity.start_pos - 20)
            context_end = min(len(original_text), entity.end_pos + 20)
            entity.metadata = entity.metadata or {}
            entity.metadata["context"] = original_text[context_start:context_end]

        return entities

    def _normalize_drug_name(self, drug_name: str) -> str:

        # Common brand to generic mappings
        brand_to_generic = {
            "tylenol": "acetaminophen",
            "advil": "ibuprofen",
            "motrin": "ibuprofen",
            "lipitor": "atorvastatin",
            "zocor": "simvastatin",
            "prilosec": "omeprazole",
            "nexium": "esomeprazole",
        }

        drug_lower = drug_name.lower()
        return brand_to_generic.get(drug_lower, drug_name)

    def _find_nearest_medication(self, dosage_entity: MedicalEntity, all_entities: List[MedicalEntity]) -> Optional[MedicalEntity]:

        medications = [e for e in all_entities if e.type == EntityType.MEDICATION]

        if not medications:
            return None

        # Find medication with minimum distance
        min_distance = float('inf')
        nearest_med = None

        for med in medications:
            distance = abs(dosage_entity.start_pos - med.end_pos)
            if distance < min_distance and distance < 50:  # Within 50 characters
                min_distance = distance
                nearest_med = med

        return nearest_med

    def _filter_by_confidence(self, entities: List[MedicalEntity]) -> List[MedicalEntity]:

        filtered = []

        for entity in entities:
            if entity.confidence >= self.config.AUTO_ACCEPT_THRESHOLD:
                entity.metadata = entity.metadata or {}
                entity.metadata["review_status"] = "auto_accepted"
                filtered.append(entity)

            elif entity.confidence >= self.config.REVIEW_THRESHOLD:
                entity.metadata = entity.metadata or {}
                entity.metadata["review_status"] = "needs_review"
                filtered.append(entity)

            else:
                # Rejected - confidence too low
                logger.debug(f"Rejected entity '{entity.text}' (confidence: {entity.confidence:.2f})")

        return filtered

    def process_document(self, text: str) -> Dict[str, Any]:

        start_time = time.time()

        # Extract entities
        entities = self.extract_entities(text)

        # Group entities by type
        grouped_entities = {
            "diseases": [],
            "symptoms": [],
            "medications": [],
            "dosages": [],
            "anatomy": [],
            "unclassified": []
        }

        # Structure medications with their dosages
        medications_with_dosage = {}

        for entity in entities:
            entity_dict = entity.to_dict()

            if entity.type == EntityType.DISEASE:
                grouped_entities["diseases"].append(entity_dict)
            elif entity.type == EntityType.SYMPTOM:
                grouped_entities["symptoms"].append(entity_dict)
            elif entity.type == EntityType.MEDICATION:
                # Check if there's a linked dosage
                med_key = entity.text
                medications_with_dosage[med_key] = {
                    "name": entity.text,
                    "normalized_name": entity.normalized_text,
                    "confidence": entity.confidence,
                    "position": {"start": entity.start_pos, "end": entity.end_pos},
                    "dosage": None,
                    "umls_code": entity.umls_code
                }
            elif entity.type == EntityType.DOSAGE:
                # Link to medication if possible
                linked_med = entity.metadata.get("linked_medication") if entity.metadata else None
                if linked_med and linked_med in medications_with_dosage:
                    medications_with_dosage[linked_med]["dosage"] = entity.text
                else:
                    grouped_entities["dosages"].append(entity_dict)
            elif entity.type == EntityType.ANATOMY:
                grouped_entities["anatomy"].append(entity_dict)
            else:
                grouped_entities["unclassified"].append(entity_dict)

        # Add structured medications to output
        grouped_entities["medications"] = list(medications_with_dosage.values())

        # Calculate processing time
        processing_time = time.time() - start_time

        # Create final output
        output = {
            "status": "success",
            "processing_time_ms": round(processing_time * 1000, 2),
            "text_length": len(text),
            "total_entities": len(entities),
            "entities": grouped_entities,
            "statistics": {
                "diseases": len(grouped_entities["diseases"]),
                "symptoms": len(grouped_entities["symptoms"]),
                "medications": len(grouped_entities["medications"]),
                "auto_accepted": sum(1 for e in entities if e.metadata and e.metadata.get("review_status") == "auto_accepted"),
                "needs_review": sum(1 for e in entities if e.metadata and e.metadata.get("review_status") == "needs_review")
            },
            "confidence_summary": {
                "high": sum(1 for e in entities if e.confidence >= 0.9),
                "medium": sum(1 for e in entities if 0.7 <= e.confidence < 0.9),
                "low": sum(1 for e in entities if e.confidence < 0.7)
            }
        }

        return output



class MedicalNLPAgent:


    def __init__(self, umls_api_key: Optional[str] = None):

        config = NERConfig()
        if umls_api_key:
            config.UMLS_API_KEY = umls_api_key

        self.pipeline = MedicalNERPipeline(config)
        logger.info("Medical NLP Agent ready for CrewAI integration")

    def extract_medical_entities(self, text: str) -> str:

        result = self.pipeline.process_document(text)

        # Format for CrewAI agent communication
        output = "Medical Entity Extraction Results:\n\n"

        if result["entities"]["diseases"]:
            output += "DISEASES IDENTIFIED:\n"
            for disease in result["entities"]["diseases"]:
                output += f"- {disease['text']} (confidence: {disease['confidence']:.2f})\n"
            output += "\n"

        if result["entities"]["symptoms"]:
            output += "SYMPTOMS IDENTIFIED:\n"
            for symptom in result["entities"]["symptoms"]:
                output += f"- {symptom['text']} (confidence: {symptom['confidence']:.2f})\n"
            output += "\n"

        if result["entities"]["medications"]:
            output += "MEDICATIONS IDENTIFIED:\n"
            for med in result["entities"]["medications"]:
                dosage_info = f" - Dosage: {med['dosage']}" if med.get('dosage') else ""
                output += f"- {med['name']}{dosage_info} (confidence: {med['confidence']:.2f})\n"
            output += "\n"

        output += f"Total entities extracted: {result['total_entities']}\n"
        output += f"Processing time: {result['processing_time_ms']}ms\n"

        return output

    def get_structured_entities(self, text: str) -> Dict:
        return self.pipeline.process_document(text)

