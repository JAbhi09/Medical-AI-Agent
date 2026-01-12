from crewai import Agent, LLM
from typing import Dict, Any
from loguru import logger
from src.pipelines.medical_ner_pipeline import MedicalNLPAgent, NERConfig
from tenacity import retry, stop_after_attempt, wait_exponential
import time
from dotenv import load_dotenv
load_dotenv()
from crewai import Crew, Task, Process


class Enhanced_Medical_NLP_Agent:


    def __init__(self, umls_api_key: str = None, use_gemini: bool = True):

        # Initialize the NER pipeline
        self.ner_pipeline = MedicalNLPAgent(umls_api_key=umls_api_key)

        if self.ner_pipeline.pipeline.umls_client.api_key:
            logger.info(f"UMLS configured successfully")
        else:
            logger.warning("Running without UMLS (rule-based only)")

        # Set up the LLM for CrewAI
        if use_gemini:
            self.llm = "gemini/gemini-2.0-flash-lite"
    #     else:
    #         self.llm = LLM(
    #     model="ollama/qwen2:1.5b",
    #     base_url="http://localhost:11434",
    #     temperature=0.3,
    #     timeout=120
    # )
            # from langchain_community.llms import Ollama
            # self.llm = Ollama(model="llama3.1", temperature=0.3)

        logger.info("Enhanced Medical NLP Agent initialized with BioBERT NER")

    def create_crew_agent(self) -> Agent:

        return Agent(
            role="Medical NLP Specialist with BioBERT",
            goal="Extract and categorize medical entities with high accuracy using BioBERT and medical knowledge bases",
            backstory="""You are an advanced medical NLP specialist equipped with BioBERT 
            for medical entity recognition. You can identify diseases, symptoms, medications, 
            dosages, and anatomical terms with high precision. You use a combination of 
            deep learning models and medical knowledge bases (UMLS) to ensure accurate 
            entity extraction and classification. Your role is critical in understanding 
            medical texts and providing structured information to other agents.""",
            verbose=True,
            allow_delegation=False,
            llm=self.llm,
            max_iter=1,
            max_rpm=5
        )
    
    @retry(
        stop=stop_after_attempt(2),
        wait=wait_exponential(multiplier=1, min=30, max=60)
    )
    def process_with_retry(self, crew, inputs=None):

        try:
            if inputs:
                return crew.kickoff(inputs=inputs)
            else:
                return crew.kickoff()
        except Exception as e:
            if "429" in str(e) or "rate limit" in str(e).lower():
                logger.warning(f"Rate limit hit, waiting...")
                time.sleep(35)
                raise
            raise

    def process_medical_text(self, text: str) -> Dict[str, Any]:

        # Use the NER pipeline to extract entities
        result = self.ner_pipeline.get_structured_entities(text)

        # Enhance the result with additional analysis if needed
        result["summary"] = self._generate_summary(result)

        return result

    def get_formatted_output(self, text: str) -> str:

        return self.ner_pipeline.extract_medical_entities(text)

    def _generate_summary(self, result: Dict) -> str:
        """Generate a summary of extracted entities"""
        summary = []

        if result["statistics"]["diseases"] > 0:
            summary.append(f"{result['statistics']['diseases']} disease(s)")
        if result["statistics"]["symptoms"] > 0:
            summary.append(f"{result['statistics']['symptoms']} symptom(s)")
        if result["statistics"]["medications"] > 0:
            summary.append(f"{result['statistics']['medications']} medication(s)")

        if summary:
            return f"Extracted: {', '.join(summary)}"
        else:
            return "No medical entities found"



class EnhancedMedicalCrewMVP:

    def __init__(self, umls_api_key: str = None, max_rpm : int = 30):

        # Initialize the enhanced NLP agent
        self.max_rpm = max_rpm
        self.nlp_agent_handler = Enhanced_Medical_NLP_Agent(
            use_gemini=True
        )

        # Create all agents including the enhanced NLP agent
        self.agents = self._create_agents()

        logger.info("Medical Crew initialized with BioBERT NER")



    def _create_agents(self) -> Dict[str, Agent]:
        
        agents = {}

        # Enhanced NLP Agent with BioBERT
        agents["medical_nlp"] = self.nlp_agent_handler.create_crew_agent()

        # Other agents (using your existing setup)
        llm = "gemini/gemini-2.0-flash-lite"
#         llm = LLM(
#     model="ollama/qwen2:1.5b",
#     base_url="http://localhost:11434",
#     temperature=0.3,
#     timeout=120
# )

        agents["clinical_reasoning"] = Agent(
            role="Clinical Reasoning Specialist",
            goal="Analyze patient symptoms and suggest possible diagnoses",
            backstory="""You are a medical doctor with 10 years of experience 
            in internal medicine. You excel at differential diagnosis and 
            clinical reasoning. You work with the Medical NLP Specialist to 
            understand the extracted entities and provide clinical insights.""",
            verbose=True,
            allow_delegation=False,
            llm=llm,
            max_iter=1,
            max_rpm = 5
        )

        agents["drug_interaction"] = Agent(
            role="Pharmacology Specialist",
            goal="Identify potential drug interactions and contraindications",
            backstory="""You are a clinical pharmacist with expertise in drug 
            interactions. You analyze medications identified by the NLP agent 
            and check for potential interactions and safety concerns.""",
            verbose=True,
            allow_delegation=False,
            llm=llm,
            max_iter=1,
            max_rpm=5
        )

        agents["knowledge_validation"] = Agent(
            role="Medical Knowledge Validator",
            goal="Verify medical facts and flag potential errors",
            backstory="""You ensure medical accuracy by validating claims against 
            established medical knowledge. You review entities extracted by the 
            NLP agent and verify their accuracy.""",
            verbose=True,
            allow_delegation=False,
            llm=llm,
            max_iter=1,
            max_rpm=5
        )

        agents["patient_education"] = Agent(
            role="Patient Education Specialist",
            goal="Create clear, simple explanations for patients",
            backstory="""You translate complex medical information into 
            easy-to-understand language for patients.""",
            verbose=True,
            allow_delegation=False,
            llm=llm,
            max_iter=1,
            max_rpm=5
        )

        return agents

    
    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=30, max=60)
    )
    def process_with_retry(self, crew, inputs=None):
        """Process crew kickoff with retry logic for rate limiting"""
        try:
            if inputs:
                return crew.kickoff(inputs=inputs)
            else:
                return crew.kickoff()
        except Exception as e:
            if "429" in str(e) or "rate limit" in str(e).lower():
                logger.warning(f"Rate limit hit, waiting...")
                time.sleep(35)
                raise
            raise

    def process_medical_query_enhanced(self, patient_input: str) -> Dict[str, Any]:

        logger.info(f"Processing query with enhanced NER: {patient_input[:100]}...")

        # First, extract entities using BioBERT NER
        ner_result = self.nlp_agent_handler.process_medical_text(patient_input)

        # Format entities for task context
        entity_context = self.nlp_agent_handler.get_formatted_output(patient_input)

        # Task 1: Enhanced NLP extraction 
        nlp_task = Task(
            description=f"""You have already extracted medical entities using BioBERT.
            Here are the results:

            {entity_context}

            Provide a summary of the key medical entities found.""",
            agent=self.agents["medical_nlp"],
            expected_output="Summary of extracted medical entities"
        )

        # Task 2: Clinical reasoning based on extracted entities
        reasoning_task = Task(
            description=f"""Based on the extracted medical entities:

            {entity_context}

            Original text: {patient_input}

            Provide:
            1. Top 3 possible diagnoses based on the identified symptoms and conditions
            2. Clinical significance of the identified medications
            3. Any red flags requiring immediate attention""",
            agent=self.agents["clinical_reasoning"],
            expected_output="Clinical assessment with differential diagnosis",
            context=[nlp_task]
        )

        # Task 3: Drug interaction check for identified medications
        drug_task = Task(
            description=f"""Review the medications identified:

            Medications found: {ner_result['entities']['medications']}

            Check for:
            1. Potential drug-drug interactions
            2. Contraindications based on identified conditions
            3. Appropriate dosing concerns""",
            agent=self.agents["drug_interaction"],
            expected_output="Drug safety assessment"
        )

        # Task 4: Validate extracted entities and clinical reasoning
        validation_task = Task(
            description=f"""Validate the medical information:

            Extracted Entities:
            - Diseases: {ner_result['statistics']['diseases']} found
            - Symptoms: {ner_result['statistics']['symptoms']} found
            - Medications: {ner_result['statistics']['medications']} found

            Entities needing review: {ner_result['statistics']['needs_review']}

            Verify the accuracy and flag any concerns.""",
            agent=self.agents["knowledge_validation"],
            expected_output="Validation report with confidence levels",
            context=[nlp_task, reasoning_task, drug_task]
        )

        # Task 5: Patient education
        education_task = Task(
            description="""Create a patient-friendly summary that:
            1. Explains the identified conditions in simple terms
            2. Clarifies medication purposes
            3. Provides clear next steps

            Use simple language, no medical jargon.""",
            agent=self.agents["patient_education"],
            expected_output="Patient education summary",
            context=[reasoning_task, validation_task]
        )

        # Create and run the crew
        crew = Crew(
            agents=list(self.agents.values()),
            tasks=[nlp_task, reasoning_task, drug_task, validation_task, education_task],
            process=Process.sequential,
            verbose=True,
            max_rpm = 30
        )

        try:
            crew_result = self.process_with_retry(crew)

            # Combine NER results with crew analysis
            output = {
                "status": "success",
                "patient_input": patient_input,
                "ner_extraction": ner_result,
                "clinical_analysis": {
                    "entities_summary": str(nlp_task.output.raw) if hasattr(nlp_task.output, 'raw') else str(
                        nlp_task.output),
                    "clinical_reasoning": str(reasoning_task.output.raw) if hasattr(reasoning_task.output,
                                                                                    'raw') else str(
                        reasoning_task.output),
                    "drug_safety": str(drug_task.output.raw) if hasattr(drug_task.output, 'raw') else str(
                        drug_task.output),
                    "validation": str(validation_task.output.raw) if hasattr(validation_task.output, 'raw') else str(
                        validation_task.output),
                    "patient_summary": str(education_task.output.raw) if hasattr(education_task.output, 'raw') else str(
                        education_task.output)
                },
                "metrics": {
                    "processing_time_ms": ner_result["processing_time_ms"],
                    "entities_extracted": ner_result["total_entities"],
                    "confidence": {
                        "high": ner_result["confidence_summary"]["high"],
                        "medium": ner_result["confidence_summary"]["medium"],
                        "low": ner_result["confidence_summary"]["low"]
                    }
                },
                "requires_human_review": ner_result["statistics"]["needs_review"] > 0
            }

            logger.success(
                f"Medical query processed successfully with {ner_result['total_entities']} entities extracted")
            return output

        except Exception as e:
            logger.error(f"Error processing medical query: {e}")
            return {
                "status": "error",
                "error_message": str(e),
                "ner_extraction": ner_result,  # Still return NER results
                "requires_human_review": True
            }


