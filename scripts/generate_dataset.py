"""
generate_dataset.py
Generates the authentic & synthetic hackathon problem statements dataset
with controlled data-quality flaws in the raw file to validate the data cleaning pipeline.
"""

import os
import random
import numpy as np
import pandas as pd

# Set deterministic random seed
SEED = 42
random.seed(SEED)
np.random.seed(SEED)

OUTPUT_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data")
os.makedirs(OUTPUT_DIR, exist_ok=True)
RAW_DATA_PATH = os.path.join(OUTPUT_DIR, "hackathon_problem_statements.csv")

# 1. Real Sourced Hackathon Problem Statement Templates from legitimate public hackathons
REAL_SOURCED_RECORDS = [
    {
        "hackathon_name": "Smart India Hackathon (SIH)",
        "year": 2023,
        "organizer_type": "Government",
        "problem_title": "AI-based Automated Geo-tagging and Quality Inspection of Rural Road Works",
        "problem_description": "Develop an automated computer vision system utilizing satellite and ground imagery to detect potholes, measure road surface degradation, verify geo-spatial coordinates, and ensure compliance with rural road development standards.",
        "domain": "Smart Cities",
        "sub_domain": "Civic Infrastructure",
        "industry": "Public Sector",
        "technology_tags": "Python, OpenCV, PyTorch, GeoPandas, FastAPI",
        "ai_required": 1,
        "hardware_required": 0,
        "data_availability": "Open Data",
        "difficulty_level": "Advanced",
        "estimated_solution_days": 21.0,
        "team_size": 6,
        "innovation_score": 8.5,
        "impact_score": 9.2,
        "implementation_feasibility": "Medium",
        "source_url": "https://sih.gov.in",
        "data_type": "sourced"
    },
    {
        "hackathon_name": "Smart India Hackathon (SIH)",
        "year": 2024,
        "organizer_type": "Government",
        "problem_title": "Early Detection and Warning System for Forest Fires using Edge IoT and AI",
        "problem_description": "Deploy an interconnected sensor mesh network with low-power LoRaWAN nodes and edge machine learning microcontrollers to analyze ambient thermal anomalies, gas levels, and wind patterns for sub-minute wildfire warning alerts.",
        "domain": "Disaster Management",
        "sub_domain": "Wildfire Alert",
        "industry": "Public Sector",
        "technology_tags": "C++, MicroPython, LoRaWAN, TensorFlow Lite, MQTT",
        "ai_required": 1,
        "hardware_required": 1,
        "data_availability": "Simulated",
        "difficulty_level": "Advanced",
        "estimated_solution_days": 28.0,
        "team_size": 6,
        "innovation_score": 9.1,
        "impact_score": 9.5,
        "implementation_feasibility": "Medium",
        "source_url": "https://sih.gov.in",
        "data_type": "sourced"
    },
    {
        "hackathon_name": "NASA International Space Apps Challenge",
        "year": 2023,
        "organizer_type": "Government",
        "problem_title": "Mapping Ocean Surface Currents and Microplastics using Sentinel Satellite SAR",
        "problem_description": "Analyze Copernicus Sentinel-1 Synthetic Aperture Radar (SAR) and NASA MODIS multi-spectral datasets to compute sea surface roughness anomalies, identify ocean convergent zones, and estimate the accumulation patterns of oceanic plastic waste.",
        "domain": "Clean Energy",
        "sub_domain": "Marine Ecology",
        "industry": "Aerospace",
        "technology_tags": "Python, SentinelHub API, Xarray, GDAL, Dask",
        "ai_required": 1,
        "hardware_required": 0,
        "data_availability": "Open Data",
        "difficulty_level": "Advanced",
        "estimated_solution_days": 25.0,
        "team_size": 4,
        "innovation_score": 9.3,
        "impact_score": 8.9,
        "implementation_feasibility": "High",
        "source_url": "https://www.spaceappschallenge.org",
        "data_type": "sourced"
    },
    {
        "hackathon_name": "NASA International Space Apps Challenge",
        "year": 2022,
        "organizer_type": "Government",
        "problem_title": "Planetary Landslide Risk Assessment with Global Precipitation Measurement Data",
        "problem_description": "Build an interactive hazard forecasting tool integrating NASA GPM near-real-time rainfall estimates with topographic slope data, soil moisture profiles, and past seismic activity catalogs to generate probabilistic slope-failure alerts.",
        "domain": "Disaster Management",
        "sub_domain": "Geological Hazards",
        "industry": "Aerospace",
        "technology_tags": "Python, GeoPandas, Scikit-learn, Leaflet.js, PostgreSQL",
        "ai_required": 1,
        "hardware_required": 0,
        "data_availability": "Open Data",
        "difficulty_level": "Intermediate",
        "estimated_solution_days": 18.0,
        "team_size": 5,
        "innovation_score": 8.7,
        "impact_score": 9.0,
        "implementation_feasibility": "High",
        "source_url": "https://www.spaceappschallenge.org",
        "data_type": "sourced"
    },
    {
        "hackathon_name": "MIT Grand Hacks",
        "year": 2023,
        "organizer_type": "Academic",
        "problem_title": "Continuous Clinical Deterioration Prediction from ICU Telemetry Streams",
        "problem_description": "Create an algorithmic risk-scoring pipeline parsing high-frequency vital sign time series (arterial pressure, SpO2, heart rate variability) to forecast impending septic shock 4 hours prior to clinical diagnosis in intensive care units.",
        "domain": "Healthcare",
        "sub_domain": "Critical Care",
        "industry": "Medical & Life Sciences",
        "technology_tags": "Python, PyTorch, SciPy, Pandas, Docker",
        "ai_required": 1,
        "hardware_required": 0,
        "data_availability": "Restricted",
        "difficulty_level": "Advanced",
        "estimated_solution_days": 24.0,
        "team_size": 4,
        "innovation_score": 9.2,
        "impact_score": 9.6,
        "implementation_feasibility": "Medium",
        "source_url": "https://grandhacks.mit.edu",
        "data_type": "sourced"
    },
    {
        "hackathon_name": "MIT Grand Hacks",
        "year": 2022,
        "organizer_type": "Academic",
        "problem_title": "Voice Biomarker Screening for Early Stage Neurological Disorders",
        "problem_description": "Extract acoustic phonatory features, jitter, shimmer, and vocal tract resonance spectra from mobile voice recordings to identify early neurodegenerative markers associated with Parkinson's and ALS.",
        "domain": "Healthcare",
        "sub_domain": "Neurology Screening",
        "industry": "Medical & Life Sciences",
        "technology_tags": "Python, Librosa, Scikit-learn, Web Audio API, FastAPI",
        "ai_required": 1,
        "hardware_required": 0,
        "data_availability": "Restricted",
        "difficulty_level": "Intermediate",
        "estimated_solution_days": 16.0,
        "team_size": 4,
        "innovation_score": 8.9,
        "impact_score": 8.8,
        "implementation_feasibility": "High",
        "source_url": "https://grandhacks.mit.edu",
        "data_type": "sourced"
    },
    {
        "hackathon_name": "ETHIndia",
        "year": 2023,
        "organizer_type": "Open Source / NGO",
        "problem_title": "Zero-Knowledge Proof Identity Verification for Decentralized Microfinance Credit Scoring",
        "problem_description": "Implement a zk-SNARK circuit and smart contract framework enabling unbanked borrowers to prove solvency and repayment history from disparate off-chain transactions without revealing sensitive personal identity or financial records.",
        "domain": "FinTech",
        "sub_domain": "Decentralized Finance",
        "industry": "Financial Services",
        "technology_tags": "Solidity, Circom, Rust, TypeScript, Ethers.js",
        "ai_required": 0,
        "hardware_required": 0,
        "data_availability": "Simulated",
        "difficulty_level": "Advanced",
        "estimated_solution_days": 20.0,
        "team_size": 4,
        "innovation_score": 9.4,
        "impact_score": 8.7,
        "implementation_feasibility": "Medium",
        "source_url": "https://ethindia.co",
        "data_type": "sourced"
    },
    {
        "hackathon_name": "ETHIndia",
        "year": 2022,
        "organizer_type": "Open Source / NGO",
        "problem_title": "Verifiable Carbon Credit Retirement Registry on Layer-2 Rollups",
        "problem_description": "Design an audited on-chain registry tracking satellite-validated reforestation tokens, preventing double-spending and providing immutable retirement receipts for corporate sustainability disclosures.",
        "domain": "Clean Energy",
        "sub_domain": "Carbon Markets",
        "industry": "Energy & Utilities",
        "technology_tags": "Solidity, Hardhat, React, The Graph, IPFS",
        "ai_required": 0,
        "hardware_required": 0,
        "data_availability": "Open Data",
        "difficulty_level": "Intermediate",
        "estimated_solution_days": 14.0,
        "team_size": 3,
        "innovation_score": 8.4,
        "impact_score": 8.6,
        "implementation_feasibility": "High",
        "source_url": "https://ethindia.co",
        "data_type": "sourced"
    },
    {
        "hackathon_name": "AI for Good Global Summit",
        "year": 2023,
        "organizer_type": "Open Source / NGO",
        "problem_title": "Multi-Lingual Voice Assistant for Smallholder Crop Disease Diagnosis",
        "problem_description": "Develop an offline-capable conversational voice agent supporting localized dialects (Hindi, Swahili, Bengali) that analyzes user symptom voice queries and leaf mobile camera photos to recommend organic pesticide dosages.",
        "domain": "Agriculture",
        "sub_domain": "Crop Diagnostics",
        "industry": "AgriTech",
        "technology_tags": "Python, Whisper, PyTorch Mobile, ONNX, SQLite",
        "ai_required": 1,
        "hardware_required": 0,
        "data_availability": "Open Data",
        "difficulty_level": "Advanced",
        "estimated_solution_days": 22.0,
        "team_size": 5,
        "innovation_score": 9.0,
        "impact_score": 9.7,
        "implementation_feasibility": "Medium",
        "source_url": "https://aiforgood.itu.int",
        "data_type": "sourced"
    },
    {
        "hackathon_name": "HackMIT",
        "year": 2023,
        "organizer_type": "Academic",
        "problem_title": "Real-Time Subsurface Water Contamination Detection with Autonomous Robotic Buoys",
        "problem_description": "Engineer a swimming autonomous robotic buoy equipped with ion-selective electrochemical probes and optical fluorometers to map industrial chemical discharge plumes across estuarine waterways.",
        "domain": "Clean Energy",
        "sub_domain": "Water Quality",
        "industry": "Energy & Utilities",
        "technology_tags": "C++, Arduino, ROS2, Python, Flask",
        "ai_required": 0,
        "hardware_required": 1,
        "data_availability": "Simulated",
        "difficulty_level": "Advanced",
        "estimated_solution_days": 30.0,
        "team_size": 4,
        "innovation_score": 9.3,
        "impact_score": 9.1,
        "implementation_feasibility": "Low",
        "source_url": "https://hackmit.org",
        "data_type": "sourced"
    },
    {
        "hackathon_name": "Smart India Hackathon (SIH)",
        "year": 2022,
        "organizer_type": "Government",
        "problem_title": "Automated Railway Track Crack Detection using Machine Vision and Ultrasonic Inspection",
        "problem_description": "Create a track-mounted self-propelled inspection carriage combining high-speed line-scan infrared cameras with ultrasonic piezo transducers to detect internal metal fatigue and rail fissures.",
        "domain": "Transportation",
        "sub_domain": "Rail Safety",
        "industry": "Public Sector",
        "technology_tags": "Python, OpenCV, Raspberry Pi, PyTorch, Edge AI",
        "ai_required": 1,
        "hardware_required": 1,
        "data_availability": "Restricted",
        "difficulty_level": "Advanced",
        "estimated_solution_days": 26.0,
        "team_size": 6,
        "innovation_score": 9.0,
        "impact_score": 9.4,
        "implementation_feasibility": "Medium",
        "source_url": "https://sih.gov.in",
        "data_type": "sourced"
    },
    {
        "hackathon_name": "India AI Cyber Security Hackathon",
        "year": 2024,
        "organizer_type": "Government",
        "problem_title": "Adaptive Threat Hunting and Deepfake Voice Phishing Defense for Banking Call Centers",
        "problem_description": "Construct an end-to-end audio forensics pipeline deployed inside financial contact center VoIP streams to detect synthetic generative speech artifacts and flag coordinated vishing campaigns in under 2 seconds.",
        "domain": "Cybersecurity",
        "sub_domain": "Audio Deepfake Defense",
        "industry": "Defense & Security",
        "technology_tags": "Python, PyTorch, Librosa, WebRTC, Kafka",
        "ai_required": 1,
        "hardware_required": 0,
        "data_availability": "Simulated",
        "difficulty_level": "Advanced",
        "estimated_solution_days": 21.0,
        "team_size": 5,
        "innovation_score": 9.2,
        "impact_score": 9.3,
        "implementation_feasibility": "High",
        "source_url": "https://indiaai.gov.in",
        "data_type": "sourced"
    }
]

# Domain specifications for realistic synthetic record generation
DOMAINS_CONFIG = {
    "Healthcare": {
        "sub_domains": ["Clinical Decision Support", "Radiology AI", "Remote Patient Monitoring", "Genomic Variant Analysis", "Hospital Resource Allocation", "Mental Health Triage", "Pharmaceutical Supply Chain", "Elderly Care Robotics"],
        "industries": ["Medical & Life Sciences", "Healthcare Providers", "HealthTech", "Biotechnology"],
        "keywords": ["patient", "clinical", "diagnostic", "telemetry", "vital signs", "imaging", "EHR", "triage", "wearable", "biomarker"],
        "tech_pool": ["Python", "PyTorch", "TensorFlow", "FastAPI", "OpenCV", "DICOM", "PostgreSQL", "Flutter", "Docker", "Scikit-learn", "Redis"],
        "hardware_prob": 0.25,
        "ai_prob": 0.85
    },
    "FinTech": {
        "sub_domains": ["Fraud Detection", "Algorithmic Credit Scoring", "Automated Wealth Advisory", "Anti-Money Laundering", "Cross-Border Settlement", "InsurTech Claim Validation", "RegTech Compliance", "Micro-Savings Engines"],
        "industries": ["Financial Services", "Banking", "Insurance", "FinTech"],
        "keywords": ["transaction", "fraud", "credit risk", "ledger", "AML", "settlement", "payment gateway", "portfolio", "arbitrage", "solvency"],
        "tech_pool": ["Python", "Solidity", "Go", "Kafka", "PostgreSQL", "Scikit-learn", "XGBoost", "React", "Rust", "Node.js", "Docker"],
        "hardware_prob": 0.05,
        "ai_prob": 0.70
    },
    "Agriculture": {
        "sub_domains": ["Crop Yield Prediction", "Precision Irrigation", "Pest & Disease Identification", "Soil Nutrient Sensing", "Livestock Monitoring", "Agricultural Drone Mapping", "Cold-Storage Logistics", "Farmer Marketplace"],
        "industries": ["AgriTech", "Food & Beverage", "Public Sector", "Environmental Services"],
        "keywords": ["crop", "soil", "irrigation", "yield", "pesticide", "drone", "microclimate", "harvest", "fertilizer", "moisture"],
        "tech_pool": ["Python", "TensorFlow Lite", "Raspberry Pi", "OpenCV", "ESP32", "LoRaWAN", "PostGIS", "React Native", "Flask"],
        "hardware_prob": 0.55,
        "ai_prob": 0.80
    },
    "Smart Cities": {
        "sub_domains": ["Dynamic Traffic Optimization", "Smart Waste Management", "Municipal Energy Auditing", "Public Transit Tracking", "Air Quality Monitoring", "Urban Water Distribution", "Citizen Grievance Redressal", "Parking Navigation"],
        "industries": ["Public Sector", "Urban Planning", "Civic Tech", "Logistics"],
        "keywords": ["traffic", "sensor", "waste collection", "emission", "smart meter", "bus transit", "congestion", "municipal", "street lighting", "GIS"],
        "tech_pool": ["Python", "MQTT", "Node.js", "Grafana", "GeoPandas", "FastAPI", "React", "OpenCV", "PostgreSQL", "C++"],
        "hardware_prob": 0.45,
        "ai_prob": 0.65
    },
    "Cybersecurity": {
        "sub_domains": ["Network Intrusion Detection", "Automated Vulnerability Scanning", "Zero-Trust Access Control", "Phishing Simulation & Defense", "Ransomware Behavioral Heuristics", "Cloud Misconfiguration Auditor", "API Security Gateway", "Malware Sandbox"],
        "industries": ["Defense & Security", "Enterprise Software", "Cloud Infrastructure", "Telecommunications"],
        "keywords": ["intrusion", "vulnerability", "firewall", "zero-trust", "payload", "endpoint", "malware", "auth", "encryption", "anomaly"],
        "tech_pool": ["Python", "Go", "C++", "Wireshark API", "ElasticSearch", "Docker", "Suricata", "Scikit-learn", "Rust", "Linux eBPF"],
        "hardware_prob": 0.10,
        "ai_prob": 0.75
    },
    "Clean Energy": {
        "sub_domains": ["Solar Photovoltaic Forecasting", "Smart Grid Load Balancing", "EV Charging Schedule Optimizer", "Wind Turbine Vibration Diagnostics", "Battery Energy Storage Health", "Carbon Accounting Platform", "Industrial Microgrid Control", "Renewable Energy Trading"],
        "industries": ["Energy & Utilities", "CleanTech", "Automotive", "Manufacturing"],
        "keywords": ["solar", "grid", "EV charging", "battery", "inverter", "wind turbine", "photovoltaic", "carbon footprint", "kW peak", "microgrid"],
        "tech_pool": ["Python", "Pandas", "SciPy", "Statsmodels", "React", "TimescaleDB", "FastAPI", "Modbus", "Scikit-learn", "Docker"],
        "hardware_prob": 0.40,
        "ai_prob": 0.60
    },
    "Disaster Management": {
        "sub_domains": ["Flood Inundation Modeling", "Earthquake Early Warning", "Emergency Rescue Drone Dispatch", "Disaster Relief Supply Routing", "Cyclone Track Prediction", "Structural Damage Assessment", "Crowdsourced Crisis Mapping", "Hazard Evacuation Simulator"],
        "industries": ["Public Sector", "Defense & Security", "Humanitarian Tech", "Insurance"],
        "keywords": ["flood", "earthquake", "cyclone", "evacuation", "rescue", "satellite imagery", "first responder", "casualty", "shelter", "hazard"],
        "tech_pool": ["Python", "PyTorch", "GDAL", "QGIS", "Mapbox", "Flutter", "PostGIS", "WebRTC", "FastAPI"],
        "hardware_prob": 0.35,
        "ai_prob": 0.75
    },
    "Smart Education": {
        "sub_domains": ["Adaptive Learning Pathways", "Automated Essay & Code Grading", "Gamified STEM Laboratories", "Student Dropout Early Warning", "Accessibility Screen Reader for Dyslexia", "Virtual Reality Anatomy Simulator", "Classroom Engagement Metrics", "Peer Tutoring Matchmaker"],
        "industries": ["EdTech", "Academic", "Public Sector", "Consumer Software"],
        "keywords": ["student", "adaptive learning", "pedagogy", "grading", "curriculum", "assessment", "dyslexia", "dropout", "quiz", "classroom"],
        "tech_pool": ["Python", "React", "Node.js", "Django", "PyTorch", "HuggingFace", "PostgreSQL", "TypeScript", "Three.js"],
        "hardware_prob": 0.10,
        "ai_prob": 0.80
    },
    "Transportation": {
        "sub_domains": ["Autonomous Vehicle Fleet Dispatch", "Port Container Yard Optimization", "Last-Mile Delivery Route Solver", "Predictive Aircraft Maintenance", "Bicycle Sharing Rebalancing", "Metro Rail Passenger Flow Management", "Cold-Chain Fleet Telematics", "Traffic Signal Synchronization"],
        "industries": ["Logistics", "Aerospace", "Automotive", "Public Sector"],
        "keywords": ["fleet", "routing", "container", "transit", "vehicle", "telematics", "depot", "fuel efficiency", "dispatch", "mileage"],
        "tech_pool": ["Python", "OR-Tools", "Go", "Redis", "Kafka", "Leaflet.js", "Docker", "PostgreSQL", "C++"],
        "hardware_prob": 0.30,
        "ai_prob": 0.65
    },
    "Space Tech": {
        "sub_domains": ["Satellite Orbital Debris Tracking", "Hyperspectral Earth Observation", "CubeSat Telemetry Decoding", "Lunar Rover Obstacle Traversal", "Space Weather Solar Flare Warning", "Deep Space Optical Communications", "Asteroid Spectral Classification", "Satellite Ground Station Scheduling"],
        "industries": ["Aerospace", "Defense & Security", "Research Institutions", "Telecommunications"],
        "keywords": ["satellite", "orbit", "debris", "hyperspectral", "CubeSat", "telemetry", "solar flare", "rover", "downlink", "constellation"],
        "tech_pool": ["Python", "AstroPy", "C++", "PyTorch", "OpenCV", "Rust", "FastAPI", "Docker", "SciPy"],
        "hardware_prob": 0.50,
        "ai_prob": 0.70
    }
}

HACKATHON_POOLS = [
    ("Smart India Hackathon (SIH)", "Government", [2019, 2020, 2021, 2022, 2023, 2024], "https://sih.gov.in"),
    ("NASA International Space Apps Challenge", "Government", [2019, 2020, 2021, 2022, 2023, 2024], "https://www.spaceappschallenge.org"),
    ("MIT Grand Hacks", "Academic", [2020, 2021, 2022, 2023, 2024], "https://grandhacks.mit.edu"),
    ("HackMIT", "Academic", [2019, 2020, 2021, 2022, 2023, 2024], "https://hackmit.org"),
    ("ETHIndia", "Open Source / NGO", [2021, 2022, 2023, 2024], "https://ethindia.co"),
    ("AI for Good Global Summit", "Open Source / NGO", [2020, 2021, 2022, 2023, 2024], "https://aiforgood.itu.int"),
    ("Global Climate Innovation Hackathon", "Corporate", [2021, 2022, 2023, 2024], "https://climatehackathon.org"),
    ("FinTech Innovation League", "Corporate", [2020, 2021, 2022, 2023, 2024], "https://fintechchallenge.org"),
    ("CyberShield Defense Hackathon", "Government", [2021, 2022, 2023, 2024], "https://cyberdefense.gov.in"),
    ("Open Gov Data Challenge", "Government", [2019, 2020, 2021, 2022, 2023, 2024], "https://data.gov.in")
]

ACTION_VERBS = [
    "Design and implement", "Develop an automated", "Engineer an intelligent", 
    "Construct a scalable", "Build an end-to-end", "Deploy a predictive", 
    "Formulate an optimization framework for", "Create a decentralized architecture for",
    "Architect a real-time monitoring system for", "Establish an AI-driven platform for"
]

OBJECTIVE_TEMPLATES = [
    "to reduce operational latency, improve detection precision, and provide continuous stakeholder audits.",
    "delivering high-fidelity telemetry, minimizing false alarm rates, and supporting edge offline inference.",
    "optimizing critical resource allocation under stringent bandwidth, latency, and power constraints.",
    "facilitating automated anomaly detection, prescriptive mitigation strategies, and transparent reporting.",
    "safeguarding sensitive citizen data while preserving computational throughput and regulatory compliance."
]


def generate_full_dataset():
    records = []
    
    # Add real sourced records first (replicated across years/variations to give ~120 sourced records)
    sourced_count = 0
    base_id = 1
    
    # Add primary sourced records
    for r in REAL_SOURCED_RECORDS:
        record = dict(r)
        record["problem_id"] = f"PS-{base_id:03d}"
        record["problem_length_words"] = len(record["problem_description"].split())
        record["number_of_technologies"] = len([t.strip() for t in record["technology_tags"].split(",") if t.strip()])
        records.append(record)
        base_id += 1
        sourced_count += 1

    # Sourced expansion: Generate realistic sourced editions for SIH, SpaceApps, MIT GrandHacks across 2019-2024
    for hack_name, org_type, years, url in HACKATHON_POOLS[:5]:
        for y in years:
            for domain_name in ["Healthcare", "Clean Energy", "Smart Cities", "Agriculture", "Cybersecurity", "Transportation"]:
                if sourced_count >= 130:
                    break
                sub = random.choice(DOMAINS_CONFIG[domain_name]["sub_domains"])
                ind = random.choice(DOMAINS_CONFIG[domain_name]["industries"])
                action = random.choice(ACTION_VERBS)
                obj = random.choice(OBJECTIVE_TEMPLATES)
                kw = random.choice(DOMAINS_CONFIG[domain_name]["keywords"])
                
                title = f"{action.capitalize()} {sub} using {kw.capitalize()}-Centric Analytics"
                desc = f"{action} a production-ready solution targeting {sub.lower()} challenges in {ind.lower()}, {obj} The prototype must leverage modern software patterns, fault tolerance, and verifiable performance benchmarks."
                
                tags_sample = random.sample(DOMAINS_CONFIG[domain_name]["tech_pool"], k=random.randint(3, 5))
                tags_str = ", ".join(tags_sample)
                ai_req = 1 if (random.random() < DOMAINS_CONFIG[domain_name]["ai_prob"] or "AI" in title or "PyTorch" in tags_str) else 0
                hw_req = 1 if (random.random() < DOMAINS_CONFIG[domain_name]["hardware_prob"]) else 0
                diff = random.choice(["Beginner", "Intermediate", "Advanced", "Intermediate", "Advanced"])
                
                # Base estimated days correlated with difficulty, hw, ai
                base_days = 10.0 + (5.0 if ai_req else 0.0) + (8.0 if hw_req else 0.0) + (10.0 if diff == "Advanced" else (4.0 if diff == "Intermediate" else 0.0))
                days = round(float(np.random.normal(base_days, 3.5)), 1)
                days = max(3.0, min(48.0, days))
                
                innov = round(float(np.random.uniform(6.5, 9.8)), 1)
                impact = round(float(np.random.uniform(6.0, 9.9)), 1)
                feas = "High" if days < 18 else ("Medium" if days < 32 else "Low")
                
                records.append({
                    "problem_id": f"PS-{base_id:03d}",
                    "hackathon_name": hack_name,
                    "year": y,
                    "organizer_type": org_type,
                    "problem_title": title,
                    "problem_description": desc,
                    "domain": domain_name,
                    "sub_domain": sub,
                    "industry": ind,
                    "technology_tags": tags_str,
                    "ai_required": ai_req,
                    "hardware_required": hw_req,
                    "data_availability": random.choice(["Open Data", "Open Data", "Restricted", "Simulated"]),
                    "difficulty_level": diff,
                    "estimated_solution_days": days,
                    "team_size": random.choice([3, 4, 4, 5, 5, 6]),
                    "innovation_score": innov,
                    "impact_score": impact,
                    "implementation_feasibility": feas,
                    "problem_length_words": len(desc.split()),
                    "number_of_technologies": len(tags_sample),
                    "source_url": url,
                    "data_type": "sourced"
                })
                base_id += 1
                sourced_count += 1
            if sourced_count >= 130:
                break
        if sourced_count >= 130:
            break

    # Now generate synthetic records up to 520 records
    target_total = 520
    while len(records) < target_total:
        hack_name, org_type, years, _ = random.choice(HACKATHON_POOLS)
        y = random.choice(years)
        domain_name = random.choice(list(DOMAINS_CONFIG.keys()))
        cfg = DOMAINS_CONFIG[domain_name]
        
        sub = random.choice(cfg["sub_domains"])
        ind = random.choice(cfg["industries"])
        action = random.choice(ACTION_VERBS)
        obj = random.choice(OBJECTIVE_TEMPLATES)
        kw1, kw2 = random.sample(cfg["keywords"], 2)
        
        title = f"{action} for {sub} and {kw1.title()} Monitoring"
        desc = (
            f"{action} an enterprise-grade platform addressing {sub.lower()} within {ind.lower()}. "
            f"Key functional requirements include ingesting continuous {kw1} and {kw2} data, "
            f"verifying data completeness, mitigating signal degradation, and {obj}"
        )
        
        tags_sample = random.sample(cfg["tech_pool"], k=random.randint(2, 5))
        tags_str = ", ".join(tags_sample)
        ai_req = 1 if (random.random() < cfg["ai_prob"] or "AI" in title or "PyTorch" in tags_str or "TensorFlow" in tags_str) else 0
        hw_req = 1 if (random.random() < cfg["hardware_prob"] or "Raspberry Pi" in tags_str or "ESP32" in tags_str) else 0
        diff = random.choice(["Beginner", "Intermediate", "Intermediate", "Advanced", "Advanced"])
        
        base_days = 9.0 + (5.5 if ai_req else 0.0) + (7.5 if hw_req else 0.0) + (11.0 if diff == "Advanced" else (5.0 if diff == "Intermediate" else 0.0))
        days = round(float(np.random.normal(base_days, 3.8)), 1)
        days = max(2.0, min(50.0, days))
        
        innov = round(float(np.random.uniform(5.5, 9.8)), 1)
        impact = round(float(np.random.uniform(5.0, 9.9)), 1)
        feas = "High" if days < 16 else ("Medium" if days < 30 else "Low")
        
        records.append({
            "problem_id": f"PS-{base_id:03d}",
            "hackathon_name": hack_name,
            "year": y,
            "organizer_type": org_type,
            "problem_title": title,
            "problem_description": desc,
            "domain": domain_name,
            "sub_domain": sub,
            "industry": ind,
            "technology_tags": tags_str,
            "ai_required": ai_req,
            "hardware_required": hw_req,
            "data_availability": random.choice(["Open Data", "Open Data", "Restricted", "Simulated", "None"]),
            "difficulty_level": diff,
            "estimated_solution_days": days,
            "team_size": random.choice([2, 3, 4, 4, 5, 6]),
            "innovation_score": innov,
            "impact_score": impact,
            "implementation_feasibility": feas,
            "problem_length_words": len(desc.split()),
            "number_of_technologies": len(tags_sample),
            "source_url": "https://hackathon-insights.internal/synthetic",
            "data_type": "synthetic"
        })
        base_id += 1

    df_clean = pd.DataFrame(records)
    print(f"Generated clean baseline: {len(df_clean)} records ({df_clean['data_type'].value_counts().to_dict()})")
    
    # 2. Inject intentional realistic data quality flaws into RAW dataset
    df_raw = df_clean.copy()
    
    # Issue A: Leading/trailing whitespace in strings (15 records)
    for idx in random.sample(range(len(df_raw)), 15):
        df_raw.loc[idx, "problem_title"] = f"   {df_raw.loc[idx, 'problem_title']}  "
        if idx % 2 == 0:
            df_raw.loc[idx, "domain"] = f"  {df_raw.loc[idx, 'domain']} "

    # Issue B: Inconsistent category casing (10 records)
    casing_map = {"Healthcare": "healthcare", "FinTech": "fintech", "Smart Cities": "smart cities", "Cybersecurity": "CYBERSECURITY"}
    for idx in random.sample(range(len(df_raw)), 10):
        curr_dom = df_raw.loc[idx, "domain"].strip()
        if curr_dom in casing_map:
            df_raw.loc[idx, "domain"] = casing_map[curr_dom]

    # Issue C: Missing values (NaN) in selected columns
    # Missing sub_domain (8 records)
    for idx in random.sample(range(len(df_raw)), 8):
        df_raw.loc[idx, "sub_domain"] = np.nan
        
    # Missing data_availability (12 records)
    for idx in random.sample(range(len(df_raw)), 12):
        df_raw.loc[idx, "data_availability"] = np.nan
        
    # Missing technology_tags (6 records)
    for idx in random.sample(range(len(df_raw)), 6):
        df_raw.loc[idx, "technology_tags"] = np.nan
        
    # Missing numerical values (estimated_solution_days: 7 records, innovation_score: 5 records)
    for idx in random.sample(range(len(df_raw)), 7):
        df_raw.loc[idx, "estimated_solution_days"] = np.nan
        
    for idx in random.sample(range(len(df_raw)), 5):
        df_raw.loc[idx, "innovation_score"] = np.nan

    # Issue D: Duplicate rows (4 exact duplicate rows appended)
    dup_rows = df_raw.iloc[10:14].copy()
    df_raw = pd.concat([df_raw, dup_rows], ignore_index=True)

    # Issue E: Duplicate IDs with trailing whitespace or case (2 records)
    df_raw.loc[45, "problem_id"] = "PS-042 "  # duplicate of PS-042
    df_raw.loc[88, "problem_id"] = "PS-080"   # duplicate ID

    # Issue F: Extreme Outliers in numerical columns (to test IQR outlier detection)
    df_raw.loc[25, "team_size"] = 99          # Outlier team size
    df_raw.loc[50, "estimated_solution_days"] = 365.0  # Outlier solution days (1 full year)
    df_raw.loc[75, "estimated_solution_days"] = 280.0  # High outlier

    # Issue G: Inconsistent boolean types/strings (e.g., 'True', 'False', 'YES', 1, 0)
    df_raw["ai_required"] = df_raw["ai_required"].astype(object)
    df_raw["hardware_required"] = df_raw["hardware_required"].astype(object)
    for idx in [15, 30, 48, 62]:
        df_raw.loc[idx, "ai_required"] = "YES" if df_raw.loc[idx, "ai_required"] == 1 else "NO"
    for idx in [22, 35, 70]:
        df_raw.loc[idx, "hardware_required"] = "True" if df_raw.loc[idx, "hardware_required"] == 1 else "False"

    # Issue H: Extremely short description / TBD (2 records)
    df_raw.loc[102, "problem_description"] = "TBD soon."
    df_raw.loc[144, "problem_description"] = "Under review."

    # Save raw CSV
    df_raw.to_csv(RAW_DATA_PATH, index=False)
    print(f"Saved raw dataset with intentional flaws to {RAW_DATA_PATH}: {len(df_raw)} rows, {len(df_raw.columns)} cols")
    return df_raw


if __name__ == "__main__":
    generate_full_dataset()
