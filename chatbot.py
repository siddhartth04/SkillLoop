import os
import re
import json
from google import genai
from google.genai import types
from google.oauth2 import service_account

# ==============================================================================
# VERTEX AI CONFIG (via google-genai SDK) — replaces Azure OpenAI config
# ==============================================================================

PROJECT_ID = "gen-lang-client-0497439750"          # ← replace with your GCP project ID
LOCATION = "us-central1"                     # ← replace with your Vertex AI region
GPT35 = "gpt35turbo"
GPT4 = "gpt-4o"
EMBEDDING_MODEL_DEPLOYEMENT_NAME = "ada-002-leap"


######################################################################################
# model_to_use="gpt-4o"
model_to_use = "gemini-2.5-flash"          # ← Vertex AI model name (check exact name in your GCP console)
#model_to_use = "gpt35turbo"
#######################################################################################

# Load service account credentials from environment variable (JSON string)
# Set this in Lambda → Configuration → Environment variables → key: GOOGLE_SERVICE_ACCOUNT_JSON
service_account_info = json.loads(os.environ["GOOGLE_SERVICE_ACCOUNT_JSON"])

credentials = service_account.Credentials.from_service_account_info(
    service_account_info,
    scopes=["https://www.googleapis.com/auth/cloud-platform"]
)

client = genai.Client(
    vertexai=True,
    project=PROJECT_ID,
    location=LOCATION,
    credentials=credentials,
)


def call_gpt_model(
    model=" ",
    messages=[],
    temperature=0.1,
    max_tokens=700,
    stream=False
):
    """
    Drop-in replacement for the Azure OpenAI call.
    Converts OpenAI-style messages list into a single Gemini-compatible prompt,
    then calls Vertex AI (via google-genai SDK) generate_content.
    """
    # Separate system message from conversation messages
    system_text = ""
    convo_parts = []
    for msg in messages:
        if msg["role"] == "system":
            system_text = msg["content"]
        elif msg["role"] == "user":
            convo_parts.append(f"User: {msg['content']}")
        elif msg["role"] == "assistant":
            convo_parts.append(f"Assistant: {msg['content']}")

    full_prompt = system_text + "\n\n" + "\n".join(convo_parts) + "\nAssistant:"

    response_obj = client.models.generate_content(
        model=model,
        contents=full_prompt,
        config=types.GenerateContentConfig(
            temperature=temperature,
            max_output_tokens=max_tokens,
        ),
    )
    return response_obj

# ==============================================================================
# SYSTEM PROMPT
# ==============================================================================

system_message = """
You are an intelligent AI assistant for Yash Technologies (YASHTech.ai), specialized in answering questions about YASH Technologies' AI services, platforms, industry solutions, and NEUPAC platform.

CRITICAL RULE — SOURCE OF TRUTH:
You MUST answer ONLY from the knowledge injected below under "RELEVANT KNOWLEDGE".
Do NOT use any outside knowledge. Do NOT invent details, URLs, or features not present in the injected knowledge.
If a question cannot be answered from the injected knowledge, say: "I don't have enough information on that. To know more write to us at info@neupac.com or Click here to Contact us: https://www.neupac.ai/contact-us/"

YOUR RESPONSIBILITIES:

1. ANSWER ACCURATELY FROM DOCUMENT:
   - Answer questions using the exact information, descriptions, and URLs from the knowledge provided.
   - When URLs are available for a topic, always include them in your response as "Know more: <url>".
   - Preserve the meaning and intent of the original content — do not paraphrase in a way that changes facts.
   - Do NOT add suggestions, recommendations, or follow-up offers that are not grounded in the injected knowledge.
   - Do NOT append lines like "If you'd like, I can also..." or "You might also be interested in..." unless the user explicitly asks.

2. MATCH AND RESPOND:
   - If a user asks about a service, platform, or industry, match it to the closest entry in the knowledge and respond with full details.
   - If multiple entries are clearly relevant to the query, list all of them.
   - Do NOT suggest unrelated topics or cross-sell other sections unprompted.

3. HANDLE AMBIGUITY — NEVER DEAD-END:
   - Never say "I don't know" if there is any related knowledge available.
   - For vague queries, offer the closest matching topic and ask ONE clarifying question.
   - Example: "Are you asking about NEUPAC's security features or its integration capabilities?"

4. STRICT SCOPE:
   - Only answer questions about YASH Technologies' services, platforms, industry solutions, and NEUPAC.
   - If asked to write code, draft emails, do math, translate, or anything unrelated — politely decline:
     "I'm here to help you explore YASH Technologies' AI services and platforms. For anything else, I'm not the right tool."

5. TONE:
   - Be clear, professional, and concise.
   - Stick strictly to what the document says. Nothing more.
"""

# ==============================================================================
# KNOWLEDGE BASE — Built exclusively from YASHTech_ai_and_Neupac_Documentation.docx
# Structured as chunks. Each chunk = one topic from the document.
# ==============================================================================

PROJECT_CHUNKS = [

    {
        "title": "About YASH Technologies",
        "category": "Company",
        "content": (
            "YASH Technologies is a global technology services and solutions company focused on helping enterprises "
            "modernize, transform, and grow through the power of technology and AI. "
            "YASH partners with organizations across industries to deliver end-to-end digital transformation, "
            "combining deep domain expertise with cutting-edge AI, data analytics, cloud, and SAP capabilities.\n\n"
            "YASH Technologies serves clients across Manufacturing, Retail, Healthcare & Life Sciences, "
            "Banking & Financial Services (BFSI), Energy & Resources, and more.\n\n"
            "Website: https://www.yash.com\n"
            "AI Platform: https://www.yashtech.ai"
        ),
        "url": "https://www.yash.com",
        "keywords": [
            "yash", "yash technologies", "about yash", "who is yash", "what is yash",
            "yash company", "yash tech", "yashtech", "tell me about yash",
            "what does yash do", "yash services", "yash solutions",
            "digital transformation", "technology services", "global technology",
            "yash.com", "yashtech.ai", "yash website"
        ]
    },

    {
        "title": "ROI with AI",
        "category": "AI Service",
        "content": (
            "ROI with AI is YASH Technologies' strategic foundation for enterprise AI transformation. "
            "It is designed to help organizations move decisively from experimentation to sustained value creation "
            "by aligning AI investments with business priorities, operating realities, and measurable outcomes."
        ),
        "url": "https://www.yashtech.ai/services/roi-with-ai",
        "keywords": ["roi", "return on investment", "ai transformation", "value creation", "business priorities",
                     "measurable outcomes", "experimentation", "strategy", "ai strategy", "investment"]
    },

    {
        "title": "AI Solutions",
        "category": "AI Service",
        "content": (
            "YASH AI Solutions brings together a portfolio of enterprise-grade AI systems, each engineered to address "
            "a distinct operational challenge — from coordination and modernization to cost control, trust, and customer engagement. "
            "This is where AI stops being conceptual and starts becoming operational muscle. "
            "YASH AI Solutions are designed as purpose-built systems, each aligned to a specific enterprise reality."
        ),
        "url": "https://www.yashtech.ai/services/ai-solutions",
        "keywords": ["ai solutions", "enterprise ai", "operational", "cost control", "trust", "customer engagement",
                     "coordination", "modernization", "purpose-built", "enterprise grade",
                     "what ai solutions", "yash ai solutions", "solutions yash offer", "what does yash offer",
                     "what ai solutions does yash", "ai solutions yash offer", "yash offer ai solutions"]
    },

    {
        "title": "AI Life Cycle Management",
        "category": "AI Service",
        "content": (
            "YASH's AI life cycle management services provide operational reliability, ensuring AI systems remain accurate, "
            "controllable, cost-effective, and trustworthy long after they are deployed."
        ),
        "url": "https://www.yashtech.ai/services/ai-life-cycle-management",
        "keywords": ["life cycle", "lifecycle", "ai management", "operational reliability", "accurate", "controllable",
                     "cost-effective", "trustworthy", "deployed", "maintain", "management", "improve over time", "learn"]
    },

    {
        "title": "Trusted AI",
        "category": "AI Service",
        "content": (
            "YASH's Trusted AI services ensure AI can be used confidently, repeatedly, and at enterprise scale, "
            "without introducing unacceptable risk."
        ),
        "url": "https://www.yashtech.ai/services/trusted-ai",
        "keywords": ["trusted ai", "trust", "responsible ai", "ethical ai", "risk", "enterprise scale",
                     "confident", "reliable", "safety", "bias", "compliance", "ethical", "guardrails"]
    },

    {
        "title": "Agent Foundry",
        "category": "AI Service",
        "content": (
            "YASH's Agent Foundry includes a growing library of pre-built and configurable AI agents, designed for both "
            "industry-specific and horizontal use cases. These agents are engineered patterns, built with enterprise "
            "constraints in mind and refined through real-world usage. Organizations can deploy them as-is, adapt them "
            "to context, or extend them as part of larger workflows — accelerating adoption without sacrificing control."
        ),
        "url": "https://www.yashtech.ai/services/agent-foundry",
        "keywords": ["agent foundry", "pre-built agents", "configurable agents", "ai agents", "library",
                     "industry-specific", "horizontal", "deploy", "workflow", "adoption", "agent", "foundry"]
    },

    {
        "title": "AI Labs",
        "category": "AI Service",
        "content": (
            "YASH AI Labs provides a controlled environment where organizations can investigate new AI capabilities, "
            "test assumptions, and validate feasibility — before committing to scale."
        ),
        "url": "https://www.yashtech.ai/services/ai-labs",
        "keywords": ["ai labs", "lab", "research", "test", "feasibility", "investigate", "capabilities",
                     "controlled environment", "validate", "experiment", "pilot"]
    },

    # ─── INDUSTRY SOLUTIONS ───────────────────────────────────────────────────

    {
        "title": "AI in Manufacturing",
        "category": "Industry Solution",
        "content": (
            "YASH's AI/ML and GenAI offerings are tailored to the manufacturing industry and focus on data analytics. "
            "YASH's services are strategically crafted to streamline supply chain operations, enhance operational efficiency, "
            "promote energy conservation practices, and maximize production output."
        ),
        "url": "https://www.yashtech.ai/ai-in-industries/manufacturing",
        "keywords": ["manufacturing", "factory", "production", "supply chain", "operational efficiency",
                     "energy conservation", "data analytics", "genai manufacturing", "plant", "output",
                     "agentic manufacturing", "machine", "predictive maintenance manufacturing",
                     "solutions for manufacturing", "manufacturing solutions", "what solutions manufacturing"]
    },

    {
        "title": "AI in Retail",
        "category": "Industry Solution",
        "content": (
            "YASH Technologies provides customized AI, GenAI, and Data Analytics solutions to tackle the unique challenges "
            "retailers face, enhance operations, and boost customer engagement. Through these tailored solutions, YASH helps "
            "retailers transform their business processes, ensuring a seamless and engaging shopping experience for customers."
        ),
        "url": "https://www.yashtech.ai/ai-in-industries/retail",
        "keywords": ["retail", "shopping", "customer engagement", "ecommerce", "retailer", "store",
                     "business process", "consumer", "genai retail", "data analytics retail"]
    },

    {
        "title": "AI in Healthcare and Life Sciences",
        "category": "Industry Solution",
        "content": (
            "YASH Technologies guides healthcare providers, payers, and life sciences organizations through AI transformation, "
            "ensuring they achieve greater efficiency and enhance existing products and services."
        ),
        "url": "https://www.yashtech.ai/ai-in-industries/healthcare-and-life-sciences",
        "keywords": ["healthcare", "health", "life sciences", "lifesciences", "medical", "hospital", "payer",
                     "provider", "patient", "clinical", "pharma", "pharmaceutical", "diagnostics"]
    },

    {
        "title": "AI in Banking and Insurance (BFSI)",
        "category": "Industry Solution",
        "content": (
            "At YASH Technologies, cutting-edge AI and data solutions help financial institutions streamline operations, "
            "enhance customer experiences, and drive business growth."
        ),
        "url": "https://www.yashtech.ai/ai-in-industries/banking-and-financial-services",
        "keywords": ["banking", "insurance", "bfsi", "financial", "finance", "bank", "financial institution",
                     "fintech", "wealth", "investment banking", "risk", "loan", "credit"]
    },

    {
        "title": "AI in Energy and Resources",
        "category": "Industry Solution",
        "content": (
            "At YASH Technologies, energy-specific experience, solutions, and tools accelerate innovation, reduce costs, "
            "and maximize the value derived from data. YASH's expertise in AI, Generative AI (GenAI), and Data Analytics "
            "enables energy companies to harness the full potential of their data, fostering a paradigm shift towards safer, "
            "more reliable, and smarter energy ecosystems."
        ),
        "url": "https://www.yashtech.ai/ai-in-industries/energy-and-resources",
        "keywords": ["energy", "resources", "oil", "gas", "utilities", "renewable", "power", "grid",
                     "sustainability", "smart energy", "genai energy", "data analytics energy"]
    },

    # ─── FUNCTION SOLUTIONS ───────────────────────────────────────────────────

    {
        "title": "AI in Finance Function",
        "category": "Function Solution",
        "content": (
            "YASH brings together AI and predictive analytics, workflow automation, cloud integrations, and organizational "
            "change expertise to modernize finance functions. YASH helps CFOs and finance leaders move from hindsight to "
            "foresight, building resilience into every decision."
        ),
        "url": "https://www.yashtech.ai/services/ai-in-finance",
        "keywords": ["finance function", "cfo", "predictive analytics", "workflow automation", "cloud integration",
                     "finance modernization", "foresight", "resilience", "financial decision", "ai finance",
                     "yash finance", "what does yash offer finance", "finance solutions", "finance function solutions"]
    },

    {
        "title": "AI in Human Resources",
        "category": "Function Solution",
        "content": (
            "YASH provides HR solutions that embed advanced AI and analytics into everyday workflows. HR leaders can move "
            "beyond manual processes to anticipate workforce needs, personalize employee journeys, and drive organizational "
            "agility. YASH's Agentic AI offering complements these solutions by orchestrating tasks and surfacing insights "
            "in real time, helping HR teams act faster and with greater confidence."
        ),
        "url": "https://www.yashtech.ai/services/ai-in-human-resources",
        "keywords": ["hr", "human resources", "workforce", "employee", "onboarding", "payroll", "talent",
                     "recruitment", "personnel", "people", "agentic hr", "organizational agility",
                     "employee journey", "sentiment", "upskilling", "training hr",
                     "hr solutions", "yash hr", "what hr solutions", "human resource solutions",
                     "hr solutions yash provide", "yash provide hr", "what hr yash provide"]
    },

    {
        "title": "AI in Sales and Marketing",
        "category": "Function Solution",
        "content": (
            "YASH blends customer analytics, GenAI content creation, and other capabilities to help marketing teams move "
            "from campaigns to conversations. AI is applied where it matters most: understanding audiences, generating content, "
            "and orchestrating engagement — so marketers can focus on creativity and strategy."
        ),
        "url": "https://www.yashtech.ai/services/ai-in-marketing",
        "keywords": ["sales", "marketing", "customer analytics", "content creation", "genai content", "campaigns",
                     "audience", "engagement", "marketer", "advertising", "brand", "lead generation"]
    },

    {
        "title": "AI in IT Operations",
        "category": "Function Solution",
        "content": (
            "YASH brings AI-driven observability, automated remediation, and hybrid infrastructure management together "
            "with expertise in SAP modernization and managed services. AI is the thread here — spotting anomalies, "
            "predicting failures, and triggering fixes before users even notice."
        ),
        "url": "https://www.yashtech.ai/services/ai-in-it-operations",
        "keywords": ["it operations", "itops", "observability", "remediation", "infrastructure", "sap modernization",
                     "managed services", "anomaly", "failure prediction", "devops", "it", "operations"]
    },

    {
        "title": "AI in Supply Chain",
        "category": "Function Solution",
        "content": (
            "YASH helps make supply chains adaptive. Whether workflow automation, intelligent monitoring, or integration "
            "with enterprise platforms, YASH gives leaders a real-time view of their operations. YASH's Agentic AI platform "
            "helps orchestrate alerts and automate responses to anomalies, so supply chains adjust faster without adding complexity."
        ),
        "url": "https://www.yashtech.ai/services/ai-in-supply-chain",
        "keywords": ["supply chain", "scm", "logistics", "procurement", "inventory", "workflow automation",
                     "intelligent monitoring", "agentic supply chain", "anomaly response", "adaptive supply chain",
                     "real-time operations", "distribution",
                     "supply chain solutions", "yash supply chain", "what supply chain solutions"]
    },

    # ─── AI PLATFORMS ─────────────────────────────────────────────────────────

    {
        "title": "NEUPAC — Platform Overview",
        "category": "AI Platform",
        "content": (
            "NEUPAC (Next-Gen Enterprise Unified Platform for AI-Agent Control) is a patented enterprise-grade AI platform "
            "for governing, optimizing, and scaling AI journeys. It is designed to bridge the gap between AI strategy and "
            "execution, moving organizations from fragmented AI experimentation to becoming a unified 'AI-first' company. "
            "It acts as a 'single pane of glass' that allows organizations to orchestrate, monitor, and secure their entire "
            "AI portfolio — including both custom-built agents and out-of-the-box agents from various OEMs.\n\n"
            "Enterprise AI deployments today suffer from three critical, compounding failures:\n"
            "- Shadow AI: agents and LLM tools operating entirely outside compliance perimeters, invisible to governance teams.\n"
            "- Unmonitored LLM expenditures: costs accumulating silently across fragmented, ungoverned AI tools with no budget accountability.\n"
            "- Zero accountability: no clear ownership, audit trail, or governance visibility when AI-assisted decisions go wrong.\n\n"
            "NEUPAC solves all three."
        ),
        "url": "https://www.yashtech.ai/platforms/unified-agentic-ai-platform",
        "keywords": ["neupac", "what is neupac", "platform overview", "ai governance platform", "single pane of glass",
                     "ai first", "shadow ai", "what is shadow ai", "unmonitored llm", "zero accountability", "ai portfolio",
                     "agentic ai platform", "unified platform", "enterprise ai platform", "patented"]
    },

    {
        "title": "NEUPAC — What It Does",
        "category": "AI Platform",
        "content": (
            "NEUPAC provides:\n"
            "- A single, authoritative control plane for the entire enterprise AI ecosystem.\n"
            "- Every AI agent — OEM tools (SAP Joule, Salesforce Einstein, Microsoft Copilot), custom-built agents, "
            "BYOA (Bring Your Own Agent), BYOM (Bring Your Own Model), and RAG systems — is registered, classified, owned, "
            "and continuously monitored through one platform.\n"
            "- Real-time PII detection enforces data protection at the exact moment of interaction — not just in policy.\n"
            "- Agent Trust Score gives governance teams a continuously calculated, always-on reliability signal for every agent.\n"
            "- FinOps observability tracks LLM cost down to the individual user, agent, and token level — eliminating budget "
            "surprises and shadow AI spend."
        ),
        "url": "https://www.yashtech.ai/platforms/unified-agentic-ai-platform",
        "keywords": ["control plane", "oem agents", "sap joule", "salesforce einstein", "microsoft copilot",
                     "byoa", "bring your own agent", "bring my own agent", "bring own agent",
                     "byom", "bring your own model", "bring my own model", "rag",
                     "pii detection", "finops", "llm cost", "agent registry",
                     "what does neupac do", "neupac features", "neupac capabilities"]
    },

    {
        "title": "NEUPAC — How Users Can Use It",
        "category": "AI Platform",
        "content": (
            "NEUPAC is delivered through a web-based or Microsoft Teams interface — users interact with AI agents entirely "
            "within the collaboration environment they already use daily.\n\n"
            "Business, operational, and administrative users access agents, prompts, shared workspaces, and collaboration "
            "within user groups.\n\n"
            "Governance administrators work through a separate, secure Admin Portal to:\n"
            "- Set up and manage AI agents — register new agents, track their status, and retire them when no longer needed.\n"
            "- Control who can access what — decide which teams and individuals can use which AI agents, and at what level of access.\n"
            "- Define the rules — set safety guardrails, configure what sensitive data gets protected, and establish acceptable "
            "performance standards for each agent.\n"
            "- Keep an eye on compliance — monitor whether AI agents are behaving as expected and access a full history of "
            "every action taken on the platform for audit purposes."
        ),
        "url": "https://www.yashtech.ai/platforms/unified-agentic-ai-platform",
        "keywords": ["how to use neupac", "user interface", "microsoft teams", "web interface", "admin portal",
                     "governance admin", "register agents", "access control", "guardrails", "audit history",
                     "compliance monitor", "teams integration", "easy setup", "user groups", "workspace"]
    },

    {
        "title": "NEUPAC — Model Zoo",
        "category": "AI Platform",
        "content": (
            "The Model Zoo in NEUPAC provides:\n"
            "- Vendor Neutrality: Allows enterprises to avoid vendor lock-in by providing a library of multiple LLMs "
            "(e.g., GPT, Gemini, Llama) and Small Language Models (SLMs).\n"
            "- Dynamic Switching: Users can switch LLMs for specific agents if a newer, more cost-effective, or "
            "higher-performing version becomes available in the market.\n"
            "- Custom Enterprise Models: Provides the infrastructure to bring your own model or build a proprietary "
            "enterprise language model that serves as a unique business differentiator.\n"
            "- Ease of model upgrade: Provides the flexibility to upgrade models without losing inferencing."
        ),
        "url": "https://www.yashtech.ai/platforms/unified-agentic-ai-platform",
        "keywords": ["model zoo", "llm", "gpt", "gemini", "llama", "slm", "small language model",
                     "vendor neutrality", "vendor lock-in", "dynamic switching", "byom", "bring your own model",
                     "custom model", "proprietary model", "model upgrade", "inferencing", "language model"]
    },

    {
        "title": "NEUPAC — Agent Governance and Orchestration",
        "category": "AI Platform",
        "content": (
            "NEUPAC's Agent Governance and Orchestration capabilities:\n"
            "- Multi-Agent Orchestration: NEUPAC features an orchestration layer that determines which agent to call for "
            "specific tasks within a complex workflow.\n"
            "- Bring Your Own Agent (BYOA): Enterprises can onboard agents they have already built elsewhere into NEUPAC "
            "to gain immediate governance and observability with minimal changes.\n"
            "- Budget Control and Governance: Allows you to plan and control your spend on AI.\n"
            "- Interoperability: Uses protocols like Model Context Protocol (MCP) and Agent-to-Agent (A2A) to communicate "
            "with external OEM agents (like those from Salesforce, SAP, or any other OEMs)."
        ),
        "url": "https://www.yashtech.ai/platforms/unified-agentic-ai-platform",
        "keywords": ["multi-agent orchestration", "agent orchestration", "byoa", "what is byoa",
                     "byoa explained", "bring your own agent", "bring my own agent", "bring own agent",
                     "can i bring", "onboard my agent", "own agent", "existing agent", "already built agent",
                     "budget control", "governance", "mcp", "model context protocol", "a2a", "agent to agent",
                     "interoperability", "oem agents", "sap", "salesforce", "workflow", "complex workflow",
                     "agent collaboration", "orchestration layer"]
    },

    {
        "title": "NEUPAC — Data and Security",
        "category": "AI Platform",
        "content": (
            "NEUPAC's data and security features:\n"
            "- Dedicated Hosting: NEUPAC is deployed in the customer's own dedicated AWS environment, set up and managed by YASH.\n"
            "- Zero Data Copy Architecture: Interacts with data internally without extracting or duplicating it from core systems "
            "like ERPs, reducing security risks.\n"
            "- RBAC for Agents: Implements Role-Based Access Control (RBAC) connected to Active Directory, ensuring users only "
            "access agents relevant to their specific persona or team."
        ),
        "url": "https://www.yashtech.ai/platforms/unified-agentic-ai-platform",
        "keywords": ["data security", "security", "privacy", "dedicated hosting", "aws", "zero data copy",
                     "zero data copy architecture", "erp", "rbac", "role based access control",
                     "active directory", "data protection", "least privilege", "persona", "team access",
                     "cloud hosting", "dedicated environment",
                     "neupac data security", "neupac security features", "data and security neupac",
                     "neupac data features", "security features neupac"]
    },

    {
        "title": "NEUPAC — Advanced Monitoring and Observability",
        "category": "AI Platform",
        "content": (
            "NEUPAC provides deep insights that traditional AI platforms often lack:\n"
            "- Responsible AI (RAI) Dashboard: Tracks a 'Trust Score' for agents based on eight parameters: relevancy, "
            "faithfulness, hallucination, bias, toxicity, transparency, fairness, and safety.\n"
            "- FinOps for AI: Monitors token usage and costs at the individual user and team level, and gives you an "
            "opportunity to optimize your AI spend."
        ),
        "url": "https://www.yashtech.ai/platforms/unified-agentic-ai-platform",
        "keywords": ["monitoring", "observability", "rai dashboard", "responsible ai", "trust score",
                     "what is trust score", "relevancy", "faithfulness", "hallucination", "bias", "toxicity",
                     "transparency", "fairness", "safety", "finops", "token usage", "token cost", "ai spend",
                     "optimize spend", "8 parameters", "eight parameters"]
    },

    {
        "title": "NEUPAC — Adoption and ROI",
        "category": "AI Platform",
        "content": (
            "NEUPAC's Adoption and ROI features:\n"
            "- Adoption Intelligence: NEUPAC provides insights for tracking AI spend, utilization, and licensing, "
            "delivering the actionable intelligence required to monitor adoption trends and drive strategic cost optimization.\n"
            "- Prompt Cataloging: Every prompt and response (with an audit trail) is analyzed to check if users need more "
            "training or if the prompt library requires fine-tuning to improve adoption.\n"
            "- ROI: The system provides ROI metrics for specific agents that helps in value realization for AI."
        ),
        "url": "https://www.yashtech.ai/platforms/unified-agentic-ai-platform",
        "keywords": ["adoption", "roi", "return on investment", "prompt catalog", "prompt cataloging",
                     "adoption intelligence", "ai spend tracking", "utilization", "licensing",
                     "cost optimization", "audit trail", "value realization", "fine-tuning", "training"]
    },

    {
        "title": "NEUPAC — Unique User Features",
        "category": "AI Platform",
        "content": (
            "NEUPAC's unique user features:\n"
            "- Collaborative Workspaces: Unlike standard one-to-one AI chats, this allows teams to share a context and "
            "chat history for a specific project (like an RFP), preventing redundant token consumption.\n"
            "- Offline Processing: For non-urgent tasks, users can select batch processing, which can reduce LLM costs by up to 60%.\n"
            "- Teams Integration: The platform is deeply integrated with Microsoft Teams, allowing users to access their "
            "authorized agents directly within their daily communication tool.\n"
            "- Multi Organization Management: The platform can govern AI initiatives across group companies to bring a unified experience."
        ),
        "url": "https://www.yashtech.ai/platforms/unified-agentic-ai-platform",
        "keywords": ["collaborative workspace", "workspace", "rfp", "token consumption", "offline processing",
                     "batch processing", "60% cost reduction", "llm cost reduction", "teams integration",
                     "microsoft teams", "multi organization", "group companies", "unified experience",
                     "unique features", "user features", "unique user features", "neupac features unique"]
    },

    {
        "title": "NEUPAC — How It Helps CEOs",
        "category": "AI Platform",
        "content": (
            "How NEUPAC helps CEOs:\n"
            "- Accelerated AI Adoption: Rapidly transition from pilots to enterprise-wide deployment, embedding AI into "
            "core business workflows.\n"
            "- Enterprise Integration at Scale: Seamlessly orchestrate AI agents across systems, functions, and OEM "
            "ecosystems — without disruption.\n"
            "- AI Cost Control and Visibility: Built-in FinOps ensures real-time transparency and budget control on AI "
            "investments — so innovation doesn't come at the expense of margins."
        ),
        "url": "https://www.yashtech.ai/platforms/unified-agentic-ai-platform",
        "keywords": ["ceo", "chief executive", "ai adoption", "enterprise deployment", "pilot to production",
                     "oem ecosystem", "ai cost control", "finops ceo", "budget control", "ai investment",
                     "margins", "business workflows", "how neupac helps ceo", "neupac ceo", "help ceo"]
    },

    {
        "title": "NEUPAC — How It Helps CFOs",
        "category": "AI Platform",
        "content": (
            "How NEUPAC helps CFOs:\n"
            "- AI Cost Optimization: Built-in FinOps provides real-time visibility and budget control on AI spend — "
            "ensuring efficiency at scale.\n"
            "- Smarter Resource Allocation: Optimize infrastructure, models, and workloads to maximize utilization and minimize waste.\n"
            "- Bottom-Line Impact: Scale AI initiatives with financial accountability — aligning innovation with profitability "
            "and business outcomes."
        ),
        "url": "https://www.yashtech.ai/platforms/unified-agentic-ai-platform",
        "keywords": ["cfo", "chief financial officer", "cost optimization", "finops cfo", "resource allocation",
                     "infrastructure optimization", "workload optimization", "financial accountability",
                     "profitability", "bottom line", "ai budget", "ai spend cfo",
                     "how neupac helps cfo", "neupac cfo", "help cfo"]
    },

    {
        "title": "NEUPAC — How It Helps CISOs",
        "category": "AI Platform",
        "content": (
            "How NEUPAC helps CISOs:\n"
            "- End-to-End Governance: Built-in frameworks for policy enforcement, auditability, and lifecycle control "
            "across all AI agents.\n"
            "- Responsible AI by Design: Bias controls, explainability, and audit trails ensure compliance with evolving "
            "regulatory standards.\n"
            "- Secure Architecture and Access Control: RBAC-native design enforces least-privilege access across workflows, "
            "models, and data pipelines."
        ),
        "url": "https://www.yashtech.ai/platforms/unified-agentic-ai-platform",
        "keywords": ["ciso", "chief information security officer", "governance", "policy enforcement", "auditability",
                     "lifecycle control", "responsible ai ciso", "bias controls", "explainability", "audit trail ciso",
                     "regulatory compliance", "rbac ciso", "least privilege", "data pipelines", "security officer",
                     "how neupac helps ciso", "neupac ciso", "help ciso"]
    },

    {
        "title": "NEUPAC — How It Helps CIOs",
        "category": "AI Platform",
        "content": (
            "How NEUPAC helps CIOs:\n"
            "- Interoperable by Design: Orchestrate agents from any OEM ecosystem — no rip-and-replace required. "
            "Custom-built or off-the-shelf, they all operate as one.\n"
            "- Built-in FinOps and Governance: Integrated FinOps and observability capabilities deliver real-time cost "
            "transparency, budget control on AI spend, and end-to-end oversight.\n"
            "- Responsible AI Framework: Trust AI architecture with bias controls, explainability, and audit trails — "
            "engineered to meet the governance expectations of regulated industries.\n"
            "- Secure and RBAC-Native: Role-Based Access Control baked in from day one. Enforce least-privilege policies "
            "across every agent, workflow, and data pipeline.\n"
            "- Accelerated AI Adoption: Move faster from pilots to production with a unified platform that simplifies "
            "deployment, scaling, and management."
        ),
        "url": "https://www.yashtech.ai/platforms/unified-agentic-ai-platform",
        "keywords": ["cio", "chief information officer", "interoperable", "oem ecosystem", "no rip and replace",
                     "finops cio", "observability cio", "responsible ai cio", "rbac cio", "regulated industries",
                     "pilot to production cio", "deployment", "scaling", "management cio",
                     "how neupac helps cio", "neupac cio", "help cio"]
    },

    {
        "title": "NEUPAC — Start Free Trial and Schedule Demo",
        "category": "AI Platform",
        "content": (
            "You can start a free trial or schedule a demo of NEUPAC at:\n"
            "Start Free Trial: https://www.neupac.ai/contact-form/\n"
            "Schedule Demo: https://www.neupac.ai/contact-form/"
        ),
        "url": "https://www.neupac.ai/contact-form/",
        "keywords": ["free trial", "trial", "demo", "schedule demo", "contact", "get started neupac",
                     "try neupac", "neupac demo", "start neupac", "sign up"]
    },

    # ─── NEUPAC FAQs ──────────────────────────────────────────────────────────

    {
        "title": "FAQ — What is the unified Agentic AI platform offered by YASH Technologies?",
        "category": "FAQ",
        "content": (
            "YASH Technologies offers a suite of enterprise AI solutions, anchored by its proprietary platform called NEUPAC. "
            "It serves as a unified ecosystem designed to help organizations design, orchestrate, govern, and scale autonomous "
            "AI agents across their enterprise.\n"
            "Know more: https://www.neupac.ai/"
        ),
        "url": "https://www.neupac.ai/",
        "keywords": ["unified agentic ai", "agentic ai platform", "yash platform", "what platform does yash offer",
                     "neupac faq", "enterprise ai solutions yash"]
    },

    {
        "title": "FAQ — What does NEUPAC stand for?",
        "category": "FAQ",
        "content": (
            "NEUPAC stands for 'Next-gen Ecosystem Unifying Platform for Agent Control.' "
            "Its primary purpose is to bridge the gap between isolated, experimental AI initiatives and production-grade "
            "enterprise deployment, giving leadership the control needed to scale AI without accumulating governance risks.\n"
            "Know more: https://www.neupac.ai/"
        ),
        "url": "https://www.neupac.ai/",
        "keywords": ["neupac stand for", "neupac full form", "what does neupac mean", "neupac abbreviation",
                     "next gen ecosystem", "agent control", "primary purpose neupac"]
    },

    {
        "title": "FAQ — How is Agentic AI different from traditional Generative AI?",
        "category": "FAQ",
        "content": (
            "While Generative AI is primarily designed to generate content (text, code, or images) in response to a user's prompt, "
            "Agentic AI is proactive. It acts as an autonomous digital teammate capable of reasoning, planning, making decisions, "
            "executing multi-step tasks, and interacting directly with enterprise tools to achieve specific business goals with "
            "minimal human intervention."
        ),
        "url": "https://www.yashtech.ai/platforms/unified-agentic-ai-platform",
        "keywords": ["agentic ai vs generative ai", "difference agentic generative", "what is agentic ai",
                     "how is agentic ai different", "generative ai vs agentic", "autonomous ai", "proactive ai",
                     "multi-step tasks", "digital teammate", "reasoning planning"]
    },

    {
        "title": "FAQ — What are the core features of NEUPAC?",
        "category": "FAQ",
        "content": (
            "NEUPAC's core capabilities include:\n"
            "- Seamless multi-agent orchestration\n"
            "- Zero-data-copy architecture\n"
            "- AI FinOps for tracking and accountability\n"
            "- A robust compliance framework built to meet strict global and European regulatory requirements.\n"
            "Know more: https://www.neupac.ai/"
        ),
        "url": "https://www.neupac.ai/",
        "keywords": ["core features neupac", "neupac capabilities", "key features", "main features neupac",
                     "multi-agent orchestration", "zero data copy", "ai finops", "compliance framework",
                     "european regulatory", "eu ai act"]
    },

    {
        "title": "FAQ — Does NEUPAC integrate with existing enterprise systems?",
        "category": "FAQ",
        "content": (
            "Yes. YASH's AI agents are designed to integrate natively into established enterprise environments, including "
            "ERPs (like SAP and QAD), CRMs, supply chain systems, and finance platforms. This allows agents to pull context "
            "from your unique knowledge base and trigger actions across systems.\n"
            "Know more: https://www.neupac.ai/"
        ),
        "url": "https://www.neupac.ai/",
        "keywords": ["neupac integration", "enterprise integration", "sap integration", "qad", "crm integration",
                     "erp integration", "supply chain integration", "finance platform integration",
                     "existing systems", "native integration", "knowledge base"]
    },

    {
        "title": "FAQ — How does NEUPAC handle data security and privacy?",
        "category": "FAQ",
        "content": (
            "NEUPAC utilizes a zero-data-copy architecture. This means the AI agents can access, reason with, and utilize "
            "your enterprise data to execute tasks without moving, duplicating, or exposing the underlying data, greatly "
            "minimizing security risks.\n"
            "Know more: https://www.neupac.ai/"
        ),
        "url": "https://www.neupac.ai/",
        "keywords": ["data security faq", "privacy faq", "zero data copy faq", "data protection faq",
                     "neupac security", "how does neupac handle data", "data exposure", "security risks",
                     "duplicate data", "moving data"]
    },

    {
        "title": "FAQ — What is AI FinOps?",
        "category": "FAQ",
        "content": (
            "AI FinOps is a financial operations capability built into NEUPAC that provides cost accountability. "
            "As organizations scale AI usage, tracking compute and API costs becomes complex. AI FinOps gives leadership "
            "visibility and control to manage budgets effectively and ensure a high ROI.\n"
            "Know more: https://www.neupac.ai/"
        ),
        "url": "https://www.neupac.ai/",
        "keywords": ["ai finops faq", "what is ai finops", "finops explained", "cost accountability",
                     "compute costs", "api costs", "budget management", "roi finops", "llm spend tracking"]
    },

    {
        "title": "FAQ — What other AI platforms does YASH Technologies offer?",
        "category": "FAQ",
        "content": (
            "In addition to NEUPAC, YASH offers:\n"
            "- AMURAA: an overarching enterprise AI platform. Know more: https://www.yashtech.ai/platforms/amuraa\n"
            "- AMURAA Logistics: Know more: https://www.yashtech.ai/platforms/amuraa-logistics\n"
            "- Lingo: a specialized Customer Experience (CX) platform. Know more: https://www.yashtech.ai/platforms/lingo-by-yash-technologies"
        ),
        "url": "https://www.yashtech.ai/platforms/amuraa",
        "keywords": ["other platforms yash", "amuraa", "amuraa logistics", "lingo", "yash platforms",
                     "what platforms does yash offer", "cx platform", "enterprise ai platform yash",
                     "yash technologies platforms", "platform portfolio"]
    },

    {
        "title": "FAQ — What is Lingo?",
        "category": "FAQ",
        "content": (
            "Lingo is YASH's multilingual, sentiment-aware AI platform. It is designed to power smarter, highly personalized "
            "customer support, HR, IT, and sales operations by understanding both user intent and emotional sentiment in real-time.\n"
            "Know more: https://www.yashtech.ai/platforms/lingo-by-yash-technologies"
        ),
        "url": "https://www.yashtech.ai/platforms/lingo-by-yash-technologies",
        "keywords": ["lingo", "what is lingo", "lingo platform", "multilingual platform", "sentiment aware",
                     "customer support platform", "cx platform lingo", "personalized support",
                     "user intent", "emotional sentiment", "hr platform lingo", "sales platform lingo"]
    },

    {
        "title": "FAQ — How does Agentic AI transform Manufacturing and Supply Chain?",
        "category": "FAQ",
        "content": (
            "In manufacturing, YASH's Agentic AI moves operations from reactive monitoring to autonomous correction. "
            "Agents can dynamically optimize supply chain routing, balance workloads across plants, and execute predictive "
            "maintenance by identifying a machine fault and automatically scheduling the repair before downtime occurs.\n"
            "Know more:\n"
            "Manufacturing: https://www.yashtech.ai/ai-in-industries/manufacturing\n"
            "Supply Chain: https://www.yashtech.ai/services/ai-in-supply-chain"
        ),
        "url": "https://www.yashtech.ai/ai-in-industries/manufacturing",
        "keywords": ["agentic ai manufacturing", "supply chain agentic", "reactive to autonomous",
                     "supply chain routing", "workload balancing", "predictive maintenance agentic",
                     "machine fault", "downtime prevention", "autonomous correction", "plant operations"]
    },

    {
        "title": "FAQ — Can NEUPAC facilitate collaboration between AI agents?",
        "category": "FAQ",
        "content": (
            "Absolutely. NEUPAC excels at multi-agent orchestration. It facilitates seamless collaboration between various "
            "specialized AI agents (for example, an inventory agent talking to a procurement agent) to resolve intricate, "
            "cross-functional workflows.\n"
            "Know more: https://www.neupac.ai/"
        ),
        "url": "https://www.neupac.ai/",
        "keywords": ["agent collaboration", "multi-agent collaboration", "inventory agent", "procurement agent",
                     "cross-functional", "agent to agent", "specialized agents", "agent communication",
                     "can neupac collaborate", "agent workflows"]
    },

    {
        "title": "FAQ — What impact does NEUPAC have on Human Resources?",
        "category": "FAQ",
        "content": (
            "Agentic AI shifts HR from administrative processing to intelligent workforce orchestration. AI agents can "
            "autonomously handle complex onboarding workflows across multiple software systems, analyze workforce sentiment "
            "continuously, ensure global payroll compliance, and dynamically map personalized upskilling paths for employees.\n"
            "Know more: https://www.yashtech.ai/services/ai-in-human-resources"
        ),
        "url": "https://www.yashtech.ai/services/ai-in-human-resources",
        "keywords": ["hr impact", "neupac hr", "onboarding automation", "workforce sentiment", "payroll compliance",
                     "upskilling paths", "employee journey", "intelligent workforce", "hr orchestration",
                     "administrative processing", "hr transformation"]
    },

    {
        "title": "FAQ — How does NEUPAC ensure ethical and responsible AI?",
        "category": "FAQ",
        "content": (
            "Every YASH solution integrates bias detection, compliance checks, and explainability frameworks. "
            "The platform utilizes built-in guardrails — such as content filtering, PII redaction, and hallucination detection — "
            "ensuring that all autonomous actions are reliable, fair, and socially responsible.\n"
            "Know more: https://www.yashtech.ai/services/trusted-ai"
        ),
        "url": "https://www.yashtech.ai/services/trusted-ai",
        "keywords": ["ethical ai", "responsible ai faq", "bias detection", "explainability", "guardrails",
                     "content filtering", "pii redaction", "hallucination detection", "socially responsible",
                     "fair ai", "reliable ai", "autonomous actions ethical"]
    },

    {
        "title": "FAQ — What is YASH's recommended approach for implementing Agentic AI?",
        "category": "FAQ",
        "content": (
            "YASH recommends a 5-step methodology:\n"
            "1. Begin with clear business value and KPIs, not just algorithms.\n"
            "2. Build a unified, data-rich foundation.\n"
            "3. Design intelligent workflows with defined scopes of autonomy.\n"
            "4. Pilot in a controlled environment, learn, and scale gradually.\n"
            "5. Embed governance, continuous improvement, and human-machine collaboration.\n"
            "Start AI Transformation Journey: https://www.yashtech.ai/ai-survey"
        ),
        "url": "https://www.yashtech.ai/ai-survey",
        "keywords": ["implementation approach", "5 step", "five step", "methodology", "how to implement",
                     "business value kpi", "data foundation", "intelligent workflows", "pilot scale",
                     "governance continuous improvement", "human machine collaboration",
                     "ai transformation journey", "get started yash", "ai survey"]
    },

    {
        "title": "FAQ — How does NEUPAC handle global regulatory compliance?",
        "category": "FAQ",
        "content": (
            "NEUPAC features robust governance tools tailored to meet strict regulatory standards. It utilizes comprehensive "
            "auditing, role-based access controls (IAM policies), and model invocation logging to ensure that AI operations "
            "remain transparent and compliant with laws like the EU AI Act.\n"
            "Know more: https://www.neupac.ai/"
        ),
        "url": "https://www.neupac.ai/",
        "keywords": ["regulatory compliance", "eu ai act", "gdpr", "iam policies", "model invocation logging",
                     "transparent ai", "compliance tools", "global compliance", "governance tools",
                     "regulatory standards", "auditing compliance"]
    },

    {
        "title": "FAQ — Do AI agents learn and improve over time?",
        "category": "FAQ",
        "content": (
            "Yes. YASH's autonomous systems feature continuous learning mechanisms. They constantly refine their "
            "decision-making capabilities based on new data, changing circumstances, and operator feedback, improving "
            "their accuracy and efficiency over time.\n"
            "Know more: https://www.yashtech.ai/services/ai-life-cycle-management"
        ),
        "url": "https://www.yashtech.ai/services/ai-life-cycle-management",
        "keywords": ["learn over time", "improve over time", "continuous learning", "ai learning",
                     "decision making refine", "operator feedback", "accuracy improvement",
                     "efficiency improvement", "autonomous learning", "adaptive ai"]
    },

    {
        "title": "FAQ — What measurable business outcomes can organizations expect?",
        "category": "FAQ",
        "content": (
            "Organizations leveraging YASH's AI strategies have reported significant impact, including:\n"
            "- Up to 85% faster time-to-market\n"
            "- 40% cost reductions\n"
            "- 24/7 autonomous operations\n"
            "- Up to 30% increase in employee productivity by automating internal tasks.\n"
            "Know more: https://www.yashtech.ai/services/roi-with-ai"
        ),
        "url": "https://www.yashtech.ai/services/roi-with-ai",
        "keywords": ["business outcomes", "measurable outcomes", "85% faster", "40% cost reduction",
                     "24/7 autonomous", "30% productivity", "employee productivity", "time to market",
                     "roi outcomes", "expected results", "impact", "reported outcomes", "kpi results"]
    },

    {
        "title": "NEUPAC — Pricing and Contact",
        "category": "AI Platform",
        "content": (
            "For pricing related information, please reach out at info@neupac.com or click here to contact us: "
            "https://www.neupac.ai/contact-us/"
        ),
        "url": None,
        "keywords": [
            "pricing", "price", "cost", "how much", "cost of neupac", "neupac pricing",
            "neupac cost", "neupac price", "how much does neupac cost", "what is the pricing",
            "contact", "contact us", "contact neupac", "reach out", "get in touch",
            "who should i contact", "who to contact", "how to contact", "email",
            "info@neupac.com", "neupac email", "neupac contact", "question",
            "write to us", "support", "help", "inquiry", "enquiry"
        ]
    },

]

# ==============================================================================
# KEYWORD-BASED RETRIEVAL
# No embeddings — pure keyword + token overlap scoring.
# Three levels of matching:
#   Level 1 — Title token overlap        (weight ×3)
#   Level 2 — Category token overlap     (weight ×2)
#   Level 3a — Exact keyword phrase hit  (weight ×2)
#   Level 3b — Partial keyword token hit (weight ×1) catches paraphrases
# ==============================================================================

def _score_chunk(query: str, chunk: dict) -> float:
    """Score a single chunk against the user query."""
    query_lower = query.lower()
    query_tokens = set(re.split(r'\W+', query_lower)) - {"a", "an", "the", "is", "are", "was", "i", "me", "my", "we"}

    score = 0.0

    # Level 1: title token overlap
    title_tokens = set(re.split(r'\W+', chunk["title"].lower()))
    score += 3 * len(title_tokens & query_tokens)

    # Level 2: category token overlap
    category_tokens = set(re.split(r'\W+', chunk["category"].lower()))
    score += 2 * len(category_tokens & query_tokens)

    # Level 3a: exact keyword phrase match
    for kw in chunk.get("keywords", []):
        if kw.lower() in query_lower:
            score += 2

    # Level 3b: partial keyword token overlap
    # catches "bring MY own agent" vs keyword "bring your own agent"
    ignore = {"a", "an", "the", "of", "in", "for", "and", "or", "to", "with", "your", "my", "own", "our"}
    for kw in chunk.get("keywords", []):
        kw_tokens = set(re.split(r'\W+', kw.lower())) - ignore
        if len(kw_tokens) > 1 and len(kw_tokens & query_tokens) >= max(1, len(kw_tokens) - 1):
            score += 1

    return score


def get_relevant_chunks(query: str, top_k: int = 6) -> list:
    """Return top_k most relevant chunks for the query using keyword scoring."""
    scored = [(i, _score_chunk(query, chunk)) for i, chunk in enumerate(PROJECT_CHUNKS)]
    scored.sort(key=lambda x: x[1], reverse=True)

    # Only return chunks that scored > 0; if nothing matches, return top 3 anyway
    filtered = [(i, s) for i, s in scored if s > 0]
    top = filtered[:top_k] if filtered else scored[:3]

    print(f"[Retrieval] Query: '{query[:70]}' | Top matches:")
    for idx, sim in top:
        print(f"   {PROJECT_CHUNKS[idx]['title'][:60]} (score: {sim:.0f})")

    return [PROJECT_CHUNKS[i] for i, _ in top]


def format_chunks(chunks: list) -> str:
    """Format selected chunks into a clean context block for the LLM."""
    parts = []
    for chunk in chunks:
        url_line = f"\nURL: {chunk['url']}" if chunk.get("url") else ""
        parts.append(
            f"[{chunk['category']}] {chunk['title']}\n"
            f"{chunk['content']}"
            f"{url_line}"
        )
    return "\n\n---\n\n".join(parts)


# ==============================================================================
# LAMBDA ENTRY POINT
# ==============================================================================

def service_function(user_request):

    print("request------->", user_request)

    conversation = user_request["previous_data_list"][-6:]

    # Extract last user message for retrieval
    last_user_message = ""
    for msg in reversed(conversation):
        if msg.get("role") == "user":
            last_user_message = msg.get("content", "")
            break

    # Retrieve relevant chunks via keyword scoring
    relevant_chunks = get_relevant_chunks(last_user_message, top_k=6)
    context_block = format_chunks(relevant_chunks)

    full_system = (
        system_message
        + "\n\n--- RELEVANT KNOWLEDGE ---\n\n"
        + context_block
        + "\n\n--- END OF KNOWLEDGE ---"
    )

    messages = [{"role": "system", "content": full_system}] + conversation

    dataResponse = call_gpt_model(model=model_to_use, messages=messages, stream=False)

    response = dataResponse.text

    return response


# ==============================================================================
# LAMBDA HANDLER — entry point AWS Lambda calls directly
# ==============================================================================

def lambda_handler(event, context):

    print("event---------->", event)

    data = json.loads(event["body"])

    response = service_function(data)

    return {
        'statusCode': 200,
        'body': json.dumps(response)
    }