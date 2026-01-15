# 🚀 Development Journey - RAG Chatbot

A brutally honest chronicle of building this RAG system from scratch.

---

## 🎯 The Vision

**Goal:** Build a multimodal RAG chatbot that handles text, tables, and images from a 60-page Agentic AI PDF.

**Reality Check:** Ended up with a solid text-based RAG because multimodal is a fucking rabbit hole.

---

## 📖 Chapter 1: The Beginning - Embedding Hell

### Phase 1: Initial Approach (The Naive Era)
**Decision:** Let's start simple with Sentence Transformers

```python
# First attempt
embeddings = HuggingFaceEmbeddings(
    model_name='sentence-transformers/all-MiniLM-L6-v2'
)
```

**Problem:** 
- Shit results on tables
- Poor semantic understanding
- Dimension: 384 (kinda small)
- Not great for complex technical content

**Lesson Learned:** Free local models ≠ good results for specialized content

---

### Phase 2: The Gemini Shift
**Decision:** Fuck it, use Gemini Embedding-001

```python
embeddings = GoogleGenerativeAIEmbeddings(
    model='models/embedding-001',
    google_api_key=os.getenv("GOOGLE_API_KEY")
)
```

**Benefits:**
- Dimension: 768 (better)
- Actually understands technical terms
- Handles Agentic AI jargon properly
- Much better table comprehension

**Trade-off:** API costs (but worth it)

---

## 📖 Chapter 2: The Chunking Saga

### Initial Setup
```python
# First naive approach
chunk_size = 500  # Too small, lost context
chunk_overlap = 100  # Not enough continuity
```

**Problem:** Generated 119 embedding chunks, but answers were fragmented and missing connections between concepts.

### The Fix
```python
splitter = RecursiveCharacterTextSplitter(
    chunk_size=1000,      # Sweet spot
    chunk_overlap=200,    # Better context preservation
    separators=["\n\n", "\n", " ", ""]
)
```

**Why this works:**
- 1000 chars ≈ 1-2 paragraphs
- 200 overlap ensures concept continuity
- Recursive splitting respects natural boundaries

**Result:** ~500-800 chunks with better semantic coherence

---

## 📖 Chapter 3: Rate Limiting Nightmare

### The Problem
```python
# Naive batch upload
for chunk in chunks:
    vector_store.add_documents([chunk])  # BOOM! Rate limit
```

**Error:** 
```
RateLimitError: You exceeded your quota
```

### The Solution
```python
batch_size = 20
for i in range(0, len(chunks), batch_size):
    batch = chunks[i:i + batch_size]
    vector_store.add_documents(batch)
    time.sleep(2)  # The magic sauce
```

**Lesson:** Always batch + sleep when dealing with free tiers

**Alternative Tried:**
```python
time.sleep(5)  # Too slow
time.sleep(1)  # Still hit limits
time.sleep(2)  # Goldilocks zone
```

---

## 📖 Chapter 4: Retrieval Engineering

### Initial Retriever
```python
# Basic similarity search
docs = vector_store.similarity_search(query, k=5)
```

**Problem:** No scores, no metadata, no insights into retrieval quality

### Enhanced Retriever
```python
docs_with_scores = vector_store.similarity_search_with_score(query, k=8)

# Added metadata
for doc, score in docs_with_scores:
    metadata = {
        'score': score,
        'page': doc.metadata.get('page'),
        'source': doc.metadata.get('source')
    }
```

**Why k=8?**
- Tried k=5: Too few, missed relevant info
- Tried k=12: Too many, added noise
- k=8: Just right, good signal-to-noise ratio

---

## 📖 Chapter 5: Prompt Engineering Hell

### Version 1: The Naive Prompt
```python
prompt = f"Answer this: {question}\nContext: {context}"
```

**Problem:** 
- Hallucinations everywhere
- Ignored context
- Made up page numbers

### Version 2: The Strict Prompt
```python
RAG_PROMPT = """
You are an expert. Answer ONLY from context.
Do not use outside knowledge.
If answer not in context, say so.

Context: {context}
Question: {question}
"""
```

**Better, but:** Still verbose, no page citations

### Version 3: The Final Form
```python
RAG_PROMPT = """
You are an expert AI assistant specialized in Agentic AI. 
Answer based on provided context.

**REASONING INSTRUCTIONS:**
1. Synthesize information from different context parts
2. Ground all facts in provided text
3. If answer cannot be derived, say so explicitly
4. Output ONLY the answer with page numbers
5. Quote relevant text snippets in ""

Context: {context}
Question: {question}

ANSWER:
"""
```

**Changes Made:**
- ✅ Added page number requirement
- ✅ Added quote extraction
- ✅ Added synthesis instruction
- ✅ Added complexity analysis capability
- ✅ Clearer output format

**Iterations:** ~6-7 prompt versions before landing on this

---

## 📖 Chapter 6: Hyperparameter Tuning

### Temperature Wars

#### Attempt 1: Creative Mode
```python
temperature = 0.7
```
**Result:** Too creative, started hallucinating "facts"

#### Attempt 2: Deterministic Mode
```python
temperature = 0.0
```
**Result:** Too robotic, lost nuance

#### Attempt 3: The Goldilocks Zone
```python
temperature = 0.1
```
**Result:** Perfect balance - factual but natural

**Final Config:**
```python
chat_model = ChatOpenAI(
    model='google/gemini-2.5-flash',
    max_tokens=4000,
    temperature=0.1,  # The magic number
)
```

### Other Hyperparameters Tested

**top_k (retrieval):**
- 4: Too few
- 12: Too many
- **8: Perfect**

**max_tokens:**
- 2000: Too short for complex answers
- 8000: Overkill, slower
- **4000: Just right**

**chunk_overlap:**
- 100: Lost context
- 300: Too much redundancy
- **200: Sweet spot**

---

## 📖 Chapter 7: The Multimodal Dream (And Nightmare)

### The Ambitious Plan
```
Goal: Extract and understand images, tables, and flowcharts
Tool: Unstructured library
Vision: Gemini Vision / LLaVA
```

### Phase 1: Unstructured Setup
```python
elements = partition_pdf(
    filename=pdf_path,
    strategy="hi_res",
    extract_images_in_pdf=True
)
```

**Initial Joy:** 
```
✅ Text extracted
✅ Tables extracted
✅ Images extracted
```

### Phase 2: The Image Explosion
```
Extracting images...
Progress: 10/80+ images
Progress: 20/80+ images
...
```

**Reality Check:**
- 80+ images extracted from 60-page PDF
- Each image needs Vision LLM description
- Gemini Vision: ~2-3 seconds per image
- **Total time: 4-6 minutes just for images**

### Phase 3: Cost Analysis
```python
# Gemini Vision pricing (approx)
images = 80
cost_per_image = $0.001  # rough estimate
total_cost = $0.08 per run

# Plus embedding costs for descriptions
embedding_cost = $0.02 per run

# Total: ~$0.10 per ingestion
```

**For a free assignment:** Not worth it

### Phase 4: Alternative Considered - LLaVA
```python
# Local vision model
model = "llava-v1.5-7b"
```

**Pros:**
- Free
- Local inference
- No API limits

**Cons:**
- Requires GPU (don't have it)
- Slower than Gemini Vision
- Quality not as good for diagrams
- Setup complexity

**Decision:** Abort multimodal mission

### Phase 5: The Retreat
**Final Call:** Stick with text-only RAG

**Reasoning:**
1. 80% of value is in text
2. Tables handled decently by text extraction
3. Diagrams can be inferred from surrounding text
4. Time/cost not justified for assignment
5. Text RAG working beautifully

**Lesson:** Perfect is the enemy of good

---

## 📖 Chapter 8: Behavior Analysis & Testing

### Test Suite
```python
hard_mode_queries = [
    "Explain the BDI model",
    "What is Emergence AI vs Konverge AI?",
    "Retail Copilot impact metrics?",
]
```

### Results Analysis

**Query 1: BDI Model**
```
Confidence: 0.74
Quality: Excellent
Retrieved correct pages: 21, 19, 32
```

**Query 2: Company Roles**
```
Confidence: 0.76
Quality: Perfect
Distinguished between entities correctly
```

**Query 3: Specific Metrics**
```
Confidence: 0.72
Quality: Good
Found exact numbers: 25% engagement, 15% conversion
```

### Behavioral Observations

**What Works:**
- ✅ Technical concept explanations
- ✅ Entity disambiguation
- ✅ Specific metric retrieval
- ✅ Multi-hop reasoning (when chunks overlap)

**What's Meh:**
- ⚠️ Flowchart/diagram questions (no visual understanding)
- ⚠️ Questions requiring info from multiple disconnected sections
- ⚠️ Very specific table lookups

**What Fails:**
- ❌ Off-topic questions (correctly refuses)
- ❌ Contradictory info (but detects it)
- ❌ Ambiguous queries without context

---

## 📖 Chapter 9: Temperature Impact Study

### Experiment Setup
```python
test_query = "Explain the BDI model"
temperatures = [0.0, 0.1, 0.3, 0.5, 0.7, 1.0]
```

### Results

**Temperature: 0.0**
```
Response: "The BDI model is a model. Agents have beliefs. Agents have desires..."
Issue: Robotic, repetitive, lacks flow
```

**Temperature: 0.1** ⭐
```
Response: "The BDI model characterizes agents by their beliefs..."
Quality: Natural, accurate, good flow
```

**Temperature: 0.3**
```
Response: Similar to 0.1 but occasionally adds unnecessary elaboration
Issue: Sometimes strays slightly from context
```

**Temperature: 0.7**
```
Response: Started making connections not in the text
Issue: Hallucination risk increases
```

**Winner: 0.1** - Perfect balance of accuracy and naturalness

---

## 📖 Chapter 10: Future Improvements (TODO)

### Planned: PyMuPDF Migration
```python
# Current: PyPDF
from pypdf import PdfReader

# Future: PyMuPDF (fitz)
import fitz
```

**Why switch?**
- Better table extraction
- Better text positioning
- Better metadata preservation
- Faster processing

**Why not done yet?**
- Current solution works
- Assignment deadline pressure
- "Not that deep bro"

### Other Potential Improvements

1. **Better Chunking**
   ```python
   # Semantic chunking instead of character-based
   # Chunk by topics/sections
   ```

2. **Re-ranking**
   ```python
   # Add Cohere re-ranker or similar
   # Boost relevance of retrieved chunks
   ```

3. **Query Expansion**
   ```python
   # Expand user query with synonyms
   # Better retrieval for varied phrasing
   ```

4. **Hybrid Search**
   ```python
   # Combine vector search + keyword search
   # Better for specific terms/names
   ```

5. **Caching**
   ```python
   # Cache frequent queries
   # Faster response times
   ```

---

## 🎯 Final Stats

### What We Built
```
- Lines of Code: ~800
- Files: 5 core files
- Dependencies: 12 packages
- Ingestion Time: ~10 minutes
- Query Time: ~3-5 seconds
- Confidence Score: 0.65-0.85 avg
- Chunks Stored: ~500-800
- Vector Dimensions: 768
```

### Time Investment
```
- Research: 2 hours
- Initial setup: 1 hour
- Embedding iterations: 3 hours
- Prompt engineering: 2 hours
- Multimodal attempt: 4 hours (wasted)
- Testing & tuning: 3 hours
- Documentation: 1 hour
- Total: ~16 hours
```

### What I Learned

**Technical:**
- RAG pipeline architecture
- Vector embeddings (local vs API)
- LangGraph state machines
- Prompt engineering matters A LOT
- Temperature tuning is crucial
- Rate limiting is real

**Practical:**
- Don't overcomplicate
- Text-only RAG is powerful enough
- Multimodal is cool but overkill
- Free tiers have limits (respect them)
- Testing >> features

**Philosophy:**
- Perfect is the enemy of good
- Ship working solutions
- Document your failures
- "Not that deep bro" is valid engineering

---

## 🙏 Acknowledgments

**To the homies:**
- Claude AI for putting up with my shit
- Gemini for not rate-limiting me to death
- Pinecone for free tier
- Stack Overflow for existing

**To myself:**
- For not giving up during multimodal hell
- For realizing when to pivot
- For writing this journey doc

---

## 📝 Final Thoughts

This RAG chatbot is:
- ✅ Production-ready (for assignment)
- ✅ Well-documented
- ✅ Actually works
- ✅ Not overengineered

Could it be better? Yes.
Is it good enough? Fuck yes.

**Mission Status:** ✅ ACCOMPLISHED

---

*Built with blood, sweat, and a lot of `time.sleep(2)` calls* 🔥
