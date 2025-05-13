# Pictora: Caption Your Dream Picture  

## **Overview**  
Imagine an AI that turns visuals into vivid stories—**Pictora** does just that! This cutting-edge platform automates image captioning using **self-supervised learning (SSL)** and **transformer-based architectures**, eliminating the drudgery of manual labeling. By generating high-quality, context-aware descriptions without relying on human-annotated data, Pictora empowers creators and enterprises to scale their workflows effortlessly. Think of it as your AI co-pilot for visual storytelling!  

---

## **Why This Approach Rocks**  
The magic lies in blending pre-trained models like **YOLO** (for object detection) and **BLIP/CLIP** (for scoring) to create pseudo-captions—think of these as AI-generated "first drafts." These raw captions are refined through BLIP/CLIP scoring (filtering low-quality outputs) and polished by **Gemini API** for grammar perfection. This isn’t just clever—it’s validated by SSL research like [*Self-Supervised Image Captioning with Pseudo-Labels* (Zhang et al., 2025)](https://arxiv.org/abs/2504.08531).  

### **Key Innovations**  
- **Transformer Decoder Magic**: Cross-attention mechanisms align MobileNetV3 image features with text, ensuring captions *get* the context.  
- **Effortless Scaling**: Caches CNN features to slash training time, while early stopping keeps things snappy.  
- **Real-World Ready**: Integrates into a web app with freemium tiers (filtered pseudo-captions) and premium plans (direct model inference).  

---

### **✨ Core Features**  
- **AI-Powered Automation**:  
  - **Self-Supervised Learning**: Trains on pseudo-labeled data—no human annotation needed.  
  - **Transformer Decoder**: Combines MobileNetV3’s vision with cross-attention for spot-on captions.  
  - **Quality Control**: BLIP/CLIP scoring + Gemini API = error-free, polished results.  

- **Freemium & API Access**:  
  - **Free Tier**: 5 captions every 5 minutes—perfect for casual users.  
  - **Premium Plans**: Up to 20k credits/month, batch processing, and multilingual support.  
  - **API Integration**: Plug-and-play REST API for developers!  

- **Batch Processing**: Upload ZIP files, get JSON captions—ideal for scaling.  
- **Secure Payments**: Seamlessly pay via eSewa (Nepal), with role-based access control.  

---

## **How It Works: From Pixels to Poetry**  

### **1. Model Training Pipeline**  
#### **a. Data Prep Like a Pro**  
- **Pseudo-Captions**: Generated from `pseudo_caption.txt` using YOLO + rule-based templates.  
- **Preprocessing**: Images resized to 224×224, tokenized with `[START]`/`[END]` markers.  
- **Caching Wins**: MobileNetV3 features stored on disk—no redundant work!  

#### **b. Transformer Architecture**  
- **Feature Extractor**: MobileNetV3 (frozen) captures spatial details.  
- **Decoder Wizardry**: Causal self-attention for smooth text flow + cross-attention to spotlight key image regions.  
- **Smart Training**: Masked loss ignores padding tokens; rare tokens (like `[UNK]`) get penalized.  

#### **c. Inference Like a Boss**  
- **Freemium Mode**: Generates 50 captions, picks the best via BLIP/CLIP, then polishes with Gemini.  
- **Premium Mode**: Direct model inference with credit-based quotas.  

---

### **2. Web App Workflow**  
#### **User Journeys**  
1. **Upload**: Drag/drop images or ZIP files.  
2. **Caption Generation**:  
   - Free users: 5 requests/5 mins.  
   - Premium users: Deduct credits from MongoDB balance.  
3. **Output**: Captions with metadata (model used, remaining quota).  

#### **Tech Behind the Scenes**  
- **FastAPI Backend**: Handles auth, payments, and inference.  
- **MongoDB**: Secure storage for user data, encrypted images.  
- **Security First**: Fernet encryption, rate limiting, CSRF protection.  

#### **Payment System**  
- **Freemium**: 5 free requests every 5 mins.  
- **Premium Plans**:  
  - **Starter Plan (₹500/month)**: 6.5k credits.  
  - **Pro Plan (₹1500/month)**: 20k credits + batch processing.  

---

## **Technical Brilliance Under the Hood**  
- **Attention Mechanisms**: Self-attention for context, cross-attention for image-text alignment.  
- **Tokenization**: Custom text standardization handles punctuation/punctuation like a pro.  
- **Optimized Training**: Adam optimizer (`lr=1e-4`), masked loss, early stopping.  

### **Code Structure**  
- **Frontend**: HTML/CSS/JS with drag-and-drop flair.  
- **Backend**: FastAPI + MongoDB for scalability.  
- **Inference Modules**: `freemiumApp` (BLIP/CLIP + Gemini) vs. `premiumApp` (direct model use).  

### **Scalability Wins**  
- **Caching & Parallelism**: Thread pooling speeds up batch processing.  
- **Rate Limiting**: Prevents abuse (memory-based, implied in code).  

---

## **Conclusion**  
Pictora isn’t just another AI tool—it’s a bridge between groundbreaking research and real-world impact. By ditching labeled data and embracing SSL, it democratizes access to advanced captioning for creators and businesses alike. Whether you’re a solo blogger or a Fortune 500 team, Pictora scales with you, turning pixels into prose with unmatched efficiency.  

## **Live Demo**  
Try it yourself at [**pictora.co.in**](https://pictora.co.in) —see AI-generated captions and batch processing in action!  

**License:** MIT License—free to use, adapt, and innovate!  

---  
*Ready to transform your images into stories? Pictora’s got your back!* ✨
