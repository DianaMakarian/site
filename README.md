# DOE4230: Exoplanet Discovery Platform 🔭

**DOE4230** is a web-based educational platform that utilizes **real astrophysical data** and **Machine Learning (ML)** to teach users how scientists detect exoplanets using the **transit method**.

The project transforms complex astrophysical data into an approachable, interactive interface, aiming to cultivate the next generation of space scientists by making real discovery engaging and accessible.

---

## ✨ Project Functionality and Mechanics

The platform guides users through the exoplanet detection workflow, mirroring the process used by professional astronomers by allowing them to test hypotheses and adjust parameters in real-time.

We feature **two primary modes**, each powered by a different ML model to cater to different user proficiency levels:

### 🌟 Beginner Mode: The Smart Eye (Simple CNN)

* **Model:** Simple **Convolutional Neural Network (CNN)**.
* **Focus:** Quick signal-to-noise distinction.
* **How it Works:** Users adjust physical parameters (orbital, transit, planet, and stellar characteristics) to **"create their exoplanet."** The model acts as a quick, trained eye, providing an instant verdict on the probability of the object being an exoplanet candidate. This helps beginners immediately distinguish between likely planetary signals and noise/errors.

### 🔬 Advanced Mode: Stacking Ensemble

* **Model:** Sophisticated **Stacking Ensemble** of ML models.
* **Focus:** Superior predictive accuracy for robust classification.
* **How it Works:** Designed for seasoned users and researchers, it demands a richer dataset (detailed light curves, radial velocity, multi-wavelength observations). The model delivers a precise classification:
    * **False Positive** (noise or artifact)
    * **Candidate** (promising signal warranting follow-up)
    * **Confirmed Exoplanet** (robust planetary evidence)

---

## 🚀 Benefits and Intended Impact

The **Universe is vast**, yet we've confirmed only around 7,000 exoplanets. The current pace is too slow. The main impact of DOE4230 is **science outreach** and the **cultivation of future scientific talent** by accelerating the rate of discovery.

* **Inspiring the Future:** We aim to turn "curious beginners" into **"junior scientists"** by blending the authenticity of working with **real NASA Kepler mission data** with the motivational mechanics of **gamification** (achievements and progress tracking).
* **Accessible Learning:** A **clean, intuitive UI** minimizes the feeling of being overwhelmed, providing immediate, accurate analysis and customizable data input for experimental flexibility.

---

## 🛠️ Tools and Technology Stack

The project’s creativity lies in merging **scientific rigor with playful learning mechanics**, turning scientific training into an engaging journey.

### 💻 Development Tools

| Category | Tools/Libraries |
| :--- | :--- |
| **ML/Backend** | Flask, Torch, Scikit-learn, CatBoost |
| **Data Analysis** | NumPy, Pandas, Matplotlib |
| **Data Sources** | Kepler Objects of Interest (KOI) |
| **Platform/Frontend** | Web-based interface (HTML, CSS, JavaScript) |

### 🎨 Creative Approach

We introduced **hidden achievements and gamified milestones** that reward curiosity, not just correct answers. DOE4230 transforms real exoplanet research into an accessible and inspiring experience, proving that science can be both serious and fun.

---

## 🗺️ Future Development

Our platform will continue to evolve in three key directions to ensure it remains both a leading learning journey and a valuable scientific aid.

1.  **Educational Enrichment:**
    * Interactive visualizations of the transit method.
    * Step-by-step tutorials.
    * A new **Scientist Level** to guide users from beginner to advanced analysis mastery.
2.  **Gamification & Engagement:**
    * More achievements, challenges, and exclusive rewards (articles, interactives, themes) to sustain curiosity and motivation.
3.  **Research Utility:**
    * Integration of new datasets like **TESS**.
    * **Confidence scoring** for model predictions, making the tool a valuable assistant for astronomers handling massive data streams.
